from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from eppy.modeleditor import IDF

from rom_automation.epconfig.loader import load_idf_edit_config
from rom_automation.mpc.config_types import MpcConfig
from rom_automation.perturbation.idf_editor import IDFEditor


# Names of the objects the preparer injects into the run IDF. The equipment name
# is also the actuator key used by the pyenergyplus harness.
MPC_EQUIPMENT_NAME = "MPC HVAC Injection"
MPC_SCHEDULE_NAME = "MPC Always On"
MPC_TYPELIMITS_NAME = "MPC Any Number"
ZONE_AIR_TEMP_VARIABLE = "Zone Mean Air Temperature"

# Wide-deadband setpoints used to neutralize the principal HVAC when the MPC is
# the sole conditioner. These are far outside any realistic zone temperature, so
# the native thermostat never calls for heating or cooling.
MPC_WIDE_HEAT_SP_SCHEDULE = "MPC Heating Setpoint Wide"
MPC_WIDE_COOL_SP_SCHEDULE = "MPC Cooling Setpoint Wide"
MPC_WIDE_HEAT_SP_C = -60.0
MPC_WIDE_COOL_SP_C = 60.0


@dataclass(frozen=True)
class PreparedIdf:
    """
    Result of preparing the run IDF for a closed-loop MPC run.
    """

    run_idf_path: Path
    zone_name: str
    equipment_name: str


class IdfPreparer:
    """
    Ensures an MPC-folder IDF exists (copying from the raw folder if missing) and
    builds a run-specific IDF in two stages:

      1. Run the standard `IDFEditor` (from the idf_edit module) to bring the IDF
         to the same runnable state as the processed IDF: rewrite Schedule:File
         paths to the local schedule dir, set setpoints, timestep, and output
         variables (which include Zone Mean Air Temperature).
      2. Overlay MPC-specific changes: override the RunPeriod to the closed-loop
         window and the Timestep to the control step, and add an actuated
         OtherEquipment object for direct thermal-power injection.

    The base MPC IDF is never mutated; edits are written to `run_idf_path`.
    """

    def __init__(self, idd_path: str | Path) -> None:
        self.idd_path = Path(idd_path)
        if not self.idd_path.exists():
            raise FileNotFoundError(
                f"EnergyPlus IDD not found: {self.idd_path}. Check "
                "energyplus.install_dir in the config."
            )
        # IDFEditor sets the IDD name (once per process); reuse it so we don't
        # double-set it here.
        self._editor = IDFEditor(idd_path=self.idd_path)

    def ensure_base_idf(self, cfg: MpcConfig) -> Path:
        """
        Return the base MPC IDF path, copying it from the raw IDF folder if it
        does not already exist under `mpc_idf_root`.
        """
        base_path = cfg.base_mpc_idf_path()
        if base_path.exists():
            return base_path

        raw_path = cfg.raw_idf_path()
        if not raw_path.exists():
            raise FileNotFoundError(
                f"MPC IDF does not exist at {base_path} and the raw source IDF "
                f"to copy from was not found at {raw_path}."
            )

        base_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_path, base_path)
        return base_path

    def prepare_run_idf(self, cfg: MpcConfig, run_idf_path: str | Path) -> PreparedIdf:
        """
        Build the run IDF from the (ensured) base MPC IDF and return metadata
        needed by the plant harness.
        """
        idf, run_idf_path, zone_name = self._prepare_common(cfg, run_idf_path)

        if cfg.is_direct_power:
            # Direct injection: add the actuated OtherEquipment and (optionally)
            # take the principal HVAC out of the loop.
            self._add_injection_equipment(idf, zone_name)
            if cfg.disable_native_hvac:
                self._widen_thermostat_deadband(idf)
        else:
            # Supervisory: the principal HVAC is the actuator (its setpoints are
            # overridden live), so it must stay enabled. Confirm the thermostat
            # the setpoint actuator targets actually exists.
            self._require_dual_setpoint(idf)

        idf.saveas(str(run_idf_path))

        return PreparedIdf(
            run_idf_path=run_idf_path,
            zone_name=zone_name,
            equipment_name=MPC_EQUIPMENT_NAME,
        )

    def prepare_calibration_idf(
        self,
        cfg: MpcConfig,
        run_idf_path: str | Path,
    ) -> PreparedIdf:
        """
        Build an IDF for the HVAC capacity-calibration pass: the principal HVAC
        stays fully enabled (no deadband widening, no injection) and its setpoint
        is overridden live to a very low value to force full-tilt cooling. Reuses
        the supervisory-style thermostat requirement.
        """
        idf, run_idf_path, zone_name = self._prepare_common(cfg, run_idf_path)
        self._require_dual_setpoint(idf)
        idf.saveas(str(run_idf_path))
        return PreparedIdf(
            run_idf_path=run_idf_path,
            zone_name=zone_name,
            equipment_name=MPC_EQUIPMENT_NAME,
        )

    def _prepare_common(
        self,
        cfg: MpcConfig,
        run_idf_path: str | Path,
    ) -> tuple[IDF, Path, str]:
        """
        Shared preparation: ensure the base IDF, run the standard idf_edit pass
        (schedule paths, setpoints, timestep, outputs), then apply the common
        MPC edits (run period, timestep, zone temperature output). Returns the
        open IDF (not yet saved), the resolved run path, and the zone name.
        """
        base_path = self.ensure_base_idf(cfg)
        run_idf_path = Path(run_idf_path)
        run_idf_path.parent.mkdir(parents=True, exist_ok=True)

        edit_config = load_idf_edit_config(cfg.paths.idf_edit_config)
        self._editor.edit_idf(
            input_path=base_path,
            output_path=run_idf_path,
            config=edit_config,
        )

        idf = IDF(str(run_idf_path))
        zone_name = self._resolve_zone_name(idf, cfg.selection.zone_name)

        self._edit_run_period(idf, cfg)
        self._edit_timestep(idf, cfg)
        self._add_zone_temp_output(idf, zone_name, cfg)

        return idf, run_idf_path, zone_name

    # ------------------------------------------------------------------ #
    # Zone resolution
    # ------------------------------------------------------------------ #

    def _resolve_zone_name(self, idf: IDF, requested: str | None) -> str:
        zone_names = [str(z.Name).strip() for z in idf.idfobjects["ZONE"]]

        if requested is not None:
            match = [z for z in zone_names if z.lower() == requested.strip().lower()]
            if not match:
                raise ValueError(
                    f"selection.zone_name {requested!r} not found in IDF. "
                    f"Available zones: {sorted(zone_names)}"
                )
            return match[0]

        # Auto-detect: prefer the zone referenced by a ZoneControl:Thermostat.
        thermostats = idf.idfobjects["ZONECONTROL:THERMOSTAT"]
        controlled = [
            str(t.Zone_or_ZoneList_Name).strip()
            for t in thermostats
            if str(t.Zone_or_ZoneList_Name).strip()
        ]
        # Keep only entries that are actual zones (not zone lists).
        controlled_zones = [z for z in controlled if z in zone_names]

        if len(controlled_zones) == 1:
            return controlled_zones[0]

        if not controlled_zones and len(zone_names) == 1:
            return zone_names[0]

        raise ValueError(
            "Could not unambiguously auto-detect the conditioned zone. "
            f"Thermostat-controlled zones: {sorted(set(controlled_zones))}; "
            f"all zones: {sorted(zone_names)}. "
            "Set selection.zone_name explicitly in the config."
        )

    # ------------------------------------------------------------------ #
    # IDF edits
    # ------------------------------------------------------------------ #

    def _edit_run_period(self, idf: IDF, cfg: MpcConfig) -> None:
        run_periods = idf.idfobjects["RUNPERIOD"]
        if not run_periods:
            raise ValueError("No RunPeriod object found in IDF.")

        start = cfg.window.start
        # RunPeriod is day-granular; cover the calendar days the window touches.
        # The window is [start, start + duration); its last touched day is the
        # day containing (end - 1s).
        last_instant = cfg.window.end - timedelta(seconds=1)

        rp = run_periods[0]
        rp.Begin_Month = start.month
        rp.Begin_Day_of_Month = start.day
        rp.Begin_Year = start.year
        rp.End_Month = last_instant.month
        rp.End_Day_of_Month = last_instant.day
        rp.End_Year = last_instant.year

        # Drop any extra RunPeriod objects so only our window runs.
        for extra in run_periods[1:]:
            idf.removeidfobject(extra)

    def _edit_timestep(self, idf: IDF, cfg: MpcConfig) -> None:
        timesteps = idf.idfobjects["TIMESTEP"]
        if not timesteps:
            idf.newidfobject(
                "TIMESTEP",
                Number_of_Timesteps_per_Hour=cfg.control.timesteps_per_hour,
            )
            return
        timesteps[0].Number_of_Timesteps_per_Hour = cfg.control.timesteps_per_hour

    def _add_injection_equipment(self, idf: IDF, zone_name: str) -> None:
        """
        Add an always-on OtherEquipment object whose Power Level is actuated live
        by the MPC. Fuel Type 'None' => pure sensible gain, no fuel meter. All of
        the injected power goes to the zone air as convective sensible heat
        (Fraction Latent/Radiant/Lost = 0). Negative actuator values remove heat
        (cooling).
        """
        self._ensure_type_limits(idf)
        self._ensure_always_on_schedule(idf)

        idf.newidfobject(
            "OTHEREQUIPMENT",
            Name=MPC_EQUIPMENT_NAME,
            Fuel_Type="None",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name=MPC_SCHEDULE_NAME,
            Design_Level_Calculation_Method="EquipmentLevel",
            Design_Level=0.0,
            Fraction_Latent=0.0,
            Fraction_Radiant=0.0,
            Fraction_Lost=0.0,
            EndUse_Subcategory="MPC",
        )

    def _ensure_type_limits(self, idf: IDF) -> None:
        existing = {
            str(t.Name).strip().lower()
            for t in idf.idfobjects["SCHEDULETYPELIMITS"]
        }
        if MPC_TYPELIMITS_NAME.lower() in existing:
            return
        idf.newidfobject(
            "SCHEDULETYPELIMITS",
            Name=MPC_TYPELIMITS_NAME,
            Numeric_Type="Continuous",
        )

    def _ensure_always_on_schedule(self, idf: IDF) -> None:
        self._ensure_constant_schedule(idf, MPC_SCHEDULE_NAME, 1.0)

    def _ensure_constant_schedule(self, idf: IDF, name: str, value: float) -> None:
        for s in idf.idfobjects["SCHEDULE:CONSTANT"]:
            if str(s.Name).strip().lower() == name.lower():
                s.Hourly_Value = value
                return
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name=name,
            Schedule_Type_Limits_Name=MPC_TYPELIMITS_NAME,
            Hourly_Value=value,
        )

    def _widen_thermostat_deadband(self, idf: IDF) -> None:
        """
        Repoint every dual-setpoint thermostat to extreme constant heating/cooling
        setpoints so the principal HVAC never activates, leaving the MPC-injected
        power as the sole conditioner. Overriding the schedule *names* on the
        thermostat is robust to whatever schedule type the setpoints originally
        used (Schedule:File, Day:Interval, etc.).
        """
        self._ensure_type_limits(idf)
        self._ensure_constant_schedule(idf, MPC_WIDE_HEAT_SP_SCHEDULE, MPC_WIDE_HEAT_SP_C)
        self._ensure_constant_schedule(idf, MPC_WIDE_COOL_SP_SCHEDULE, MPC_WIDE_COOL_SP_C)

        duals = self._require_dual_setpoint(idf)
        for d in duals:
            d.Heating_Setpoint_Temperature_Schedule_Name = MPC_WIDE_HEAT_SP_SCHEDULE
            d.Cooling_Setpoint_Temperature_Schedule_Name = MPC_WIDE_COOL_SP_SCHEDULE

    def _require_dual_setpoint(self, idf: IDF) -> list:
        duals = idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
        if not duals:
            raise ValueError(
                "No ThermostatSetpoint:DualSetpoint object was found. The MPC "
                "supervisory setpoint actuator and the direct-mode deadband "
                "widening both require a dual-setpoint thermostat. Inspect the "
                "IDF's thermostat setup."
            )
        return list(duals)

    def _add_zone_temp_output(
        self,
        idf: IDF,
        zone_name: str,
        cfg: MpcConfig,
    ) -> None:
        # Avoid duplicating an identical request if the base IDF already has one.
        for obj in idf.idfobjects["OUTPUT:VARIABLE"]:
            same_var = str(obj.Variable_Name).strip().lower() == ZONE_AIR_TEMP_VARIABLE.lower()
            key = str(obj.Key_Value).strip().lower()
            if same_var and key in (zone_name.lower(), "*"):
                return

        idf.newidfobject(
            "OUTPUT:VARIABLE",
            Key_Value=zone_name,
            Variable_Name=ZONE_AIR_TEMP_VARIABLE,
            Reporting_Frequency="Timestep",
        )
