from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional


ZONE_AIR_TEMP_VARIABLE = "Zone Mean Air Temperature"
ZONE_SENSIBLE_HEATING_RATE = "Zone Air System Sensible Heating Rate"
ZONE_SENSIBLE_COOLING_RATE = "Zone Air System Sensible Cooling Rate"


@dataclass(frozen=True)
class PlantObservation:
    """
    What the plant reports to the controller each control step.

    - `t_in_c`         : measured zone air temperature (the state x0's T_in).
    - `hvac_power_kw`  : sensible power the *principal* HVAC delivered to the zone
                         over the previous step (heating - cooling), kW signed.
                         ~0 in direct_power mode (native HVAC disabled); the real
                         heat-pump power in supervisory_setpoint mode. Used as the
                         4R2C observer's power input in supervisory mode.
    """

    t_in_c: float
    hvac_power_kw: float


# A step function: given (interval_start_datetime, PlantObservation) return the
# control command for the interval [t, t+dt), or None to apply nothing (outside
# the acting window). The command's meaning depends on the actuation mode:
#   direct_power         -> HVAC thermal power in kW (signed: + heating, - cooling)
#   supervisory_setpoint -> target zone-temperature setpoint in degC
StepFn = Callable[[datetime, PlantObservation], Optional[float]]


class EnergyPlusPlant:
    """
    Closed-loop EnergyPlus harness built on the pyenergyplus runtime API.

    Each zone timestep (after the heat balance is initialized) it reads the zone
    air temperature and the principal HVAC's delivered sensible power, asks the
    supplied `step_fn` for a command, and applies it via the configured actuation
    strategy:

    - "direct_power":         write the command (kW -> W) to the actuated
                              OtherEquipment 'Power Level'.
    - "supervisory_setpoint": write the command (degC) to the thermostat via the
                              'Zone Temperature Control' heating/cooling setpoint
                              actuators, with a deadband around the target.

    Warmup timesteps are skipped and control is applied only once E+ API data is
    ready. The timestamp handed to `step_fn` is the interval start (E+ reports the
    end-of-timestep clock; we subtract one control step).
    """

    def __init__(
        self,
        install_dir: str | Path,
        weather_path: str | Path,
        zone_name: str,
        dt_seconds: float,
        calendar_year: int,
        actuation_mode: str = "direct_power",
        equipment_name: str | None = None,
        setpoint_deadband_c: float = 0.5,
        console_output: bool = False,
    ) -> None:
        self.install_dir = Path(install_dir)
        self.weather_path = Path(weather_path)
        self.zone_name = zone_name
        self.dt = timedelta(seconds=float(dt_seconds))
        # E+ reports the weather file's calendar year (e.g. 2002 for a TMY EPW),
        # not the RunPeriod's Begin Year. The processed pipeline stamps this same
        # weather with a fixed calendar year, so we impose it here too, so the
        # reported timestamps align with the processed CSV and the config window.
        self.calendar_year = int(calendar_year)
        self.actuation_mode = actuation_mode
        self.equipment_name = equipment_name
        self.setpoint_deadband_c = float(setpoint_deadband_c)
        self.console_output = console_output

        if actuation_mode not in ("direct_power", "supervisory_setpoint"):
            raise ValueError(f"Unknown actuation_mode: {actuation_mode!r}.")
        if actuation_mode == "direct_power" and not equipment_name:
            raise ValueError("equipment_name is required for direct_power mode.")

        if not self.weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {self.weather_path}")

        self._api = None
        self._reset_handles()

    def _reset_handles(self) -> None:
        self._var_handle: int | None = None
        self._heat_rate_handle: int | None = None
        self._cool_rate_handle: int | None = None
        self._power_act_handle: int | None = None
        self._heat_sp_act_handle: int | None = None
        self._cool_sp_act_handle: int | None = None

    def run(
        self,
        run_idf_path: str | Path,
        output_dir: str | Path,
        step_fn: StepFn,
    ) -> None:
        run_idf_path = Path(run_idf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not run_idf_path.exists():
            raise FileNotFoundError(f"Run IDF not found: {run_idf_path}")

        api = self._load_api()
        state = api.state_manager.new_state()

        if not self.console_output:
            api.runtime.set_console_output_status(state, False)

        # Make the sensor variables available to the runtime API.
        api.exchange.request_variable(state, ZONE_AIR_TEMP_VARIABLE, self.zone_name)
        api.exchange.request_variable(state, ZONE_SENSIBLE_HEATING_RATE, self.zone_name)
        api.exchange.request_variable(state, ZONE_SENSIBLE_COOLING_RATE, self.zone_name)

        # Reset cached handles for this run (handles are valid within a run only).
        self._reset_handles()

        def _callback(s) -> None:
            self._on_zone_timestep(api, s, step_fn)

        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            state, _callback
        )

        exit_code = api.runtime.run_energyplus(
            state,
            [
                "-w",
                str(self.weather_path),
                "-d",
                str(output_dir),
                str(run_idf_path),
            ],
        )

        api.state_manager.delete_state(state)

        if exit_code != 0:
            raise RuntimeError(
                f"EnergyPlus run failed with exit code {exit_code}. "
                f"See {output_dir / 'eplusout.err'} for details."
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _load_api(self):
        if self._api is not None:
            return self._api

        install_str = str(self.install_dir)
        if install_str not in sys.path:
            sys.path.insert(0, install_str)

        try:
            from pyenergyplus.api import EnergyPlusAPI
        except Exception as exc:  # noqa: BLE001 - re-raise with guidance
            raise ImportError(
                "Could not import pyenergyplus from the EnergyPlus install at "
                f"{self.install_dir}. Confirm energyplus.install_dir points at a "
                "valid EnergyPlus installation (containing a 'pyenergyplus' "
                f"package) matching the running Python. Original error: {exc}"
            ) from exc

        self._api = EnergyPlusAPI()
        return self._api

    def _on_zone_timestep(self, api, state, step_fn: StepFn) -> None:
        if api.exchange.warmup_flag(state):
            return
        if not api.exchange.api_data_fully_ready(state):
            return

        self._resolve_sensor_handles(api, state)
        self._resolve_actuator_handles(api, state)

        t_in_c = api.exchange.get_variable_value(state, self._var_handle)
        heating_w = api.exchange.get_variable_value(state, self._heat_rate_handle)
        cooling_w = api.exchange.get_variable_value(state, self._cool_rate_handle)
        hvac_power_kw = (heating_w - cooling_w) / 1000.0

        obs = PlantObservation(t_in_c=t_in_c, hvac_power_kw=hvac_power_kw)
        interval_start = self._interval_start(api, state)

        command = step_fn(interval_start, obs)
        self._apply_command(api, state, command)

    def _resolve_sensor_handles(self, api, state) -> None:
        if self._var_handle is None:
            self._var_handle = self._require_variable(
                api, state, ZONE_AIR_TEMP_VARIABLE, self.zone_name
            )
        if self._heat_rate_handle is None:
            self._heat_rate_handle = self._require_variable(
                api, state, ZONE_SENSIBLE_HEATING_RATE, self.zone_name
            )
        if self._cool_rate_handle is None:
            self._cool_rate_handle = self._require_variable(
                api, state, ZONE_SENSIBLE_COOLING_RATE, self.zone_name
            )

    def _resolve_actuator_handles(self, api, state) -> None:
        if self.actuation_mode == "direct_power":
            if self._power_act_handle is None:
                self._power_act_handle = self._require_actuator(
                    api, state, "OtherEquipment", "Power Level", self.equipment_name
                )
        else:
            if self._heat_sp_act_handle is None:
                self._heat_sp_act_handle = self._require_actuator(
                    api, state, "Zone Temperature Control", "Heating Setpoint",
                    self.zone_name,
                )
            if self._cool_sp_act_handle is None:
                self._cool_sp_act_handle = self._require_actuator(
                    api, state, "Zone Temperature Control", "Cooling Setpoint",
                    self.zone_name,
                )

    def _apply_command(self, api, state, command: Optional[float]) -> None:
        if self.actuation_mode == "direct_power":
            # None (outside window) -> apply zero power.
            power_kw = 0.0 if command is None else float(command)
            api.exchange.set_actuator_value(
                state, self._power_act_handle, power_kw * 1000.0
            )
            return

        # supervisory_setpoint: None -> leave the native thermostat untouched.
        if command is None:
            return
        half = 0.5 * self.setpoint_deadband_c
        api.exchange.set_actuator_value(
            state, self._heat_sp_act_handle, float(command) - half
        )
        api.exchange.set_actuator_value(
            state, self._cool_sp_act_handle, float(command) + half
        )

    @staticmethod
    def _require_variable(api, state, name: str, key: str) -> int:
        handle = api.exchange.get_variable_handle(state, name, key)
        if handle == -1:
            raise RuntimeError(
                f"Could not resolve variable handle for {name!r} at {key!r}."
            )
        return handle

    @staticmethod
    def _require_actuator(api, state, comp_type: str, ctrl_type: str, key: str) -> int:
        handle = api.exchange.get_actuator_handle(state, comp_type, ctrl_type, key)
        if handle == -1:
            raise RuntimeError(
                f"Could not resolve actuator handle {comp_type!r}/{ctrl_type!r} "
                f"on {key!r}. Check the run IDF."
            )
        return handle

    def _interval_start(self, api, state) -> datetime:
        """
        Build the interval-start datetime from E+ time functions. E+ reports the
        clock at the END of the current timestep (minutes in 1..60, hour in
        0..23); timedelta normalizes any hour/day rollover. Subtracting one
        control step yields the start of the interval being controlled. The year
        is imposed from `calendar_year` (E+ reports the weather file's year).
        """
        month = api.exchange.month(state)
        day = api.exchange.day_of_month(state)
        hour = api.exchange.hour(state)
        minute = api.exchange.minutes(state)

        end_of_interval = datetime(self.calendar_year, month, day) + timedelta(
            hours=hour, minutes=minute
        )
        return end_of_interval - self.dt
