from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

ActuationMode = Literal["direct_power", "supervisory_setpoint"]
ObserverMode = Literal["innovation_consistency", "fixed"]


@dataclass(frozen=True)
class PathsConfig:
    """
    Filesystem roots for the MPC workflow.

    - `raw_idf_root`      : source IDFs, used only for the copy-if-missing step.
    - `mpc_idf_root`      : the MPC IDF folder the config selects from. If the
                            selected IDF does not exist here, it is copied from
                            `raw_idf_root`.
    - `processed_root`    : processed_5min.csv location (perfect-foresight
                            disturbances + wall-init history).
    - `sysid_results_root`: where the identified `model.json` lives.
    - `weather`           : EPW weather file for the E+ run.
    - `output_root`       : MPC results output root.
    - `idf_edit_config`   : the standard idf_edit YAML, reused to bring the MPC
                            IDF to the same state as the processed IDF (schedule
                            file paths, setpoints, timestep, outputs) before the
                            MPC-specific injection object is overlaid.
    """

    raw_idf_root: Path
    mpc_idf_root: Path
    processed_root: Path
    sysid_results_root: Path
    weather: Path
    output_root: Path
    idf_edit_config: Path


@dataclass(frozen=True)
class EnergyPlusConfig:
    """
    EnergyPlus install directory. Used to locate the IDD (for eppy) and to make
    `pyenergyplus` importable (the package ships inside the E+ install).
    """

    install_dir: Path

    @property
    def idd_path(self) -> Path:
        return self.install_dir / "Energy+.idd"


@dataclass(frozen=True)
class SelectionConfig:
    """
    Building selection. `zone_name` may be None, in which case the conditioned
    zone is auto-detected from the IDF's ZoneControl:Thermostat.
    """

    city: str
    house_name: str
    zone_name: str | None


@dataclass(frozen=True)
class ModelConfig:
    """
    Identified 4R2C model location. If `path` is None, the workflow resolves it
    to `sysid_results_root/<city>/<house_name>/model.json`.
    """

    path: Path | None


@dataclass(frozen=True)
class WindowConfig:
    """
    Closed-loop simulation window: `[start, start + duration_hours)`.
    """

    start: datetime
    duration_hours: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=self.duration_hours)


@dataclass(frozen=True)
class ControlConfig:
    """
    Control / MPC timing and actuation limits.

    - `dt_minutes`    : control step, which also sets the E+ zone timestep
                        (Number of Timesteps per Hour = 60 / dt_minutes).
    - `horizon_steps` : MPC prediction horizon length in control steps.
    - `p_hvac_max_kw` : symmetric heating/cooling thermal capacity [kW].
    """

    dt_minutes: float
    horizon_steps: int
    p_hvac_max_kw: float
    # When True, `p_hvac_max_kw` is overridden at runtime by a calibration pass
    # that measures the actual HVAC max sensible cooling capacity (a full-tilt
    # EnergyPlus run). The configured value is used as a fallback.
    auto_capacity: bool = False

    @property
    def dt_seconds(self) -> float:
        return self.dt_minutes * 60.0

    @property
    def timesteps_per_hour(self) -> int:
        per_hour = 60.0 / self.dt_minutes
        rounded = int(round(per_hour))
        if abs(per_hour - rounded) > 1e-9:
            raise ValueError(
                f"control.dt_minutes={self.dt_minutes} does not divide 60 evenly; "
                f"cannot map to an integer E+ Timestep (got {per_hour})."
            )
        return rounded


@dataclass(frozen=True)
class ComfortConfig:
    """
    Indoor-temperature comfort band [degC]. Enforced as a soft constraint via
    slack in the MPC objective.
    """

    lower_c: float
    upper_c: float


@dataclass(frozen=True)
class TouWindow:
    """
    A time-of-use rate window. `[start_hour, end_hour)` on a 24h clock (end_hour
    exclusive). Wrapping past midnight is not supported here — express it as two
    windows. `rate` is in currency per kWh.
    """

    start_hour: int
    end_hour: int
    rate: float


@dataclass(frozen=True)
class TouConfig:
    """
    Time-of-use rate schedule. `default_rate` applies to any hour not covered by
    a window.
    """

    default_rate: float
    windows: tuple[TouWindow, ...]

    def rate_at_hour(self, hour: int) -> float:
        for w in self.windows:
            if w.start_hour <= hour < w.end_hour:
                return w.rate
        return self.default_rate


@dataclass(frozen=True)
class ObjectiveConfig:
    """
    Weights on the (linear) MPC objective: energy cost vs. comfort-slack penalty.
    """

    w_energy: float
    w_comfort: float


@dataclass(frozen=True)
class ActuationConfig:
    """
    How the MPC acts on EnergyPlus.

    - "direct_power": the MPC injects its optimal HVAC thermal power directly into
      the zone (via an actuated OtherEquipment object). The principal HVAC is
      typically disabled (see `disable_native_hvac`) so the injection is the sole
      conditioner. Purest test of the 4R2C.

    - "supervisory_setpoint": the MPC commands the zone-temperature setpoint (its
      planned next-step temperature) and the building's principal HVAC tracks it.
      Realistic supervisory control — the MPC never directly moves heat; the real
      heat pump delivers the power. `setpoint_deadband_c` is the total thermostat
      deadband placed around the commanded target.
    """

    mode: ActuationMode
    setpoint_deadband_c: float
    # supervisory_setpoint only: the planned power trajectory is binned into
    # `num_power_zones` equal fractions of p_hvac_max; the commanded setpoint is
    # the planned temperature at the end of the current power-zone run (with a
    # `min_lookahead_steps` floor so a brief regime still clears the deadband).
    num_power_zones: int = 4
    min_lookahead_steps: int = 3


@dataclass(frozen=True)
class ObserverConfig:
    """
    MPC state-observer tuning. The observer gain is built from the sysid-stored
    Q/R ingredients via one of:

    - "innovation_consistency": scale Q so the filter's assumed innovation
      variance matches the observed one-step innovation (from sysid). The gain
      auto-relaxes as the identified model improves. This is the default.
    - "fixed": scale Q by a constant `inflation_factor` (1.0 = raw sysid Q).

    Larger effective process noise => the filter trusts the T_in measurement more
    (K_in -> 1) and corrects the walls from the innovation.
    """

    mode: ObserverMode
    inflation_factor: float


@dataclass(frozen=True)
class MpcConfig:
    """
    Strongly-typed view of a run_mpc YAML config.
    """

    paths: PathsConfig
    energyplus: EnergyPlusConfig
    selection: SelectionConfig
    model: ModelConfig
    window: WindowConfig
    history_hours: float
    control: ControlConfig
    comfort: ComfortConfig
    tou: TouConfig
    objective: ObjectiveConfig
    actuation: ActuationConfig
    observer: ObserverConfig
    # Only meaningful in "direct_power" mode: when True, the run IDF's thermostat
    # deadband is widened so the principal HVAC never activates, leaving the MPC's
    # injected power as the sole conditioner. Ignored in "supervisory_setpoint"
    # mode, where the principal HVAC is the actuator and must stay enabled.
    disable_native_hvac: bool = True

    @property
    def is_supervisory(self) -> bool:
        return self.actuation.mode == "supervisory_setpoint"

    @property
    def is_direct_power(self) -> bool:
        return self.actuation.mode == "direct_power"

    def resolved_model_path(self) -> Path:
        if self.model.path is not None:
            return self.model.path
        return (
            self.paths.sysid_results_root
            / self.selection.city
            / self.selection.house_name
            / "model.json"
        )

    def base_mpc_idf_path(self) -> Path:
        return (
            self.paths.mpc_idf_root
            / self.selection.city
            / f"{self.selection.house_name}.idf"
        )

    def raw_idf_path(self) -> Path:
        return (
            self.paths.raw_idf_root
            / self.selection.city
            / f"{self.selection.house_name}.idf"
        )

    def output_dir(self) -> Path:
        return self.paths.output_root / self.selection.city / self.selection.house_name
