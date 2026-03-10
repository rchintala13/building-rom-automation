from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunPeriodConfig:
    begin_month: int
    begin_day_of_month: int
    end_month: int
    end_day_of_month: int


@dataclass
class SimulationControlConfig:
    run_period: RunPeriodConfig
    timestep_per_hour: int


@dataclass(frozen=True)
class ScheduleIntervalConfig:
    until: str
    value_c: float


@dataclass(frozen=True)
class SetpointValueConfig:
    enabled: bool
    schedule_type: str
    schedule_name: str
    default_value_c: float
    intervals: list[ScheduleIntervalConfig]


@dataclass
class SetpointsConfig:
    heating: SetpointValueConfig
    cooling: SetpointValueConfig


@dataclass
class PerturbationConfig:
    enabled: bool
    cooling_offset_c: float
    heating_offset_c: float


@dataclass
class PathsConfig:
    raw_idf_root: Path
    output_root: Path


@dataclass
class SelectionConfig:
    mode: str
    single_file: str | None
    pattern: str
    recursive: bool


@dataclass
class OutputConfig:
    overwrite: bool
    suffix: str


@dataclass
class EditsConfig:
    simulation_control: SimulationControlConfig
    setpoints: SetpointsConfig
    perturbation: PerturbationConfig


@dataclass
class IDFEditConfig:
    paths: PathsConfig
    selection: SelectionConfig
    output: OutputConfig
    edits: EditsConfig