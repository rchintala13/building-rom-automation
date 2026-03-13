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
class ScheduleFileConfig:
    root_dir: Path
    rewrite_paths: bool

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

@dataclass(frozen=True)
class OutputVariablesConfig:
    site: list[str]
    zone: list[str]

@dataclass(frozen=True)
class SimOutputsConfig:
    zone_name: str
    reporting_frequency: str
    variables: OutputVariablesConfig

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
    simoutputs: SimOutputsConfig
    edits: EditsConfig
    schedule_files: ScheduleFileConfig | None = None


# -----------------------------
# New simulation config types
# -----------------------------

@dataclass(frozen=True)
class SimulationPathsConfig:
    edited_idf_root: Path
    output_root: Path


@dataclass(frozen=True)
class EnergyPlusConfig:
    eplus_exe: Path
    weather_file: Path


@dataclass(frozen=True)
class RunOptionsConfig:
    overwrite: bool
    preserve_relative_structure: bool


@dataclass(frozen=True)
class SimulationConfig:
    simulation: EnergyPlusConfig
    paths: SimulationPathsConfig
    selection: SelectionConfig
    run_options: RunOptionsConfig