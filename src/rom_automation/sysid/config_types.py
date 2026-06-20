from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rom_automation.sysid.noise_builders import P0Spec, ProcessNoiseSpec
from rom_automation.sysid.parameter_grid import (
    ParameterBoundFractions,
    ParameterCandidateGrid,
)


@dataclass(frozen=True)
class PathsConfig:
    raw_idf_root: Path
    processed_root: Path
    output_root: Path


@dataclass(frozen=True)
class SelectionConfig:
    city: str
    house_name: str


@dataclass(frozen=True)
class SplitsConfig:
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class DatasetConfig:
    history_hours: float
    splits: SplitsConfig


@dataclass(frozen=True)
class ObjectiveWeightsConfig:
    one_step: float
    n_step: float


@dataclass(frozen=True)
class EKFConfig:
    n_steps_ahead: int
    objective_weights: ObjectiveWeightsConfig
    r_value: float
    p0: P0Spec
    process_noise: ProcessNoiseSpec


@dataclass(frozen=True)
class SysIDConfig:
    """
    Strongly-typed view of a run_ekf_sysid YAML config.

    Built by `config_loader.load_sysid_config`; consumed by the EKF sysid
    workflow and CLI.
    """

    paths: PathsConfig
    selection: SelectionConfig
    dataset: DatasetConfig
    ekf: EKFConfig
    parameter_grid: ParameterCandidateGrid
    parameter_bounds: ParameterBoundFractions
