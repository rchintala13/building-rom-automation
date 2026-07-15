from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


ModeKind = Literal["simulation", "one_step", "n_step"]
InputSource = Literal["processed", "custom"]


@dataclass(frozen=True)
class PathsConfig:
    processed_root: Path
    output_root: Path


@dataclass(frozen=True)
class SelectionConfig:
    city: str
    house_name: str


@dataclass(frozen=True)
class ModelConfig:
    path: Path


@dataclass(frozen=True)
class InputsConfig:
    """
    Input source spec. If `source == 'processed'`, the workflow builds the
    CSV path from `paths.processed_root` + `selection` (matches sysid layout).
    If `source == 'custom'`, `custom_path` must point at a CSV with the same
    schema as processed_5min.csv.
    """

    source: InputSource
    custom_path: Path | None


@dataclass(frozen=True)
class WindowConfig:
    """
    Simulation window. `start` is an inclusive datetime; the workflow selects
    the segment `[start, start + duration_hours)` from the input CSV.
    """

    start: datetime
    duration_hours: float


@dataclass(frozen=True)
class WallInitConfig:
    """
    Wall temperature initialization from history means:
        T_iw(0) = alpha_iw * mean(T_oa_history) + (1 - alpha_iw) * mean(T_in_history)
        T_ow(0) = alpha_ow * mean(T_oa_history) + (1 - alpha_ow) * mean(T_in_history)
    """

    alpha_iw: float
    alpha_ow: float


@dataclass(frozen=True)
class ModeConfig:
    """
    Prediction mode:
      - "simulation": open-loop rollout from the initial state.
      - "one_step"  : reset T_in to measurement each step, propagate one step,
                      compare predicted T_in[k+1] to measured T_in[k+1].
      - "n_step"    : reset T_in to measurement each step, propagate n steps,
                      compare predicted T_in[k+n] to measured T_in[k+n].

    `n_steps_ahead` is required when `kind == "n_step"` and ignored otherwise.
    """

    kind: ModeKind
    n_steps_ahead: int | None


@dataclass(frozen=True)
class SimulationConfig:
    """
    Strongly-typed view of a run_4r2c_simulation YAML config.

    `wall_init` is optional: if omitted from the YAML, the workflow falls back
    to `init_alpha_iw` / `init_alpha_ow` stored in the model.json produced by
    the sysid workflow.
    """

    paths: PathsConfig
    selection: SelectionConfig
    model: ModelConfig
    inputs: InputsConfig
    window: WindowConfig
    history_hours: float
    wall_init: WallInitConfig | None
    mode: ModeConfig
