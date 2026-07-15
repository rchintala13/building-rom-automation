from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rom_automation.rom_sim.config_types import (
    InputsConfig,
    ModeConfig,
    ModelConfig,
    PathsConfig,
    SelectionConfig,
    SimulationConfig,
    WallInitConfig,
    WindowConfig,
)


_VALID_MODE_KINDS = {"simulation", "one_step", "n_step"}
_VALID_INPUT_SOURCES = {"processed", "custom"}


def load_simulation_config(config_path: str | Path) -> SimulationConfig:
    """
    Load a run_4r2c_simulation YAML config into a strongly-typed
    SimulationConfig. Fails fast on missing sections / keys / invalid values.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    _validate_top_level_sections(
        raw,
        config_path,
        required_sections=[
            "paths",
            "selection",
            "model",
            "inputs",
            "window",
            "history_hours",
            "mode",
        ],
    )

    paths = _build_paths(raw["paths"])
    selection = _build_selection(raw["selection"])
    model = _build_model(raw["model"])
    inputs = _build_inputs(raw["inputs"])
    window = _build_window(raw["window"])
    wall_init = (
        _build_wall_init(raw["wall_init"]) if raw.get("wall_init") is not None else None
    )
    mode = _build_mode(raw["mode"])

    history_hours = float(raw["history_hours"])
    if history_hours <= 0:
        raise ValueError(f"history_hours must be positive. Got {history_hours}.")

    return SimulationConfig(
        paths=paths,
        selection=selection,
        model=model,
        inputs=inputs,
        window=window,
        history_hours=history_hours,
        wall_init=wall_init,
        mode=mode,
    )


def _build_paths(raw_paths: dict[str, Any]) -> PathsConfig:
    _require_keys(
        raw_paths,
        ["processed_root", "output_root"],
        section_name="paths",
    )
    return PathsConfig(
        processed_root=Path(raw_paths["processed_root"]),
        output_root=Path(raw_paths["output_root"]),
    )


def _build_selection(raw_selection: dict[str, Any]) -> SelectionConfig:
    _require_keys(raw_selection, ["city", "house_name"], section_name="selection")
    return SelectionConfig(
        city=str(raw_selection["city"]),
        house_name=str(raw_selection["house_name"]),
    )


def _build_model(raw_model: dict[str, Any]) -> ModelConfig:
    _require_keys(raw_model, ["path"], section_name="model")
    return ModelConfig(path=Path(raw_model["path"]))


def _build_inputs(raw_inputs: dict[str, Any]) -> InputsConfig:
    _require_keys(raw_inputs, ["source"], section_name="inputs")

    source = str(raw_inputs["source"])
    if source not in _VALID_INPUT_SOURCES:
        raise ValueError(
            f"inputs.source must be one of {sorted(_VALID_INPUT_SOURCES)}. "
            f"Got {source!r}."
        )

    custom_path_raw = raw_inputs.get("custom_path")
    custom_path = Path(custom_path_raw) if custom_path_raw else None

    if source == "custom" and custom_path is None:
        raise ValueError(
            "inputs.custom_path is required when inputs.source == 'custom'."
        )

    return InputsConfig(source=source, custom_path=custom_path)  # type: ignore[arg-type]


def _build_window(raw_window: dict[str, Any]) -> WindowConfig:
    _require_keys(raw_window, ["start", "duration_hours"], section_name="window")

    start_raw = raw_window["start"]
    if isinstance(start_raw, datetime):
        start = start_raw
    else:
        start = datetime.fromisoformat(str(start_raw))

    duration_hours = float(raw_window["duration_hours"])
    if duration_hours <= 0:
        raise ValueError(
            f"window.duration_hours must be positive. Got {duration_hours}."
        )

    return WindowConfig(start=start, duration_hours=duration_hours)


def _build_wall_init(raw_wall_init: dict[str, Any]) -> WallInitConfig:
    _require_keys(
        raw_wall_init, ["alpha_iw", "alpha_ow"], section_name="wall_init"
    )
    alpha_iw = float(raw_wall_init["alpha_iw"])
    alpha_ow = float(raw_wall_init["alpha_ow"])
    return WallInitConfig(alpha_iw=alpha_iw, alpha_ow=alpha_ow)


def _build_mode(raw_mode: dict[str, Any]) -> ModeConfig:
    _require_keys(raw_mode, ["kind"], section_name="mode")

    kind = str(raw_mode["kind"])
    if kind not in _VALID_MODE_KINDS:
        raise ValueError(
            f"mode.kind must be one of {sorted(_VALID_MODE_KINDS)}. Got {kind!r}."
        )

    n_steps_ahead_raw = raw_mode.get("n_steps_ahead")
    n_steps_ahead: int | None
    if kind == "n_step":
        if n_steps_ahead_raw is None:
            raise ValueError(
                "mode.n_steps_ahead is required when mode.kind == 'n_step'."
            )
        n_steps_ahead = int(n_steps_ahead_raw)
        if n_steps_ahead <= 0:
            raise ValueError(
                f"mode.n_steps_ahead must be positive. Got {n_steps_ahead}."
            )
    else:
        n_steps_ahead = None

    return ModeConfig(kind=kind, n_steps_ahead=n_steps_ahead)  # type: ignore[arg-type]


def _validate_top_level_sections(
    raw: dict[str, Any],
    config_path: Path,
    required_sections: list[str],
) -> None:
    missing = [section for section in required_sections if section not in raw]
    if missing:
        raise KeyError(
            f"Missing required top-level section(s) in {config_path}: {missing}"
        )


def _require_keys(
    raw_section: dict[str, Any],
    required_keys: list[str],
    section_name: str,
) -> None:
    missing = [key for key in required_keys if key not in raw_section]
    if missing:
        raise KeyError(f"Missing key(s) in section '{section_name}': {missing}")
