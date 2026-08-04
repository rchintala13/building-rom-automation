from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rom_automation.mpc.config_types import (
    ActuationConfig,
    ComfortConfig,
    ControlConfig,
    EnergyPlusConfig,
    ModelConfig,
    MpcConfig,
    ObjectiveConfig,
    PathsConfig,
    SelectionConfig,
    TouConfig,
    TouWindow,
    WindowConfig,
)

_VALID_ACTUATION_MODES = {"direct_power", "supervisory_setpoint"}


def load_mpc_config(config_path: str | Path) -> MpcConfig:
    """
    Load a run_mpc YAML config into a strongly-typed MpcConfig. Fails fast on
    missing sections / keys / invalid values.
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
            "energyplus",
            "selection",
            "model",
            "window",
            "history_hours",
            "control",
            "comfort",
            "tou",
            "objective",
            "actuation",
        ],
    )

    history_hours = float(raw["history_hours"])
    if history_hours <= 0:
        raise ValueError(f"history_hours must be positive. Got {history_hours}.")

    return MpcConfig(
        paths=_build_paths(raw["paths"]),
        energyplus=_build_energyplus(raw["energyplus"]),
        selection=_build_selection(raw["selection"]),
        model=_build_model(raw["model"]),
        window=_build_window(raw["window"]),
        history_hours=history_hours,
        control=_build_control(raw["control"]),
        comfort=_build_comfort(raw["comfort"]),
        tou=_build_tou(raw["tou"]),
        objective=_build_objective(raw["objective"]),
        actuation=_build_actuation(raw["actuation"]),
        disable_native_hvac=bool(raw.get("disable_native_hvac", True)),
    )


def _build_paths(raw_paths: dict[str, Any]) -> PathsConfig:
    _require_keys(
        raw_paths,
        [
            "raw_idf_root",
            "mpc_idf_root",
            "processed_root",
            "sysid_results_root",
            "weather",
            "output_root",
            "idf_edit_config",
        ],
        section_name="paths",
    )
    return PathsConfig(
        raw_idf_root=Path(raw_paths["raw_idf_root"]),
        mpc_idf_root=Path(raw_paths["mpc_idf_root"]),
        processed_root=Path(raw_paths["processed_root"]),
        sysid_results_root=Path(raw_paths["sysid_results_root"]),
        weather=Path(raw_paths["weather"]),
        output_root=Path(raw_paths["output_root"]),
        idf_edit_config=Path(raw_paths["idf_edit_config"]),
    )


def _build_energyplus(raw_ep: dict[str, Any]) -> EnergyPlusConfig:
    _require_keys(raw_ep, ["install_dir"], section_name="energyplus")
    return EnergyPlusConfig(install_dir=Path(raw_ep["install_dir"]))


def _build_selection(raw_selection: dict[str, Any]) -> SelectionConfig:
    _require_keys(raw_selection, ["city", "house_name"], section_name="selection")
    zone_name_raw = raw_selection.get("zone_name")
    zone_name = str(zone_name_raw) if zone_name_raw else None
    return SelectionConfig(
        city=str(raw_selection["city"]),
        house_name=str(raw_selection["house_name"]),
        zone_name=zone_name,
    )


def _build_model(raw_model: dict[str, Any]) -> ModelConfig:
    # `path` is optional (may be null => resolved from sysid_results_root).
    path_raw = raw_model.get("path") if isinstance(raw_model, dict) else None
    path = Path(path_raw) if path_raw else None
    return ModelConfig(path=path)


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


def _build_control(raw_control: dict[str, Any]) -> ControlConfig:
    _require_keys(
        raw_control,
        ["dt_minutes", "horizon_steps", "p_hvac_max_kw"],
        section_name="control",
    )

    dt_minutes = float(raw_control["dt_minutes"])
    if dt_minutes <= 0:
        raise ValueError(f"control.dt_minutes must be positive. Got {dt_minutes}.")

    horizon_steps = int(raw_control["horizon_steps"])
    if horizon_steps <= 0:
        raise ValueError(
            f"control.horizon_steps must be positive. Got {horizon_steps}."
        )

    p_hvac_max_kw = float(raw_control["p_hvac_max_kw"])
    if p_hvac_max_kw <= 0:
        raise ValueError(
            f"control.p_hvac_max_kw must be positive. Got {p_hvac_max_kw}."
        )

    control = ControlConfig(
        dt_minutes=dt_minutes,
        horizon_steps=horizon_steps,
        p_hvac_max_kw=p_hvac_max_kw,
        auto_capacity=bool(raw_control.get("auto_capacity", False)),
    )
    # Trigger the divisibility check early so config errors surface at load time.
    _ = control.timesteps_per_hour
    return control


def _build_comfort(raw_comfort: dict[str, Any]) -> ComfortConfig:
    _require_keys(raw_comfort, ["lower_c", "upper_c"], section_name="comfort")
    lower_c = float(raw_comfort["lower_c"])
    upper_c = float(raw_comfort["upper_c"])
    if lower_c >= upper_c:
        raise ValueError(
            f"comfort.lower_c ({lower_c}) must be strictly less than "
            f"comfort.upper_c ({upper_c})."
        )
    return ComfortConfig(lower_c=lower_c, upper_c=upper_c)


def _build_tou(raw_tou: dict[str, Any]) -> TouConfig:
    _require_keys(raw_tou, ["default_rate"], section_name="tou")

    default_rate = float(raw_tou["default_rate"])
    if default_rate < 0:
        raise ValueError(f"tou.default_rate must be non-negative. Got {default_rate}.")

    windows_raw = raw_tou.get("windows") or []
    windows: list[TouWindow] = []
    for i, w in enumerate(windows_raw):
        _require_keys(w, ["start_hour", "end_hour", "rate"], section_name=f"tou.windows[{i}]")
        start_hour = int(w["start_hour"])
        end_hour = int(w["end_hour"])
        rate = float(w["rate"])
        if not (0 <= start_hour < end_hour <= 24):
            raise ValueError(
                f"tou.windows[{i}] must satisfy 0 <= start_hour < end_hour <= 24. "
                f"Got start_hour={start_hour}, end_hour={end_hour}."
            )
        if rate < 0:
            raise ValueError(f"tou.windows[{i}].rate must be non-negative. Got {rate}.")
        windows.append(TouWindow(start_hour=start_hour, end_hour=end_hour, rate=rate))

    return TouConfig(default_rate=default_rate, windows=tuple(windows))


def _build_actuation(raw_actuation: dict[str, Any]) -> ActuationConfig:
    _require_keys(raw_actuation, ["mode"], section_name="actuation")

    mode = str(raw_actuation["mode"])
    if mode not in _VALID_ACTUATION_MODES:
        raise ValueError(
            f"actuation.mode must be one of {sorted(_VALID_ACTUATION_MODES)}. "
            f"Got {mode!r}."
        )

    deadband = float(raw_actuation.get("setpoint_deadband_c", 0.5))
    if deadband < 0:
        raise ValueError(
            f"actuation.setpoint_deadband_c must be non-negative. Got {deadband}."
        )

    num_power_zones = int(raw_actuation.get("num_power_zones", 4))
    if num_power_zones < 1:
        raise ValueError(
            f"actuation.num_power_zones must be >= 1. Got {num_power_zones}."
        )

    min_lookahead_steps = int(raw_actuation.get("min_lookahead_steps", 3))
    if min_lookahead_steps < 1:
        raise ValueError(
            f"actuation.min_lookahead_steps must be >= 1. Got {min_lookahead_steps}."
        )

    return ActuationConfig(  # type: ignore[arg-type]
        mode=mode,
        setpoint_deadband_c=deadband,
        num_power_zones=num_power_zones,
        min_lookahead_steps=min_lookahead_steps,
    )


def _build_objective(raw_objective: dict[str, Any]) -> ObjectiveConfig:
    _require_keys(raw_objective, ["w_energy", "w_comfort"], section_name="objective")
    w_energy = float(raw_objective["w_energy"])
    w_comfort = float(raw_objective["w_comfort"])
    if w_energy < 0 or w_comfort < 0:
        raise ValueError(
            f"objective weights must be non-negative. "
            f"Got w_energy={w_energy}, w_comfort={w_comfort}."
        )
    return ObjectiveConfig(w_energy=w_energy, w_comfort=w_comfort)


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
