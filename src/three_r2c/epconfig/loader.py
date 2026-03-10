from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from three_r2c.epconfig.types import (
    EditsConfig,
    IDFEditConfig,
    OutputConfig,
    PathsConfig,
    PerturbationConfig,
    RunPeriodConfig,
    ScheduleIntervalConfig,
    SelectionConfig,
    SetpointValueConfig,
    SetpointsConfig,
    SimulationControlConfig,
)

def load_idf_edit_config(config_path: str | Path) -> IDFEditConfig:
    """
    Load an IDF editing YAML config into an IDFEditConfig object.

    Parameters
    ----------
    config_path
        Path to the YAML configuration file.

    Returns
    -------
    IDFEditConfig
        Parsed and structured configuration object.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    _validate_top_level_sections(raw, config_path)

    paths_cfg = _build_paths_config(raw["paths"])
    selection_cfg = _build_selection_config(raw["selection"])
    output_cfg = _build_output_config(raw["output"])
    edits_cfg = _build_edits_config(raw["edits"])

    return IDFEditConfig(
        paths=paths_cfg,
        selection=selection_cfg,
        output=output_cfg,
        edits=edits_cfg,
    )


def _validate_top_level_sections(raw: dict[str, Any], config_path: Path) -> None:
    required_sections = ["paths", "selection", "output", "edits"]

    missing = [section for section in required_sections if section not in raw]
    if missing:
        raise KeyError(
            f"Missing required top-level section(s) in {config_path}: {missing}"
        )

def _build_paths_config(raw_paths: dict[str, Any]) -> PathsConfig:
    _require_keys(raw_paths, ["raw_idf_root", "output_root"], section_name="paths")

    return PathsConfig(
        raw_idf_root=Path(raw_paths["raw_idf_root"]),
        output_root=Path(raw_paths["output_root"]),
    )

def _build_selection_config(raw_selection: dict[str, Any]) -> SelectionConfig:
    _require_keys(
        raw_selection,
        ["mode", "single_file", "pattern", "recursive"],
        section_name="selection",
    )
    mode = raw_selection["mode"]
    if mode not in {"single", "batch"}:
        raise ValueError(
            f"selection.mode must be 'single' or 'batch'. Got: {mode!r}"
        )

    single_file = raw_selection["single_file"]
    if mode == "single" and not single_file:
        raise ValueError(
            "selection.single_file must be provided when selection.mode='single'"
        )

    return SelectionConfig(
        mode=mode,
        single_file=single_file,
        pattern=raw_selection["pattern"],
        recursive=bool(raw_selection["recursive"]),
    )

def _build_output_config(raw_output: dict[str, Any]) -> OutputConfig:
    _require_keys(raw_output, ["overwrite", "suffix"], section_name="output")

    return OutputConfig(
        overwrite=bool(raw_output["overwrite"]),
        suffix=str(raw_output["suffix"]),
    )

def _build_edits_config(raw_edits: dict[str, Any]) -> EditsConfig:
    _require_keys(
        raw_edits,
        ["simulation_control", "setpoints", "perturbation"],
        section_name="edits",
    )

    simulation_control_cfg = _build_simulation_control_config(
        raw_edits["simulation_control"]
    )
    setpoints_cfg = _build_setpoints_config(raw_edits["setpoints"])
    perturbation_cfg = _build_perturbation_config(raw_edits["perturbation"])

    return EditsConfig(
        simulation_control=simulation_control_cfg,
        setpoints=setpoints_cfg,
        perturbation=perturbation_cfg,
    )


def _build_simulation_control_config(
    raw_sim_control: dict[str, Any],
) -> SimulationControlConfig:
    _require_keys(
        raw_sim_control,
        ["run_period", "timestep_per_hour"],
        section_name="edits.simulation_control",
    )

    run_period_cfg = _build_run_period_config(raw_sim_control["run_period"])

    timestep_per_hour = int(raw_sim_control["timestep_per_hour"])
    if timestep_per_hour <= 0:
        raise ValueError("edits.simulation_control.timestep_per_hour must be > 0")

    return SimulationControlConfig(
        run_period=run_period_cfg,
        timestep_per_hour=timestep_per_hour,
    )


def _build_run_period_config(raw_run_period: dict[str, Any]) -> RunPeriodConfig:
    _require_keys(
        raw_run_period,
        [
            "begin_month",
            "begin_day_of_month",
            "end_month",
            "end_day_of_month",
        ],
        section_name="edits.simulation_control.run_period",
    )

    begin_month = int(raw_run_period["begin_month"])
    begin_day = int(raw_run_period["begin_day_of_month"])
    end_month = int(raw_run_period["end_month"])
    end_day = int(raw_run_period["end_day_of_month"])

    _validate_month_day(begin_month, begin_day, "run_period begin date")
    _validate_month_day(end_month, end_day, "run_period end date")

    return RunPeriodConfig(
        begin_month=begin_month,
        begin_day_of_month=begin_day,
        end_month=end_month,
        end_day_of_month=end_day,
    )


def _build_setpoints_config(raw_setpoints: dict[str, Any]) -> SetpointsConfig:
    _require_keys(
        raw_setpoints,
        ["heating", "cooling"],
        section_name="edits.setpoints",
    )

    heating_cfg = _build_setpoint_value_config(
        raw_setpoints["heating"],
        section_name="edits.setpoints.heating",
    )
    cooling_cfg = _build_setpoint_value_config(
        raw_setpoints["cooling"],
        section_name="edits.setpoints.cooling",
    )

    return SetpointsConfig(
        heating=heating_cfg,
        cooling=cooling_cfg,
    )


def _build_setpoint_value_config(
    raw_value: dict[str, Any],
    section_name: str,
) -> SetpointValueConfig:
    _require_keys(
        raw_value,
        ["enabled", "target_c", "schedule_name", "default_value_c", "intervals"],
        section_name=section_name)

    intervals = [
        ScheduleIntervalConfig(
            until=str(item["until"]),
            value_c=float(item["value_c"]),
        )
        for item in raw_value["intervals"]
    ]

    return SetpointValueConfig(
        enabled=bool(raw_value["enabled"]),
        schedule_type=str(raw_value["schedule_type"]),
        schedule_name=str(raw_value["schedule_name"]),
        default_value_c=float(raw_value["default_value_c"]),
        intervals=intervals,
    )


def _build_perturbation_config(
    raw_perturbation: dict[str, Any],
) -> PerturbationConfig:
    _require_keys(
        raw_perturbation,
        ["enabled", "cooling_offset_c", "heating_offset_c"],
        section_name="edits.perturbation",
    )

    return PerturbationConfig(
        enabled=bool(raw_perturbation["enabled"]),
        cooling_offset_c=float(raw_perturbation["cooling_offset_c"]),
        heating_offset_c=float(raw_perturbation["heating_offset_c"]),
    )

def _require_keys(
    raw_section: dict[str, Any],
    required_keys: list[str],
    section_name: str,
) -> None:
    missing = [key for key in required_keys if key not in raw_section]
    if missing:
        raise KeyError(f"Missing key(s) in section '{section_name}': {missing}")


def _validate_month_day(month: int, day: int, label: str) -> None:
    if not 1 <= month <= 12:
        raise ValueError(f"{label}: month must be between 1 and 12. Got {month}")

    if not 1 <= day <= 31:
        raise ValueError(f"{label}: day must be between 1 and 31. Got {day}")