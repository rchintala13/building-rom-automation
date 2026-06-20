from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rom_automation.sysid.config_types import (
    DatasetConfig,
    EKFConfig,
    ObjectiveWeightsConfig,
    PathsConfig,
    SelectionConfig,
    SplitsConfig,
    SysIDConfig,
)
from rom_automation.sysid.noise_builders import P0Spec, ProcessNoiseSpec
from rom_automation.sysid.parameter_grid import (
    ParameterBoundFractions,
    ParameterCandidateGrid,
    parameter_bound_fractions_from_dict,
    parameter_names,
)


def load_sysid_config(config_path: str | Path) -> SysIDConfig:
    """
    Load an EKF sysid YAML config into a strongly-typed SysIDConfig.

    Raises on missing sections / keys, invalid types, or invalid value
    ranges so failures surface at CLI startup instead of mid-workflow.
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
            "dataset",
            "ekf",
            "parameter_grid",
            "parameter_bounds",
        ],
    )

    paths_cfg = _build_paths_config(raw["paths"])
    selection_cfg = _build_selection_config(raw["selection"])
    dataset_cfg = _build_dataset_config(raw["dataset"])
    ekf_cfg = _build_ekf_config(raw["ekf"])
    parameter_grid_cfg = _build_parameter_grid_config(raw["parameter_grid"])
    parameter_bounds_cfg = _build_parameter_bounds_config(raw["parameter_bounds"])

    return SysIDConfig(
        paths=paths_cfg,
        selection=selection_cfg,
        dataset=dataset_cfg,
        ekf=ekf_cfg,
        parameter_grid=parameter_grid_cfg,
        parameter_bounds=parameter_bounds_cfg,
    )


def _build_paths_config(raw_paths: dict[str, Any]) -> PathsConfig:
    _require_keys(
        raw_paths,
        ["raw_idf_root", "processed_root", "output_root"],
        section_name="paths",
    )

    return PathsConfig(
        raw_idf_root=Path(raw_paths["raw_idf_root"]),
        processed_root=Path(raw_paths["processed_root"]),
        output_root=Path(raw_paths["output_root"]),
    )


def _build_selection_config(raw_selection: dict[str, Any]) -> SelectionConfig:
    _require_keys(
        raw_selection,
        ["city", "house_name"],
        section_name="selection",
    )

    return SelectionConfig(
        city=str(raw_selection["city"]),
        house_name=str(raw_selection["house_name"]),
    )


def _build_dataset_config(raw_dataset: dict[str, Any]) -> DatasetConfig:
    _require_keys(
        raw_dataset,
        ["history_hours", "splits"],
        section_name="dataset",
    )

    history_hours = float(raw_dataset["history_hours"])
    if history_hours <= 0:
        raise ValueError(
            f"dataset.history_hours must be positive. Got {history_hours}."
        )

    splits_cfg = _build_splits_config(raw_dataset["splits"])

    return DatasetConfig(
        history_hours=history_hours,
        splits=splits_cfg,
    )


def _build_splits_config(raw_splits: dict[str, Any]) -> SplitsConfig:
    _require_keys(raw_splits, ["train", "test"], section_name="dataset.splits")

    train = float(raw_splits["train"])
    test = float(raw_splits["test"])
    val = float(raw_splits.get("val", 0.0))

    for name, value in (("train", train), ("val", val), ("test", test)):
        if value < 0 or value > 1:
            raise ValueError(
                f"dataset.splits.{name} must lie in [0, 1]. Got {value}."
            )

    total = train + val + test
    if total > 1.0 + 1e-9:
        raise ValueError(
            f"dataset.splits must sum to <= 1.0. Got train+val+test = {total}."
        )

    if train <= 0:
        raise ValueError(f"dataset.splits.train must be positive. Got {train}.")

    if test <= 0:
        raise ValueError(f"dataset.splits.test must be positive. Got {test}.")

    return SplitsConfig(train=train, val=val, test=test)


def _build_ekf_config(raw_ekf: dict[str, Any]) -> EKFConfig:
    _require_keys(
        raw_ekf,
        [
            "n_steps_ahead",
            "objective_weights",
            "r_value",
            "p0",
            "process_noise",
        ],
        section_name="ekf",
    )

    n_steps_ahead = int(raw_ekf["n_steps_ahead"])
    if n_steps_ahead <= 0:
        raise ValueError(
            f"ekf.n_steps_ahead must be positive. Got {n_steps_ahead}."
        )

    r_value = float(raw_ekf["r_value"])
    if r_value <= 0:
        raise ValueError(f"ekf.r_value must be positive. Got {r_value}.")

    objective_weights_cfg = _build_objective_weights_config(
        raw_ekf["objective_weights"]
    )
    p0_spec = _build_p0_spec(raw_ekf["p0"])
    process_noise_spec = _build_process_noise_spec(raw_ekf["process_noise"])

    return EKFConfig(
        n_steps_ahead=n_steps_ahead,
        objective_weights=objective_weights_cfg,
        r_value=r_value,
        p0=p0_spec,
        process_noise=process_noise_spec,
    )


def _build_objective_weights_config(
    raw_weights: dict[str, Any],
) -> ObjectiveWeightsConfig:
    _require_keys(
        raw_weights,
        ["one_step", "n_step"],
        section_name="ekf.objective_weights",
    )

    one_step = float(raw_weights["one_step"])
    n_step = float(raw_weights["n_step"])

    if one_step < 0 or n_step < 0:
        raise ValueError(
            f"ekf.objective_weights must be nonnegative. Got {raw_weights}."
        )

    if one_step == 0 and n_step == 0:
        raise ValueError(
            "At least one of ekf.objective_weights.one_step / .n_step must be positive."
        )

    return ObjectiveWeightsConfig(one_step=one_step, n_step=n_step)


def _build_p0_spec(raw_p0: dict[str, Any]) -> P0Spec:
    _require_keys(
        raw_p0,
        [
            "temperature_std_c",
            "resistance_std_ratio",
            "capacitance_std_ratio",
            "alpha_std_ratio",
            "alpha_std_floor",
        ],
        section_name="ekf.p0",
    )

    raw_temp = raw_p0["temperature_std_c"]
    _require_keys(
        raw_temp,
        ["t_in_c", "t_iw_c", "t_ow_c"],
        section_name="ekf.p0.temperature_std_c",
    )

    temperature_std_c = {
        "t_in_c": float(raw_temp["t_in_c"]),
        "t_iw_c": float(raw_temp["t_iw_c"]),
        "t_ow_c": float(raw_temp["t_ow_c"]),
    }
    for name, value in temperature_std_c.items():
        if value <= 0:
            raise ValueError(
                f"ekf.p0.temperature_std_c.{name} must be positive. Got {value}."
            )

    resistance_std_ratio = float(raw_p0["resistance_std_ratio"])
    capacitance_std_ratio = float(raw_p0["capacitance_std_ratio"])
    alpha_std_ratio = float(raw_p0["alpha_std_ratio"])
    alpha_std_floor = float(raw_p0["alpha_std_floor"])

    for name, value in (
        ("resistance_std_ratio", resistance_std_ratio),
        ("capacitance_std_ratio", capacitance_std_ratio),
        ("alpha_std_ratio", alpha_std_ratio),
        ("alpha_std_floor", alpha_std_floor),
    ):
        if value < 0:
            raise ValueError(f"ekf.p0.{name} must be nonnegative. Got {value}.")

    return P0Spec(
        temperature_std_c=temperature_std_c,
        resistance_std_ratio=resistance_std_ratio,
        capacitance_std_ratio=capacitance_std_ratio,
        alpha_std_ratio=alpha_std_ratio,
        alpha_std_floor=alpha_std_floor,
    )


def _build_process_noise_spec(raw_pn: dict[str, Any]) -> ProcessNoiseSpec:
    _require_keys(
        raw_pn,
        ["q_in_std_kw", "q_iw_std_kw", "q_ow_std_kw"],
        section_name="ekf.process_noise",
    )

    q_in = float(raw_pn["q_in_std_kw"])
    q_iw = float(raw_pn["q_iw_std_kw"])
    q_ow = float(raw_pn["q_ow_std_kw"])

    for name, value in (
        ("q_in_std_kw", q_in),
        ("q_iw_std_kw", q_iw),
        ("q_ow_std_kw", q_ow),
    ):
        if value < 0:
            raise ValueError(
                f"ekf.process_noise.{name} must be nonnegative. Got {value}."
            )

    return ProcessNoiseSpec(
        q_in_std_kw=q_in,
        q_iw_std_kw=q_iw,
        q_ow_std_kw=q_ow,
    )


def _build_parameter_grid_config(
    raw_grid: dict[str, Any],
) -> ParameterCandidateGrid:
    _require_keys(
        raw_grid,
        list(parameter_names()),
        section_name="parameter_grid",
    )

    kwargs: dict[str, list[float]] = {}
    for name in parameter_names():
        values = raw_grid[name]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"parameter_grid.{name} must be a non-empty list. Got {values!r}."
            )
        kwargs[name] = [float(v) for v in values]

    return ParameterCandidateGrid(**kwargs)


def _build_parameter_bounds_config(
    raw_bounds: dict[str, Any],
) -> ParameterBoundFractions:
    return parameter_bound_fractions_from_dict(raw_bounds)


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
