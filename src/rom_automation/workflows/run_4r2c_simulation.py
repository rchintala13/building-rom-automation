from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd

from rom_automation.logging_utils import get_logger
from rom_automation.models.types import FourR2CParameters
from rom_automation.rom_sim.config_types import SimulationConfig, WallInitConfig
from rom_automation.rom_sim.runner import FourR2CRunner
from rom_automation.sysid.dataset_adapter import (
    REQUIRED_COLUMNS,
    build_input_sequence,
)


_EDITED_SUFFIX = "__edited"
_PROCESSED_FILENAME = "processed_5min.csv"


def run_4r2c_simulation_workflow(
    cfg: SimulationConfig,
) -> None:
    """
    Run a 4R2C prediction workflow driven by a typed simulation config.

    Steps:
      1. Load identified model parameters from `cfg.model.path`.
      2. Load the input CSV (either from the processed data folder or a
         user-supplied custom CSV).
      3. Slice out the history window (n hours before start) and the
         simulation window (`cfg.window`).
      4. Initialize wall temperatures via history means + user alphas.
      5. Run the requested prediction mode.
      6. Write outputs (predictions, and measured values for one_step/n_step)
         to the output directory.
    """
    logger = get_logger(
        "rom_sim.run_4r2c_simulation",
        log_file=Path("logs") / "run_4r2c_simulation.log",
    )

    params = _load_model_parameters(cfg.model.path)
    logger.info("Loaded model parameters from %s", cfg.model.path)

    resolved_wall_init = _resolve_wall_init(
        yaml_wall_init=cfg.wall_init,
        model_path=cfg.model.path,
    )
    logger.info(
        "Wall init alphas: alpha_iw=%.4f, alpha_ow=%.4f (source: %s)",
        resolved_wall_init.alpha_iw,
        resolved_wall_init.alpha_ow,
        "yaml" if cfg.wall_init is not None else "model.json",
    )

    input_csv_path = _resolve_input_csv(cfg)
    logger.info("Reading input CSV: %s", input_csv_path)
    df = _read_input_csv(input_csv_path)

    dt_seconds = _detect_timestep_seconds(df.index)
    logger.info("Detected timestep: %s seconds", dt_seconds)

    history_df, segment_df = _slice_history_and_segment(
        df=df,
        start=cfg.window.start,
        duration_hours=cfg.window.duration_hours,
        history_hours=cfg.history_hours,
        dt_seconds=dt_seconds,
    )
    logger.info(
        "Sliced history=%d steps, segment=%d steps.",
        len(history_df),
        len(segment_df),
    )

    inputs = build_input_sequence(segment_df)
    measurements_t_in_c = segment_df["T_zone_C"].to_numpy(dtype=float)
    history_t_oa_c = history_df["T_oa_C"].to_numpy(dtype=float)
    history_t_in_c = history_df["T_zone_C"].to_numpy(dtype=float)

    runner = FourR2CRunner(params=params, dt_seconds=dt_seconds)

    initial_state = runner.initialize_state(
        t_in_0_c=float(measurements_t_in_c[0]),
        history_t_oa_c=history_t_oa_c,
        history_t_in_c=history_t_in_c,
        alpha_iw=resolved_wall_init.alpha_iw,
        alpha_ow=resolved_wall_init.alpha_ow,
    )
    logger.info(
        "Initial state: T_in=%.3f, T_iw=%.3f, T_ow=%.3f",
        initial_state.t_in_c,
        initial_state.t_iw_c,
        initial_state.t_ow_c,
    )

    output_dir = _build_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df = _run_and_build_output(
        cfg=cfg,
        runner=runner,
        initial_state=initial_state,
        inputs=inputs,
        measurements_t_in_c=measurements_t_in_c,
        timestamps=segment_df.index,
    )

    output_path = _build_output_path(cfg=cfg, output_dir=output_dir)
    logger.info("Writing %s output: %s", cfg.mode.kind, output_path)
    output_df.to_csv(output_path, index=False)

    summary_path = output_dir / "simulation_summary.json"
    logger.info("Writing simulation summary: %s", summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            _build_summary(
                cfg=cfg,
                params=params,
                resolved_wall_init=resolved_wall_init,
                wall_init_source="yaml" if cfg.wall_init is not None else "model.json",
                initial_state_dict={
                    "t_in_0_c": initial_state.t_in_c,
                    "t_iw_0_c": initial_state.t_iw_c,
                    "t_ow_0_c": initial_state.t_ow_c,
                },
                input_csv_path=input_csv_path,
                output_path=output_path,
                dt_seconds=dt_seconds,
                n_segment_steps=len(segment_df),
                n_history_steps=len(history_df),
            ),
            f,
            indent=2,
        )

    logger.info("4R2C simulation workflow complete.")


def _load_model_parameters(model_path: Path) -> FourR2CParameters:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with model_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    required = {
        "r_in_iw",
        "r_iw_ow",
        "r_ow_oa",
        "r_in_oa",
        "c_in",
        "c_w",
        "alpha_ghi_outer_wall",
        "alpha_ghi_inner_wall",
    }
    missing = required - set(raw.keys())
    if missing:
        raise KeyError(
            f"model file {model_path} is missing keys: {sorted(missing)}"
        )

    return FourR2CParameters(
        r_in_iw=float(raw["r_in_iw"]),
        r_iw_ow=float(raw["r_iw_ow"]),
        r_ow_oa=float(raw["r_ow_oa"]),
        r_in_oa=float(raw["r_in_oa"]),
        c_in=float(raw["c_in"]),
        c_w=float(raw["c_w"]),
        alpha_ghi_outer_wall=float(raw["alpha_ghi_outer_wall"]),
        alpha_ghi_inner_wall=float(raw["alpha_ghi_inner_wall"]),
    )


def _resolve_wall_init(
    yaml_wall_init: WallInitConfig | None,
    model_path: Path,
) -> WallInitConfig:
    """
    Resolve the wall-init alphas actually used to construct the initial state.

    YAML wins if provided; otherwise fall back to the `init_alpha_iw` /
    `init_alpha_ow` fields in `model.json` (written by the sysid workflow).
    Raises if neither source has them.
    """
    if yaml_wall_init is not None:
        return yaml_wall_init

    with model_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    alpha_iw = raw.get("init_alpha_iw")
    alpha_ow = raw.get("init_alpha_ow")

    if alpha_iw is None or alpha_ow is None:
        raise ValueError(
            "wall_init alphas not provided in the YAML config, and model.json "
            f"at {model_path} does not include `init_alpha_iw`/`init_alpha_ow`. "
            "Either add a wall_init section to the YAML or re-run the sysid "
            "workflow so it writes them to model.json."
        )

    return WallInitConfig(
        alpha_iw=float(alpha_iw),
        alpha_ow=float(alpha_ow),
    )


def _resolve_input_csv(cfg: SimulationConfig) -> Path:
    if cfg.inputs.source == "processed":
        csv_path = (
            cfg.paths.processed_root
            / cfg.selection.city
            / f"{cfg.selection.house_name}{_EDITED_SUFFIX}"
            / _PROCESSED_FILENAME
        )
    else:
        assert cfg.inputs.custom_path is not None  # loader enforces
        csv_path = cfg.inputs.custom_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    return csv_path


def _read_input_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV {csv_path} is missing required columns: {missing}"
        )
    return df


def _detect_timestep_seconds(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Input CSV must have a DatetimeIndex column 'timestamp'.")

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError("Cannot detect timestep from fewer than 2 timestamps.")

    return float(deltas.mode().iloc[0].total_seconds())


def _slice_history_and_segment(
    df: pd.DataFrame,
    start,
    duration_hours: float,
    history_hours: float,
    dt_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice the dataframe into (history_df, segment_df).

    Both slices are contiguous windows that must lie inside the dataframe's
    timestamp coverage; otherwise the workflow raises.
    """
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(hours=float(duration_hours))
    history_start_ts = start_ts - pd.Timedelta(hours=float(history_hours))

    if history_start_ts < df.index.min():
        raise ValueError(
            f"Requested history window starts at {history_start_ts}, which is "
            f"before the input CSV's first timestamp {df.index.min()}. "
            "Increase input coverage, move the start forward, or reduce history_hours."
        )
    if end_ts > df.index.max() + pd.Timedelta(seconds=dt_seconds):
        raise ValueError(
            f"Requested segment ends at {end_ts}, which is after the input "
            f"CSV's last timestamp {df.index.max()}. Reduce duration_hours or "
            "extend input coverage."
        )

    history_df = df.loc[(df.index >= history_start_ts) & (df.index < start_ts)]
    segment_df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]

    if len(history_df) == 0:
        raise ValueError(
            f"History window is empty for start={start_ts}, "
            f"history_hours={history_hours}."
        )
    if len(segment_df) == 0:
        raise ValueError(
            f"Segment window is empty for start={start_ts}, "
            f"duration_hours={duration_hours}."
        )

    return history_df, segment_df


def _build_output_dir(cfg: SimulationConfig) -> Path:
    return cfg.paths.output_root / cfg.selection.city / cfg.selection.house_name


def _build_output_path(cfg: SimulationConfig, output_dir: Path) -> Path:
    """
    Output filenames encode mode + start timestamp so multiple runs against
    the same home don't clobber each other.
    """
    stamp = cfg.window.start.strftime("%Y%m%dT%H%M")
    if cfg.mode.kind == "n_step":
        assert cfg.mode.n_steps_ahead is not None
        return output_dir / f"n_step_{cfg.mode.n_steps_ahead}_{stamp}.csv"
    return output_dir / f"{cfg.mode.kind}_{stamp}.csv"


def _run_and_build_output(
    cfg: SimulationConfig,
    runner: FourR2CRunner,
    initial_state,
    inputs,
    measurements_t_in_c: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    if cfg.mode.kind == "simulation":
        sim = runner.simulate(initial_state=initial_state, inputs=inputs)
        return pd.DataFrame(
            {
                "timestamp": list(timestamps),
                "t_in_pred_c": sim.t_in_pred_c,
                "t_iw_pred_c": sim.t_iw_pred_c,
                "t_ow_pred_c": sim.t_ow_pred_c,
            }
        )

    if cfg.mode.kind == "one_step":
        pred = runner.one_step_predict(
            initial_state=initial_state,
            inputs=inputs,
            measurements_t_in_c=measurements_t_in_c,
        )
        return pd.DataFrame(
            {
                "timestamp": list(timestamps),
                "t_in_measured_c": measurements_t_in_c,
                "t_in_pred_one_step_c": pred.t_in_pred_c,
            }
        )

    if cfg.mode.kind == "n_step":
        assert cfg.mode.n_steps_ahead is not None
        pred = runner.n_step_predict(
            initial_state=initial_state,
            inputs=inputs,
            measurements_t_in_c=measurements_t_in_c,
            n_steps_ahead=cfg.mode.n_steps_ahead,
        )
        col = f"t_in_pred_n_step_{cfg.mode.n_steps_ahead}_c"
        return pd.DataFrame(
            {
                "timestamp": list(timestamps),
                "t_in_measured_c": measurements_t_in_c,
                col: pred.t_in_pred_c,
            }
        )

    raise ValueError(f"Unknown mode kind: {cfg.mode.kind!r}")


def _build_summary(
    cfg: SimulationConfig,
    params: FourR2CParameters,
    resolved_wall_init: WallInitConfig,
    wall_init_source: str,
    initial_state_dict: dict,
    input_csv_path: Path,
    output_path: Path,
    dt_seconds: float,
    n_segment_steps: int,
    n_history_steps: int,
) -> dict:
    return {
        "mode": {
            "kind": cfg.mode.kind,
            "n_steps_ahead": cfg.mode.n_steps_ahead,
        },
        "selection": {
            "city": cfg.selection.city,
            "house_name": cfg.selection.house_name,
        },
        "window": {
            "start": cfg.window.start.isoformat(),
            "duration_hours": cfg.window.duration_hours,
            "end": (
                cfg.window.start
                + timedelta(hours=cfg.window.duration_hours)
            ).isoformat(),
        },
        "history_hours": cfg.history_hours,
        "wall_init": {
            **asdict(resolved_wall_init),
            "source": wall_init_source,
        },
        "inputs": {
            "source": cfg.inputs.source,
            "csv_path": str(input_csv_path),
        },
        "model_path": str(cfg.model.path),
        "model_parameters": asdict(params),
        "initial_state": initial_state_dict,
        "timestep_seconds": dt_seconds,
        "n_segment_steps": n_segment_steps,
        "n_history_steps": n_history_steps,
        "output_csv_path": str(output_path),
    }
