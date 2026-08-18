from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd

from rom_automation.logging_utils import get_logger
from rom_automation.models.types import FourR2CParameters
from rom_automation.models.warm_start import warm_start_walls
from rom_automation.rom_sim.config_types import SimulationConfig
from rom_automation.rom_sim.runner import FourR2CRunner
from rom_automation.sysid.dataset_adapter import (
    REQUIRED_COLUMNS,
    build_input_sequence,
)


_EDITED_SUFFIX = "__edited"
_PROCESSED_FILENAME = "processed_5min.csv"


def run_4r2c_simulation_workflow(
    config: SimulationConfig,
) -> None:
    """
    Run a 4R2C prediction workflow driven by a typed simulation config.

    Steps:
      1. Load identified model parameters from `config.model.path`.
      2. Load the input CSV (either from the processed data folder or a
         user-supplied custom CSV).
      3. Slice out the history window (n hours before start) and the
         simulation window (`config.window`).
      4. Initialize wall temperatures via history means + user alphas.
      5. Run the requested prediction mode.
      6. Write outputs (predictions, and measured values for one_step/n_step)
         to the output directory.
    """
    logger = get_logger(
        "rom_sim.run_4r2c_simulation",
        log_file=Path("logs") / "run_4r2c_simulation.log",
    )

    params = _load_model_parameters(config.model.path)
    logger.info("Loaded model parameters from %s", config.model.path)

    input_csv_path = _resolve_input_csv(config)
    logger.info("Reading input CSV: %s", input_csv_path)
    df = _read_input_csv(input_csv_path)

    dt_seconds = _detect_timestep_seconds(df.index)
    logger.info("Detected timestep: %s seconds", dt_seconds)

    history_df, segment_df = _slice_history_and_segment(
        df=df,
        start=config.window.start,
        duration_hours=config.window.duration_hours,
        history_hours=config.history_hours,
        dt_seconds=dt_seconds,
    )
    logger.info(
        "Sliced history=%d steps, segment=%d steps.",
        len(history_df),
        len(segment_df),
    )

    inputs = build_input_sequence(segment_df)
    measurements_t_in_c = segment_df["T_zone_C"].to_numpy(dtype=float)
    history_t_in_c = history_df["T_zone_C"].to_numpy(dtype=float)

    runner = FourR2CRunner(params=params, dt_seconds=dt_seconds)

    warm_start = warm_start_walls(
        params=params,
        history_inputs=build_input_sequence(history_df),
        history_t_in_c=history_t_in_c,
        t_in_0_c=float(measurements_t_in_c[0]),
        dt_seconds=dt_seconds,
    )
    initial_state = warm_start.initial_condition.as_state()
    logger.info(
        "Warm-started initial state: T_in=%.3f, T_iw=%.3f, T_ow=%.3f "
        "(steady-state seed T_iw=%.3f, T_ow=%.3f)",
        initial_state.t_in_c,
        initial_state.t_iw_c,
        initial_state.t_ow_c,
        warm_start.seed_t_iw_c,
        warm_start.seed_t_ow_c,
    )

    output_dir = _build_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df = _run_and_build_output(
        config=config,
        runner=runner,
        initial_state=initial_state,
        inputs=inputs,
        measurements_t_in_c=measurements_t_in_c,
        timestamps=segment_df.index,
    )

    output_path = _build_output_path(config=config, output_dir=output_dir)
    logger.info("Writing %s output: %s", config.mode.kind, output_path)
    output_df.to_csv(output_path, index=False)

    summary_path = output_dir / "simulation_summary.json"
    logger.info("Writing simulation summary: %s", summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            _build_summary(
                config=config,
                params=params,
                wall_init_seed={
                    "t_iw_c": warm_start.seed_t_iw_c,
                    "t_ow_c": warm_start.seed_t_ow_c,
                },
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




def _resolve_input_csv(config: SimulationConfig) -> Path:
    if config.inputs.source == "processed":
        csv_path = (
            config.paths.processed_root
            / config.selection.city
            / f"{config.selection.house_name}{_EDITED_SUFFIX}"
            / _PROCESSED_FILENAME
        )
    else:
        assert config.inputs.custom_path is not None  # loader enforces
        csv_path = config.inputs.custom_path

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


def _build_output_dir(config: SimulationConfig) -> Path:
    return config.paths.output_root / config.selection.city / config.selection.house_name


def _build_output_path(config: SimulationConfig, output_dir: Path) -> Path:
    """
    Output filenames encode mode + start timestamp so multiple runs against
    the same home don't clobber each other.
    """
    stamp = config.window.start.strftime("%Y%m%dT%H%M")
    if config.mode.kind == "n_step":
        assert config.mode.n_steps_ahead is not None
        return output_dir / f"n_step_{config.mode.n_steps_ahead}_{stamp}.csv"
    return output_dir / f"{config.mode.kind}_{stamp}.csv"


def _run_and_build_output(
    config: SimulationConfig,
    runner: FourR2CRunner,
    initial_state,
    inputs,
    measurements_t_in_c: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    if config.mode.kind == "simulation":
        sim = runner.simulate(initial_state=initial_state, inputs=inputs)
        return pd.DataFrame(
            {
                "timestamp": list(timestamps),
                "t_in_pred_c": sim.t_in_pred_c,
                "t_iw_pred_c": sim.t_iw_pred_c,
                "t_ow_pred_c": sim.t_ow_pred_c,
            }
        )

    if config.mode.kind == "one_step":
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

    if config.mode.kind == "n_step":
        assert config.mode.n_steps_ahead is not None
        pred = runner.n_step_predict(
            initial_state=initial_state,
            inputs=inputs,
            measurements_t_in_c=measurements_t_in_c,
            n_steps_ahead=config.mode.n_steps_ahead,
        )
        col = f"t_in_pred_n_step_{config.mode.n_steps_ahead}_c"
        return pd.DataFrame(
            {
                "timestamp": list(timestamps),
                "t_in_measured_c": measurements_t_in_c,
                col: pred.t_in_pred_c,
            }
        )

    raise ValueError(f"Unknown mode kind: {config.mode.kind!r}")


def _build_summary(
    config: SimulationConfig,
    params: FourR2CParameters,
    wall_init_seed: dict,
    initial_state_dict: dict,
    input_csv_path: Path,
    output_path: Path,
    dt_seconds: float,
    n_segment_steps: int,
    n_history_steps: int,
) -> dict:
    return {
        "mode": {
            "kind": config.mode.kind,
            "n_steps_ahead": config.mode.n_steps_ahead,
        },
        "selection": {
            "city": config.selection.city,
            "house_name": config.selection.house_name,
        },
        "window": {
            "start": config.window.start.isoformat(),
            "duration_hours": config.window.duration_hours,
            "end": (
                config.window.start
                + timedelta(hours=config.window.duration_hours)
            ).isoformat(),
        },
        "history_hours": config.history_hours,
        "wall_init": {
            "method": "warm_start",
            "steady_state_seed": wall_init_seed,
        },
        "inputs": {
            "source": config.inputs.source,
            "csv_path": str(input_csv_path),
        },
        "model_path": str(config.model.path),
        "model_parameters": asdict(params),
        "initial_state": initial_state_dict,
        "timestep_seconds": dt_seconds,
        "n_segment_steps": n_segment_steps,
        "n_history_steps": n_history_steps,
        "output_csv_path": str(output_path),
    }
