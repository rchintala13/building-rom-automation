from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from rom_automation.logging_utils import get_logger
from rom_automation.models.types import FourR2CParameters
from rom_automation.sysid.dataset_adapter import build_sysid_dataset
from rom_automation.sysid.ekf import AugmentedStateIndex
from rom_automation.sysid.parameter_grid import ParameterCandidateGrid
from rom_automation.sysid.trainer import EKFNoiseConfig, EKFSysIDTrainer


def run_sysid_ekf_workflow(
    processed_csv_path: str | Path,
    output_dir: str | Path,
    history_hours: float,
    parameter_grid_dict: dict,
    q_diag: list[float],
    r_value: float,
    p0_diag: list[float],
    n_steps_ahead: int,
    objective_weights: tuple[float, float] = (0.3, 0.7),
) -> None:
    """
    Run end-to-end EKF-based system identification from one processed CSV.

    Parameters
    ----------
    processed_csv_path
        Path to processed CSV, typically processed_5min.csv.
    output_dir
        Directory where sysid results will be written.
    history_hours
        Hours of prior data used to initialize wall temperatures.
    parameter_grid_dict
        Dict of candidate lists for ParameterCandidateGrid.
    q_diag
        Diagonal entries for EKF process noise covariance Q.
    r_value
        Scalar measurement noise variance for R.
    p0_diag
        Diagonal entries for initial covariance P0.
    n_steps_ahead
        Horizon for n-step-ahead RMSE.
    objective_weights
        Weights for one-step and n-step RMSE in model selection.
    """
    processed_csv_path = Path(processed_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        "three_r2c.run_sysid_ekf",
        log_file=Path("logs") / "run_sysid_ekf.log",
    )

    if not processed_csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv_path}")

    logger.info("Reading processed CSV: %s", processed_csv_path)
    df = pd.read_csv(processed_csv_path, parse_dates=["timestamp"], index_col="timestamp")

    timestep_seconds = _detect_timestep_seconds(df.index)
    logger.info("Detected timestep: %s seconds", timestep_seconds)

    dataset = build_sysid_dataset(
        df=df,
        history_hours=history_hours,
    )
    logger.info(
        "Built sysid dataset with %d segment steps and %d history steps.",
        len(dataset.inputs),
        len(dataset.history_t_in_c),
    )

    parameter_grid = ParameterCandidateGrid(**parameter_grid_dict)

    state_index = AugmentedStateIndex()
    n_aug = state_index.n_states

    if len(q_diag) != n_aug:
        raise ValueError(f"q_diag must have length {n_aug}, got {len(q_diag)}")

    if len(p0_diag) != n_aug:
        raise ValueError(f"p0_diag must have length {n_aug}, got {len(p0_diag)}")

    ekf_noise_config = EKFNoiseConfig(
        q=np.diag(np.asarray(q_diag, dtype=float)),
        r=np.array([[float(r_value)]], dtype=float),
        p0=np.diag(np.asarray(p0_diag, dtype=float)),
    )

    trainer = EKFSysIDTrainer(
        ekf_noise_config=ekf_noise_config,
        state_index=state_index,
        objective_weights=objective_weights,
    )

    logger.info("Starting EKF sysid training.")
    result = trainer.train(
        parameter_grid=parameter_grid,
        inputs=dataset.inputs,
        measurements_t_in_c=dataset.measurements_t_in_c,
        history_t_oa_c=dataset.history_t_oa_c,
        history_t_in_c=dataset.history_t_in_c,
        dt_seconds=timestep_seconds,
        n_steps_ahead=n_steps_ahead,
    )
    logger.info("Finished EKF sysid training.")

    final_identified_parameters = extract_final_identified_parameters(
        z_filtered=result.best_ekf_result.z_filtered,
        state_index=state_index,
    )

    summary = {
        "processed_csv_path": str(processed_csv_path),
        "history_hours": history_hours,
        "timestep_seconds": timestep_seconds,
        "n_steps_ahead": n_steps_ahead,
        "objective_weights": {
            "one_step": objective_weights[0],
            "n_step": objective_weights[1],
        },
        "n_candidates_evaluated": result.n_candidates_evaluated,
        "best_candidate_index": result.best_candidate_index,
        "best_objective_value": result.best_objective_value,
        "best_metrics": {
            "rmse_one_step_c": result.best_metrics.rmse_one_step_c,
            "rmse_n_step_c": result.best_metrics.rmse_n_step_c,
        },
        "best_initial_parameter_guess": asdict(result.best_parameters),
        "best_initial_state_result": {
            "alpha_iw": result.best_initial_state_result.alpha_iw,
            "alpha_ow": result.best_initial_state_result.alpha_ow,
            "t_oa_hist_mean_c": result.best_initial_state_result.t_oa_hist_mean_c,
            "t_in_hist_mean_c": result.best_initial_state_result.t_in_hist_mean_c,
            "objective_value": result.best_initial_state_result.objective_value,
            "success": result.best_initial_state_result.success,
            "message": result.best_initial_state_result.message,
            "n_iterations": result.best_initial_state_result.n_iterations,
            "initial_condition": {
                "t_in_0_c": result.best_initial_state_result.initial_condition.t_in_0_c,
                "t_iw_0_c": result.best_initial_state_result.initial_condition.t_iw_0_c,
                "t_ow_0_c": result.best_initial_state_result.initial_condition.t_ow_0_c,
            },
        },
        "final_identified_parameters": asdict(final_identified_parameters),
    }

    summary_path = output_dir / "sysid_summary.json"
    logger.info("Writing summary JSON: %s", summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    filtered_states_path = output_dir / "ekf_filtered_augmented_states.csv"
    logger.info("Writing filtered augmented states: %s", filtered_states_path)
    filtered_df = pd.DataFrame(
        result.best_ekf_result.z_filtered,
        columns=augmented_state_column_names(state_index),
    )
    filtered_df.to_csv(filtered_states_path, index=False)

    one_step_path = output_dir / "one_step_predictions.csv"
    logger.info("Writing one-step predictions: %s", one_step_path)
    pred_df = pd.DataFrame(
        {
            "timestamp": dataset.timestamps,
            "t_in_measured_c": dataset.measurements_t_in_c,
            "t_in_pred_one_step_c": result.best_ekf_result.y_pred_one_step,
        }
    )
    pred_df.to_csv(one_step_path, index=False)

    logger.info("EKF sysid workflow complete.")


def extract_final_identified_parameters(
    z_filtered: np.ndarray,
    state_index: AugmentedStateIndex,
) -> FourR2CParameters:
    """
    Extract final identified parameters from the last filtered augmented state.
    """
    if z_filtered.ndim != 2 or z_filtered.shape[0] == 0:
        raise ValueError("z_filtered must be a non-empty 2D array.")

    z_final = z_filtered[-1, :]

    return FourR2CParameters(
        r_in_iw=float(z_final[state_index.r_in_iw]),
        r_iw_ow=float(z_final[state_index.r_iw_ow]),
        r_ow_oa=float(z_final[state_index.r_ow_oa]),
        r_in_oa=float(z_final[state_index.r_in_oa]),
        c_in=float(z_final[state_index.c_in]),
        c_w=float(z_final[state_index.c_w]),
        alpha_ghi_outer_wall=float(z_final[state_index.alpha_ghi_outer_wall]),
        alpha_ghi_inner_wall=float(z_final[state_index.alpha_ghi_inner_wall]),
    )


def augmented_state_column_names(state_index: AugmentedStateIndex) -> list[str]:
    return [
        "t_in_c",
        "t_iw_c",
        "t_ow_c",
        "r_in_iw",
        "r_iw_ow",
        "r_ow_oa",
        "r_in_oa",
        "c_in",
        "c_w",
        "alpha_ghi_outer_wall",
        "alpha_ghi_inner_wall",
    ]


def _detect_timestep_seconds(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError("Cannot detect timestep from fewer than 2 timestamps.")

    return float(deltas.mode().iloc[0].total_seconds())