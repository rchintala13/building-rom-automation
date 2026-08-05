from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from rom_automation.logging_utils import get_logger
from rom_automation.models.types import FourR2CParameters
from rom_automation.sysid.config_types import SysIDConfig
from rom_automation.sysid.noise_builders import build_q_from_process_noise
from rom_automation.sysid.dataset_adapter import (
    DatasetSegment,
    SysIDSplitDataset,
    build_sysid_split_dataset,
)
from rom_automation.sysid.ekf import AugmentedStateIndex
from rom_automation.sysid.noise_builders import EKFNoiseConfig
from rom_automation.sysid.trainer import (
    CandidateQDiag,
    CandidateSegmentPDiag,
    EKFSysIDTrainer,
    SegmentResult,
)


def run_sysid_ekf_workflow(
    processed_csv_path: str | Path,
    output_dir: str | Path,
    cfg: SysIDConfig,
) -> None:
    """
    Run end-to-end EKF-based system identification from one processed CSV.

    Parameters
    ----------
    processed_csv_path
        Path to processed CSV, typically processed_5min.csv.
    output_dir
        Directory where sysid results will be written.
    cfg
        Strongly-typed sysid configuration (see config_types.SysIDConfig).
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

    split_dataset = build_sysid_split_dataset(
        df=df,
        history_hours=cfg.dataset.history_hours,
        train_fraction=cfg.dataset.splits.train,
        val_fraction=cfg.dataset.splits.val,
        test_fraction=cfg.dataset.splits.test,
    )
    timestep_seconds = split_dataset.timestep_seconds
    logger.info("Detected timestep: %s seconds", timestep_seconds)
    logger.info(
        "Built split dataset: history=%d, train=%d, val=%d, test=%d steps.",
        len(split_dataset.history_t_in_c),
        len(split_dataset.train.inputs),
        len(split_dataset.val.inputs) if split_dataset.val is not None else 0,
        len(split_dataset.test.inputs),
    )

    state_index = AugmentedStateIndex()

    ekf_noise_config = EKFNoiseConfig(
        r=np.array([[cfg.ekf.r_value]], dtype=float),
        p0_spec=cfg.ekf.p0,
        process_noise_spec=cfg.ekf.process_noise,
    )

    trainer = EKFSysIDTrainer(
        ekf_noise_config=ekf_noise_config,
        state_index=state_index,
        objective_weights=(
            cfg.ekf.objective_weights.one_step,
            cfg.ekf.objective_weights.n_step,
        ),
    )

    logger.info("Starting EKF sysid training.")
    result = trainer.train(
        parameter_grid=cfg.parameter_grid,
        split_dataset=split_dataset,
        n_steps_ahead=cfg.ekf.n_steps_ahead,
        bound_fractions=cfg.parameter_bounds,
    )
    logger.info("Finished EKF sysid training.")

    final_identified_parameters = extract_final_identified_parameters(
        z_filtered=result.best_train_segment.ekf_result.z_filtered,
        state_index=state_index,
    )

    selection_segment_name = "val" if result.best_val_segment is not None else "test"

    summary = {
        "processed_csv_path": str(processed_csv_path),
        "history_hours": cfg.dataset.history_hours,
        "timestep_seconds": timestep_seconds,
        "n_steps_ahead": cfg.ekf.n_steps_ahead,
        "objective_weights": {
            "one_step": cfg.ekf.objective_weights.one_step,
            "n_step": cfg.ekf.objective_weights.n_step,
        },
        "splits": {
            "train_fraction": cfg.dataset.splits.train,
            "val_fraction": cfg.dataset.splits.val,
            "test_fraction": cfg.dataset.splits.test,
            "train_steps": len(split_dataset.train.inputs),
            "val_steps": (
                len(split_dataset.val.inputs) if split_dataset.val is not None else 0
            ),
            "test_steps": len(split_dataset.test.inputs),
        },
        "selection_segment": selection_segment_name,
        "n_candidates_evaluated": result.n_candidates_evaluated,
        "best_candidate_index": result.best_candidate_index,
        "best_selection_objective": result.best_selection_objective,
        "best_metrics": {
            "train": _metrics_to_dict(result.best_train_segment.metrics),
            "val": (
                _metrics_to_dict(result.best_val_segment.metrics)
                if result.best_val_segment is not None
                else None
            ),
            "test": _metrics_to_dict(result.best_test_segment.metrics),
        },
        "best_initial_parameter_guess": asdict(result.best_initial_guess),
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

    model_path = output_dir / "model.json"
    logger.info("Writing identified model parameters: %s", model_path)
    model_payload = asdict(final_identified_parameters)
    model_payload["init_alpha_iw"] = float(result.best_initial_state_result.alpha_iw)
    model_payload["init_alpha_ow"] = float(result.best_initial_state_result.alpha_ow)

    # State-observer ingredients for the MPC. We store the temperature-block
    # process-noise covariance Q, the measurement noise R, the timestep, and the
    # observed one-step innovation std (the selection segment's one-step RMSE).
    # The MPC builds the observer gain from these -- either a fixed inflation of Q
    # or, by default, an innovation-consistent scaling that matches this observed
    # innovation -- so the observer retunes automatically as the model improves.
    selection_segment = (
        result.best_val_segment
        if result.best_val_segment is not None
        else result.best_test_segment
    )
    model_payload["observer"] = _build_observer_block(
        params=final_identified_parameters,
        dt_seconds=timestep_seconds,
        cfg=cfg,
        innovation_std_c=float(selection_segment.metrics.rmse_one_step_c),
    )

    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    filtered_states_path = output_dir / "ekf_filtered_augmented_states.csv"
    logger.info("Writing filtered augmented states: %s", filtered_states_path)
    filtered_df = _concat_filtered_states(
        split_dataset=split_dataset,
        train_segment=result.best_train_segment,
        val_segment=result.best_val_segment,
        test_segment=result.best_test_segment,
        state_index=state_index,
    )
    filtered_df.to_csv(filtered_states_path, index=False)

    one_step_path = output_dir / "one_step_predictions.csv"
    logger.info("Writing one-step predictions: %s", one_step_path)
    pred_df = _concat_one_step_predictions(
        split_dataset=split_dataset,
        train_segment=result.best_train_segment,
        val_segment=result.best_val_segment,
        test_segment=result.best_test_segment,
    )
    pred_df.to_csv(one_step_path, index=False)

    p_diag_path = output_dir / "p_diagonals_last_candidate.csv"
    logger.info(
        "Writing P-diagonals for last candidate (%d segment records): %s",
        len(result.last_p_diag_records),
        p_diag_path,
    )
    p_diag_df = _build_p_diag_history_frame(
        records=result.last_p_diag_records,
        split_dataset=split_dataset,
        state_index=state_index,
    )
    p_diag_df.to_csv(p_diag_path, index=False)

    q_diag_path = output_dir / "q_diagonal_last_candidate.csv"
    logger.info("Writing Q-diagonal for last candidate: %s", q_diag_path)
    q_diag_df = _build_q_diag_frame(
        record=result.last_q_diag_record,
        state_index=state_index,
    )
    q_diag_df.to_csv(q_diag_path, index=False)

    logger.info("EKF sysid workflow complete.")


def _metrics_to_dict(metrics) -> dict:
    return {
        "rmse_one_step_c": metrics.rmse_one_step_c,
        "rmse_n_step_c": metrics.rmse_n_step_c,
    }


def _build_observer_block(
    params: FourR2CParameters,
    dt_seconds: float,
    cfg: SysIDConfig,
    innovation_std_c: float,
) -> dict:
    """
    Build the MPC state-observer ingredients: the temperature-block process-noise
    covariance Q (its structure sets how the T_in innovation is distributed to the
    walls), the measurement noise R, the timestep, and the observed one-step
    innovation std (the target for innovation-consistent gain tuning in the MPC).
    """
    q_full = build_q_from_process_noise(
        params=params,
        dt_seconds=dt_seconds,
        q_in_std_kw=cfg.ekf.process_noise.q_in_std_kw,
        q_iw_std_kw=cfg.ekf.process_noise.q_iw_std_kw,
        q_ow_std_kw=cfg.ekf.process_noise.q_ow_std_kw,
    )
    q_temp = q_full[0:3, 0:3]

    return {
        "q_temp": q_temp.tolist(),
        "r": float(cfg.ekf.r_value),
        "dt_seconds": float(dt_seconds),
        "innovation_std_c": float(innovation_std_c),
    }


def _concat_filtered_states(
    split_dataset: SysIDSplitDataset,
    train_segment: SegmentResult,
    val_segment: SegmentResult | None,
    test_segment: SegmentResult,
    state_index: AugmentedStateIndex,
) -> pd.DataFrame:
    columns = augmented_state_column_names(state_index)
    frames: list[pd.DataFrame] = []

    frames.append(
        _segment_frame(
            timestamps=split_dataset.train.timestamps,
            z_filtered=train_segment.ekf_result.z_filtered,
            columns=columns,
            label="train",
        )
    )
    if val_segment is not None:
        assert split_dataset.val is not None
        frames.append(
            _segment_frame(
                timestamps=split_dataset.val.timestamps,
                z_filtered=val_segment.ekf_result.z_filtered,
                columns=columns,
                label="val",
            )
        )
    frames.append(
        _segment_frame(
            timestamps=split_dataset.test.timestamps,
            z_filtered=test_segment.ekf_result.z_filtered,
            columns=columns,
            label="test",
        )
    )

    return pd.concat(frames, ignore_index=True)


def _segment_frame(
    timestamps: pd.DatetimeIndex,
    z_filtered: np.ndarray,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    df = pd.DataFrame(z_filtered, columns=columns)
    df.insert(0, "segment", label)
    df.insert(1, "timestamp", list(timestamps))
    return df


def _concat_one_step_predictions(
    split_dataset: SysIDSplitDataset,
    train_segment: SegmentResult,
    val_segment: SegmentResult | None,
    test_segment: SegmentResult,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    frames.append(
        _one_step_frame(
            label="train",
            segment=split_dataset.train,
            segment_result=train_segment,
        )
    )
    if val_segment is not None:
        assert split_dataset.val is not None
        frames.append(
            _one_step_frame(
                label="val",
                segment=split_dataset.val,
                segment_result=val_segment,
            )
        )
    frames.append(
        _one_step_frame(
            label="test",
            segment=split_dataset.test,
            segment_result=test_segment,
        )
    )

    return pd.concat(frames, ignore_index=True)


def _one_step_frame(
    label: str,
    segment: DatasetSegment,
    segment_result: SegmentResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": label,
            "timestamp": list(segment.timestamps),
            "t_in_measured_c": segment.measurements_t_in_c,
            "t_in_pred_one_step_c": segment_result.ekf_result.y_pred_one_step,
        }
    )


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


def _build_p_diag_history_frame(
    records: list[CandidateSegmentPDiag],
    split_dataset: SysIDSplitDataset,
    state_index: AugmentedStateIndex,
) -> pd.DataFrame:
    """
    Flatten the per-candidate P-diagonal records into one long-form DataFrame.

    Each (candidate_index, segment) pair contributes n_steps rows. The state
    diagonals are written one column per augmented state; the row ordering is:
    all train rows for candidate 0, then val (if any), then test, then
    candidate 1, and so on.
    """
    state_columns = augmented_state_column_names(state_index)

    segment_timestamps = {
        "train": split_dataset.train.timestamps,
        "val": split_dataset.val.timestamps if split_dataset.val is not None else None,
        "test": split_dataset.test.timestamps,
    }

    frames: list[pd.DataFrame] = []

    for record in records:
        timestamps = segment_timestamps.get(record.segment)
        if timestamps is None:
            raise RuntimeError(
                f"Got P-diag record for segment {record.segment!r} but no "
                "matching segment in the split dataset."
            )

        if record.p_diag.shape[0] != len(timestamps):
            raise RuntimeError(
                f"P-diag shape {record.p_diag.shape} does not match "
                f"{record.segment} timestamps length {len(timestamps)} "
                f"for candidate {record.candidate_index}."
            )

        frame = pd.DataFrame(record.p_diag, columns=state_columns)
        frame.insert(0, "candidate_index", record.candidate_index)
        frame.insert(1, "segment", record.segment)
        frame.insert(2, "timestamp", list(timestamps))
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["candidate_index", "segment", "timestamp", *state_columns])

    return pd.concat(frames, ignore_index=True)


def _build_q_diag_frame(
    record: CandidateQDiag,
    state_index: AugmentedStateIndex,
) -> pd.DataFrame:
    """
    Build a one-row DataFrame with the Q-diagonal entries for the last
    candidate. Q is built once per candidate (constant across segments), so
    a single row per candidate is sufficient.
    """
    state_columns = augmented_state_column_names(state_index)

    if record.q_diag.shape != (len(state_columns),):
        raise RuntimeError(
            f"Q-diag shape {record.q_diag.shape} does not match expected "
            f"({len(state_columns)},) for candidate {record.candidate_index}."
        )

    row = {"candidate_index": record.candidate_index}
    for name, value in zip(state_columns, record.q_diag):
        row[name] = float(value)

    return pd.DataFrame([row])


