from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rom_automation.models.types import FourR2CParameters
from rom_automation.sysid.dataset_adapter import DatasetSegment, SysIDSplitDataset
from rom_automation.sysid.ekf import AugmentedStateEKF, AugmentedStateIndex, EKFResult
from rom_automation.sysid.evaluator import EKFEvaluator, PredictionMetrics
from rom_automation.sysid.initial_state_optimizer import (
    InitialStateOptimizationResult,
    InitialStateOptimizer,
)
from rom_automation.sysid.noise_builders import (
    EKFNoiseConfig,
    build_p0_from_ratios,
    build_q_from_process_noise,
)
from rom_automation.sysid.parameter_grid import (
    ParameterBoundFractions,
    ParameterCandidateGrid,
    compute_parameter_bounds,
    generate_parameter_candidates,
)


@dataclass(frozen=True)
class SegmentResult:
    """
    EKF result + metrics for a single data segment (train, val, or test).
    """

    ekf_result: EKFResult
    metrics: PredictionMetrics


@dataclass(frozen=True)
class SysIDTrainingResult:
    """
    Best result across all candidate parameter initializations.

    `best_initial_guess` is the candidate that won the selection objective
    (val metrics if val exists, else test metrics). `best_train_segment`
    carries the parameter identification trajectory; `best_val_segment` and
    `best_test_segment` are run with parameters frozen at the end-of-train
    values via the EKF bounds-clip mechanism.
    """

    best_initial_guess: FourR2CParameters
    best_initial_state_result: InitialStateOptimizationResult
    best_train_segment: SegmentResult
    best_val_segment: SegmentResult | None
    best_test_segment: SegmentResult
    best_selection_objective: float
    best_candidate_index: int
    n_candidates_evaluated: int


class EKFSysIDTrainer:
    """
    Train the 4R2C model using:
    - parameter candidate grid
    - history-based initial-state alpha optimization
    - augmented-state EKF
    - one-step and n-step RMSE evaluation
    """

    def __init__(
        self,
        ekf_noise_config: EKFNoiseConfig,
        state_index: AugmentedStateIndex | None = None,
        objective_weights: tuple[float, float] = (0.3, 0.7),
    ) -> None:
        self.ekf_noise_config = ekf_noise_config
        self.state_index = state_index or AugmentedStateIndex()
        self.objective_weights = objective_weights

        self._validate_noise_config()
        self._validate_objective_weights()

    def train(
        self,
        parameter_grid: ParameterCandidateGrid,
        split_dataset: SysIDSplitDataset,
        n_steps_ahead: int,
        bound_fractions: ParameterBoundFractions | None = None,
    ) -> SysIDTrainingResult:
        """
        Run full training + held-out evaluation across all parameter candidates.

        For each candidate:
          1. EKF runs on the training segment (parameters get identified).
          2. EKF continues on val (if present) and test, with parameters
             frozen at end-of-train values via zero-width bound clipping.
          3. Metrics are computed separately on each segment.

        Candidate selection uses the validation segment's objective; if no
        validation segment is provided, the test segment's objective is used.

        Parameters
        ----------
        parameter_grid
            Candidate values for each independent model parameter.
        split_dataset
            Pre-split dataset (train, optional val, test, shared history).
        n_steps_ahead
            Horizon for n-step-ahead RMSE evaluation.
        bound_fractions
            Per-parameter frac/multiple used during the training EKF for
            soft clipping. None disables clipping during training.

        Returns
        -------
        SysIDTrainingResult
        """
        candidates = generate_parameter_candidates(parameter_grid)

        if len(candidates) == 0:
            raise ValueError("No parameter candidates generated.")

        dt_seconds = split_dataset.timestep_seconds

        best_initial_guess: FourR2CParameters | None = None
        best_initial_state_result: InitialStateOptimizationResult | None = None
        best_train_segment: SegmentResult | None = None
        best_val_segment: SegmentResult | None = None
        best_test_segment: SegmentResult | None = None
        best_selection_objective = np.inf
        best_candidate_index = -1

        evaluator = EKFEvaluator(state_index=self.state_index)

        for i, params in enumerate(candidates):
            initial_state_result = self._optimize_initial_state(
                params=params,
                segment=split_dataset.train,
                history_t_oa_c=split_dataset.history_t_oa_c,
                history_t_in_c=split_dataset.history_t_in_c,
                dt_seconds=dt_seconds,
            )

            z0 = self._build_initial_augmented_state(
                params=params,
                t_in_0_c=initial_state_result.initial_condition.t_in_0_c,
                t_iw_0_c=initial_state_result.initial_condition.t_iw_0_c,
                t_ow_0_c=initial_state_result.initial_condition.t_ow_0_c,
            )

            p0 = build_p0_from_ratios(
                z0=z0,
                temperature_std_c=self.ekf_noise_config.p0_spec.temperature_std_c,
                resistance_std_ratio=self.ekf_noise_config.p0_spec.resistance_std_ratio,
                capacitance_std_ratio=self.ekf_noise_config.p0_spec.capacitance_std_ratio,
                alpha_std_ratio=self.ekf_noise_config.p0_spec.alpha_std_ratio,
                alpha_std_floor=self.ekf_noise_config.p0_spec.alpha_std_floor,
                state_index=self.state_index,
            )

            q = build_q_from_process_noise(
                params=params,
                dt_seconds=dt_seconds,
                q_in_std_kw=self.ekf_noise_config.process_noise_spec.q_in_std_kw,
                q_iw_std_kw=self.ekf_noise_config.process_noise_spec.q_iw_std_kw,
                q_ow_std_kw=self.ekf_noise_config.process_noise_spec.q_ow_std_kw,
                state_index=self.state_index,
            )

            ekf = AugmentedStateEKF(
                q=q,
                r=self.ekf_noise_config.r,
                state_index=self.state_index,
            )

            if bound_fractions is None:
                train_z_lower = None
                train_z_upper = None
            else:
                train_z_lower, train_z_upper = self._build_augmented_bounds(
                    bound_fractions=bound_fractions,
                    params=params,
                )

            # 1) Training: full augmented EKF
            train_ekf_result = ekf.run(
                inputs=split_dataset.train.inputs,
                measurements_t_in_c=split_dataset.train.measurements_t_in_c,
                z0=z0,
                p0=p0,
                dt_seconds=dt_seconds,
                z_lower=train_z_lower,
                z_upper=train_z_upper,
            )

            train_metrics = evaluator.evaluate(
                ekf_result=train_ekf_result,
                inputs=split_dataset.train.inputs,
                measurements_t_in_c=split_dataset.train.measurements_t_in_c,
                dt_seconds=dt_seconds,
                n_steps_ahead=n_steps_ahead,
            )

            # 2) Held-out evaluation: continue EKF with parameters pinned
            z_end_train = train_ekf_result.z_filtered[-1, :]
            p_end_train = train_ekf_result.p_filtered[-1, :, :]
            frozen_lower, frozen_upper = self._build_frozen_parameter_bounds(
                z_identified=z_end_train,
            )

            if split_dataset.val is not None:
                val_ekf_result = ekf.run(
                    inputs=split_dataset.val.inputs,
                    measurements_t_in_c=split_dataset.val.measurements_t_in_c,
                    z0=z_end_train,
                    p0=p_end_train,
                    dt_seconds=dt_seconds,
                    z_lower=frozen_lower,
                    z_upper=frozen_upper,
                )
                val_metrics = evaluator.evaluate(
                    ekf_result=val_ekf_result,
                    inputs=split_dataset.val.inputs,
                    measurements_t_in_c=split_dataset.val.measurements_t_in_c,
                    dt_seconds=dt_seconds,
                    n_steps_ahead=n_steps_ahead,
                )
                val_segment = SegmentResult(
                    ekf_result=val_ekf_result,
                    metrics=val_metrics,
                )
                z_test_start = val_ekf_result.z_filtered[-1, :]
                p_test_start = val_ekf_result.p_filtered[-1, :, :]
            else:
                val_segment = None
                z_test_start = z_end_train
                p_test_start = p_end_train

            test_ekf_result = ekf.run(
                inputs=split_dataset.test.inputs,
                measurements_t_in_c=split_dataset.test.measurements_t_in_c,
                z0=z_test_start,
                p0=p_test_start,
                dt_seconds=dt_seconds,
                z_lower=frozen_lower,
                z_upper=frozen_upper,
            )
            test_metrics = evaluator.evaluate(
                ekf_result=test_ekf_result,
                inputs=split_dataset.test.inputs,
                measurements_t_in_c=split_dataset.test.measurements_t_in_c,
                dt_seconds=dt_seconds,
                n_steps_ahead=n_steps_ahead,
            )

            train_segment = SegmentResult(
                ekf_result=train_ekf_result,
                metrics=train_metrics,
            )
            test_segment = SegmentResult(
                ekf_result=test_ekf_result,
                metrics=test_metrics,
            )

            # 3) Selection objective: val if available, else test
            selection_metrics = (
                val_segment.metrics if val_segment is not None else test_metrics
            )
            selection_objective = self._compute_objective(selection_metrics)

            if selection_objective < best_selection_objective:
                best_initial_guess = params
                best_initial_state_result = initial_state_result
                best_train_segment = train_segment
                best_val_segment = val_segment
                best_test_segment = test_segment
                best_selection_objective = selection_objective
                best_candidate_index = i

        if (
            best_initial_guess is None
            or best_initial_state_result is None
            or best_train_segment is None
            or best_test_segment is None
        ):
            raise RuntimeError("Training failed to produce any valid candidate result.")

        return SysIDTrainingResult(
            best_initial_guess=best_initial_guess,
            best_initial_state_result=best_initial_state_result,
            best_train_segment=best_train_segment,
            best_val_segment=best_val_segment,
            best_test_segment=best_test_segment,
            best_selection_objective=float(best_selection_objective),
            best_candidate_index=best_candidate_index,
            n_candidates_evaluated=len(candidates),
        )

    def _optimize_initial_state(
        self,
        params: FourR2CParameters,
        segment: DatasetSegment,
        history_t_oa_c: np.ndarray,
        history_t_in_c: np.ndarray,
        dt_seconds: float,
    ) -> InitialStateOptimizationResult:
        optimizer = InitialStateOptimizer(params=params)

        return optimizer.optimize(
            inputs=segment.inputs,
            measurements_t_in_c=np.asarray(segment.measurements_t_in_c, dtype=float),
            history_t_oa_c=np.asarray(history_t_oa_c, dtype=float),
            history_t_in_c=np.asarray(history_t_in_c, dtype=float),
            dt_seconds=dt_seconds,
        )

    def _build_initial_augmented_state(
        self,
        params: FourR2CParameters,
        t_in_0_c: float,
        t_iw_0_c: float,
        t_ow_0_c: float,
    ) -> np.ndarray:
        """
        Build initial augmented EKF state vector.
        """
        z0 = np.zeros(self.state_index.n_states, dtype=float)

        z0[self.state_index.t_in_c] = t_in_0_c
        z0[self.state_index.t_iw_c] = t_iw_0_c
        z0[self.state_index.t_ow_c] = t_ow_0_c

        z0[self.state_index.r_in_iw] = params.r_in_iw
        z0[self.state_index.r_iw_ow] = params.r_iw_ow
        z0[self.state_index.r_ow_oa] = params.r_ow_oa
        z0[self.state_index.r_in_oa] = params.r_in_oa
        z0[self.state_index.c_in] = params.c_in
        z0[self.state_index.c_w] = params.c_w
        z0[self.state_index.alpha_ghi_outer_wall] = params.alpha_ghi_outer_wall
        z0[self.state_index.alpha_ghi_inner_wall] = params.alpha_ghi_inner_wall

        return z0

    def _build_augmented_bounds(
        self,
        bound_fractions: ParameterBoundFractions,
        params: FourR2CParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build per-state lower/upper bound vectors for the augmented EKF state.

        Temperatures are left unbounded (-inf/+inf). Each parameter is bounded
        by [frac * initial_guess, multiple * initial_guess] from `params`.
        """
        n = self.state_index.n_states
        z_lower = np.full(n, -np.inf, dtype=float)
        z_upper = np.full(n, np.inf, dtype=float)

        bounds = compute_parameter_bounds(
            fractions=bound_fractions,
            initial_guess=params,
        )

        z_lower[self.state_index.r_in_iw], z_upper[self.state_index.r_in_iw] = bounds.r_in_iw
        z_lower[self.state_index.r_iw_ow], z_upper[self.state_index.r_iw_ow] = bounds.r_iw_ow
        z_lower[self.state_index.r_ow_oa], z_upper[self.state_index.r_ow_oa] = bounds.r_ow_oa
        z_lower[self.state_index.r_in_oa], z_upper[self.state_index.r_in_oa] = bounds.r_in_oa
        z_lower[self.state_index.c_in], z_upper[self.state_index.c_in] = bounds.c_in
        z_lower[self.state_index.c_w], z_upper[self.state_index.c_w] = bounds.c_w
        z_lower[self.state_index.alpha_ghi_outer_wall], z_upper[self.state_index.alpha_ghi_outer_wall] = bounds.alpha_ghi_outer_wall
        z_lower[self.state_index.alpha_ghi_inner_wall], z_upper[self.state_index.alpha_ghi_inner_wall] = bounds.alpha_ghi_inner_wall

        return z_lower, z_upper

    def _build_frozen_parameter_bounds(
        self,
        z_identified: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build bound vectors that pin parameter states to their identified
        values via zero-width clipping while leaving temperature states free.

        Combined with the EKF's post-update clip, parameters cannot drift
        during held-out evaluation segments.
        """
        n = self.state_index.n_states
        z_lower = np.full(n, -np.inf, dtype=float)
        z_upper = np.full(n, np.inf, dtype=float)

        param_indices = [
            self.state_index.r_in_iw,
            self.state_index.r_iw_ow,
            self.state_index.r_ow_oa,
            self.state_index.r_in_oa,
            self.state_index.c_in,
            self.state_index.c_w,
            self.state_index.alpha_ghi_outer_wall,
            self.state_index.alpha_ghi_inner_wall,
        ]

        for idx in param_indices:
            value = float(z_identified[idx])
            z_lower[idx] = value
            z_upper[idx] = value

        return z_lower, z_upper

    def _compute_objective(self, metrics: PredictionMetrics) -> float:
        """
        Weighted objective for model selection.
        """
        w1, w2 = self.objective_weights
        return (
            w1 * metrics.rmse_one_step_c
            + w2 * metrics.rmse_n_step_c
        )

    def _validate_noise_config(self) -> None:
        r = np.asarray(self.ekf_noise_config.r, dtype=float)

        if r.shape != (1, 1):
            raise ValueError(f"R must have shape (1, 1), got {r.shape}")

        p0_spec = self.ekf_noise_config.p0_spec
        required_temp_keys = {"t_in_c", "t_iw_c", "t_ow_c"}
        missing = required_temp_keys - set(p0_spec.temperature_std_c.keys())
        if missing:
            raise ValueError(
                f"p0_spec.temperature_std_c is missing keys: {sorted(missing)}"
            )

    def _validate_objective_weights(self) -> None:
        if len(self.objective_weights) != 2:
            raise ValueError(
                "objective_weights must contain exactly two values: "
                "(weight_one_step, weight_n_step)."
            )

        w1, w2 = self.objective_weights
        if w1 < 0 or w2 < 0:
            raise ValueError(
                f"Objective weights must be nonnegative. Got {self.objective_weights}."
            )

        if w1 == 0 and w2 == 0:
            raise ValueError("At least one objective weight must be positive.")