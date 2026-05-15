from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from three_r2c.models.types import FourR2CInput, FourR2CParameters
from three_r2c.sysid.ekf import AugmentedStateEKF, AugmentedStateIndex, EKFResult
from three_r2c.sysid.evaluator import EKFEvaluator, PredictionMetrics
from three_r2c.sysid.initial_state_optimizer import (
    InitialStateOptimizationResult,
    InitialStateOptimizer,
)
from three_r2c.sysid.parameter_grid import (
    ParameterCandidateGrid,
    generate_parameter_candidates,
)


@dataclass(frozen=True)
class EKFNoiseConfig:
    """
    Noise and covariance settings for the augmented-state EKF.
    """

    q: np.ndarray
    r: np.ndarray
    p0: np.ndarray


@dataclass(frozen=True)
class SysIDTrainingResult:
    """
    Best result across all candidate parameter initializations.
    """

    best_parameters: FourR2CParameters
    best_initial_state_result: InitialStateOptimizationResult
    best_ekf_result: EKFResult
    best_metrics: PredictionMetrics
    best_objective_value: float
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
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        history_t_oa_c: np.ndarray,
        history_t_in_c: np.ndarray,
        dt_seconds: float,
        n_steps_ahead: int,
    ) -> SysIDTrainingResult:
        """
        Run full training across all parameter candidates.

        Parameters
        ----------
        parameter_grid
            Candidate values for each independent model parameter.
        inputs
            Input sequence for the training segment.
        measurements_t_in_c
            Indoor-air temperature measurements for the training segment.
        history_t_oa_c
            Outdoor temperature history before segment start.
        history_t_in_c
            Indoor temperature history before segment start.
        dt_seconds
            Simulation timestep in seconds.
        n_steps_ahead
            Horizon for n-step-ahead RMSE evaluation.

        Returns
        -------
        SysIDTrainingResult
        """
        candidates = generate_parameter_candidates(parameter_grid)

        if len(candidates) == 0:
            raise ValueError("No parameter candidates generated.")

        best_parameters: FourR2CParameters | None = None
        best_initial_state_result: InitialStateOptimizationResult | None = None
        best_ekf_result: EKFResult | None = None
        best_metrics: PredictionMetrics | None = None
        best_objective_value = np.inf
        best_candidate_index = -1

        evaluator = EKFEvaluator(state_index=self.state_index)

        for i, params in enumerate(candidates):
            initial_state_result = self._optimize_initial_state(
                params=params,
                inputs=inputs,
                measurements_t_in_c=measurements_t_in_c,
                history_t_oa_c=history_t_oa_c,
                history_t_in_c=history_t_in_c,
                dt_seconds=dt_seconds,
            )

            z0 = self._build_initial_augmented_state(
                params=params,
                t_in_0_c=initial_state_result.initial_condition.t_in_0_c,
                t_iw_0_c=initial_state_result.initial_condition.t_iw_0_c,
                t_ow_0_c=initial_state_result.initial_condition.t_ow_0_c,
            )

            ekf = AugmentedStateEKF(
                q=self.ekf_noise_config.q,
                r=self.ekf_noise_config.r,
                state_index=self.state_index,
            )

            ekf_result = ekf.run(
                inputs=inputs,
                measurements_t_in_c=np.asarray(measurements_t_in_c, dtype=float),
                z0=z0,
                p0=self.ekf_noise_config.p0,
                dt_seconds=dt_seconds,
            )

            metrics = evaluator.evaluate(
                ekf_result=ekf_result,
                inputs=inputs,
                measurements_t_in_c=np.asarray(measurements_t_in_c, dtype=float),
                dt_seconds=dt_seconds,
                n_steps_ahead=n_steps_ahead,
            )

            objective_value = self._compute_objective(metrics)

            if objective_value < best_objective_value:
                best_parameters = params
                best_initial_state_result = initial_state_result
                best_ekf_result = ekf_result
                best_metrics = metrics
                best_objective_value = objective_value
                best_candidate_index = i

        if (
            best_parameters is None
            or best_initial_state_result is None
            or best_ekf_result is None
            or best_metrics is None
        ):
            raise RuntimeError("Training failed to produce any valid candidate result.")

        return SysIDTrainingResult(
            best_parameters=best_parameters,
            best_initial_state_result=best_initial_state_result,
            best_ekf_result=best_ekf_result,
            best_metrics=best_metrics,
            best_objective_value=float(best_objective_value),
            best_candidate_index=best_candidate_index,
            n_candidates_evaluated=len(candidates),
        )

    def _optimize_initial_state(
        self,
        params: FourR2CParameters,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        history_t_oa_c: np.ndarray,
        history_t_in_c: np.ndarray,
        dt_seconds: float,
    ) -> InitialStateOptimizationResult:
        optimizer = InitialStateOptimizer(params=params)

        return optimizer.optimize(
            inputs=inputs,
            measurements_t_in_c=np.asarray(measurements_t_in_c, dtype=float),
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
        n = self.state_index.n_states

        q = np.asarray(self.ekf_noise_config.q, dtype=float)
        r = np.asarray(self.ekf_noise_config.r, dtype=float)
        p0 = np.asarray(self.ekf_noise_config.p0, dtype=float)

        if q.shape != (n, n):
            raise ValueError(f"Q must have shape {(n, n)}, got {q.shape}")

        if r.shape != (1, 1):
            raise ValueError(f"R must have shape (1, 1), got {r.shape}")

        if p0.shape != (n, n):
            raise ValueError(f"P0 must have shape {(n, n)}, got {p0.shape}")

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