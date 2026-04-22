from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.types import FourR2CInput, FourR2CParameters, FourR2CState
from rom_automation.sysid.ekf import AugmentedStateIndex, EKFResult


@dataclass(frozen=True)
class PredictionMetrics:
    rmse_one_step_c: float
    rmse_n_step_c: float


class EKFEvaluator:
    """
    Evaluate EKF-based system identification performance.

    Metrics
    -------
    - one-step-ahead RMSE:
        uses the EKF one-step prediction output directly

    - n-step-ahead RMSE:
        rolls the model forward open-loop over a horizon using the filtered
        state and filtered parameters at the horizon start
    """

    def __init__(
        self,
        state_index: AugmentedStateIndex | None = None,
    ) -> None:
        self.idx = state_index or AugmentedStateIndex()

    def evaluate(
        self,
        ekf_result: EKFResult,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        dt_seconds: float,
        n_steps_ahead: int,
    ) -> PredictionMetrics:
        """
        Compute one-step-ahead and n-step-ahead RMSE.

        Parameters
        ----------
        ekf_result
            Output from the augmented-state EKF.
        inputs
            Input sequence used by the model.
        measurements_t_in_c
            Measured indoor air temperature sequence, shape (N,).
        dt_seconds
            Simulation timestep in seconds.
        n_steps_ahead
            Open-loop prediction horizon for n-step-ahead RMSE.

        Returns
        -------
        PredictionMetrics
        """
        y_true = np.asarray(measurements_t_in_c, dtype=float).reshape(-1)

        if y_true.shape[0] != len(inputs):
            raise ValueError(
                "measurements_t_in_c length must match inputs length. "
                f"Got {y_true.shape[0]} and {len(inputs)}."
            )

        if ekf_result.y_pred_one_step.shape[0] != y_true.shape[0]:
            raise ValueError(
                "EKF one-step prediction length must match measurement length. "
                f"Got {ekf_result.y_pred_one_step.shape[0]} and {y_true.shape[0]}."
            )

        rmse_one_step = compute_rmse(
            y_true=y_true,
            y_pred=ekf_result.y_pred_one_step,
        )

        y_pred_n = self.compute_n_step_predictions(
            ekf_result=ekf_result,
            inputs=inputs,
            dt_seconds=dt_seconds,
            n_steps_ahead=n_steps_ahead,
        )

        valid_mask = ~np.isnan(y_pred_n)

        if not np.any(valid_mask):
            raise ValueError(
                "No valid n-step-ahead predictions were produced. "
                "Check n_steps_ahead and sequence length."
            )

        rmse_n_step = compute_rmse(
            y_true=y_true[valid_mask],
            y_pred=y_pred_n[valid_mask],
        )

        return PredictionMetrics(
            rmse_one_step_c=rmse_one_step,
            rmse_n_step_c=rmse_n_step,
        )

    def compute_n_step_predictions(
        self,
        ekf_result: EKFResult,
        inputs: list[FourR2CInput],
        dt_seconds: float,
        n_steps_ahead: int,
    ) -> np.ndarray:
        """
        Compute n-step-ahead open-loop predictions of indoor air temperature.

        For each valid starting time k, this method:
        1. takes the filtered augmented state at time k
        2. extracts the thermal states and parameters
        3. rolls the model forward open-loop for n_steps_ahead steps
        4. stores the predicted indoor temperature at time k + n_steps_ahead

        Returns
        -------
        np.ndarray
            Array of shape (N,) with NaN where prediction is unavailable.
        """
        if n_steps_ahead <= 0:
            raise ValueError(
                f"n_steps_ahead must be positive. Got {n_steps_ahead}."
            )

        n_total = len(inputs)
        y_pred_n = np.full(n_total, np.nan, dtype=float)

        z_filtered = np.asarray(ekf_result.z_filtered, dtype=float)

        if z_filtered.shape[0] != n_total:
            raise ValueError(
                "EKF filtered state length must match inputs length. "
                f"Got {z_filtered.shape[0]} and {n_total}."
            )

        for k in range(n_total - n_steps_ahead):
            z_k = z_filtered[k, :]

            state = self._extract_state(z_k)
            params = self._extract_parameters(z_k)
            model = FourR2CModel(params=params)

            x = state
            for j in range(n_steps_ahead):
                u_j = inputs[k + j]
                x = model.step(
                    state=x,
                    inp=u_j,
                    dt_seconds=dt_seconds,
                )

            y_pred_n[k + n_steps_ahead] = x.t_in_c

        return y_pred_n

    def _extract_state(self, z: np.ndarray) -> FourR2CState:
        return FourR2CState(
            t_in_c=float(z[self.idx.t_in_c]),
            t_iw_c=float(z[self.idx.t_iw_c]),
            t_ow_c=float(z[self.idx.t_ow_c]),
        )

    def _extract_parameters(self, z: np.ndarray) -> FourR2CParameters:
        return FourR2CParameters(
            r_in_iw=float(z[self.idx.r_in_iw]),
            r_iw_ow=float(z[self.idx.r_iw_ow]),
            r_ow_oa=float(z[self.idx.r_ow_oa]),
            r_in_oa=float(z[self.idx.r_in_oa]),
            c_in=float(z[self.idx.c_in]),
            c_w=float(z[self.idx.c_w]),
            alpha_ghi_outer_wall=float(z[self.idx.alpha_ghi_outer_wall]),
            alpha_ghi_inner_wall=float(z[self.idx.alpha_ghi_inner_wall]),
        )


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute root-mean-square error.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))