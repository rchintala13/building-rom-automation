from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.types import FourR2CInitialCondition, FourR2CInput, FourR2CParameters


@dataclass(frozen=True)
class InitialStateOptimizationResult:
    initial_condition: FourR2CInitialCondition
    alpha_iw: float
    alpha_ow: float
    t_oa_hist_mean_c: float
    t_in_hist_mean_c: float
    objective_value: float
    success: bool
    message: str
    n_iterations: int


class InitialStateOptimizer:
    """
    Optimize alpha parameters used to construct initial wall temperatures
    from recent outdoor and indoor temperature history.

    Initialization formula
    ----------------------
    T_iw(0) = alpha_iw * mean(T_oa_history) + (1 - alpha_iw) * mean(T_in_history)
    T_ow(0) = alpha_ow * mean(T_oa_history) + (1 - alpha_ow) * mean(T_in_history)

    T_in(0) is fixed to the first measured indoor temperature of the current segment.
    """

    def __init__(self, params: FourR2CParameters) -> None:
        self.params = params
        self.model = FourR2CModel(params=params)

    def optimize(
        self,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        history_t_oa_c: np.ndarray,
        history_t_in_c: np.ndarray,
        dt_seconds: float,
        x0: tuple[float, float] | None = None,
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((-3.5, 4.5), (-3.5, 4.5)),
        method: str = "L-BFGS-B",
    ) -> InitialStateOptimizationResult:
        """
        Optimize alpha_iw and alpha_ow.

        Parameters
        ----------
        inputs
            Input sequence for the training segment.
        measurements_t_in_c
            Measured indoor air temperature sequence for the training segment.
        history_t_oa_c
            Outdoor air temperature history prior to the segment start.
        history_t_in_c
            Indoor air temperature history prior to the segment start.
        dt_seconds
            Timestep in seconds.
        x0
            Initial guess for (alpha_iw, alpha_ow). Defaults to (0.5, 0.5).
        bounds
            Bounds for alpha_iw and alpha_ow.
        method
            Optimization method for scipy.optimize.minimize.
        """
        y = np.asarray(measurements_t_in_c, dtype=float).reshape(-1)
        t_oa_hist = np.asarray(history_t_oa_c, dtype=float).reshape(-1)
        t_in_hist = np.asarray(history_t_in_c, dtype=float).reshape(-1)

        if len(inputs) != y.shape[0]:
            raise ValueError(
                "inputs length must match measurements length. "
                f"Got {len(inputs)} and {y.shape[0]}."
            )

        if y.size < 2:
            raise ValueError("Need at least 2 measurements for optimization.")

        if t_oa_hist.size == 0 or t_in_hist.size == 0:
            raise ValueError("History arrays must be non-empty.")

        if t_oa_hist.size != t_in_hist.size:
            raise ValueError(
                "history_t_oa_c and history_t_in_c must have the same length. "
                f"Got {t_oa_hist.size} and {t_in_hist.size}."
            )

        if x0 is None:
            x0 = (0.5, 0.5)

        t_oa_hist_mean_c = float(np.mean(t_oa_hist))
        t_in_hist_mean_c = float(np.mean(t_in_hist))
        t_in_0_c = float(y[0])

        result = minimize(
            fun=self._objective,
            x0=np.asarray(x0, dtype=float),
            args=(inputs, y, t_in_0_c, t_oa_hist_mean_c, t_in_hist_mean_c, dt_seconds),
            method=method,
            bounds=bounds,
        )

        alpha_iw = float(result.x[0])
        alpha_ow = float(result.x[1])

        initial_condition = self.build_initial_condition_from_history(
            t_in_0_c=t_in_0_c,
            t_oa_hist_mean_c=t_oa_hist_mean_c,
            t_in_hist_mean_c=t_in_hist_mean_c,
            alpha_iw=alpha_iw,
            alpha_ow=alpha_ow,
        )

        return InitialStateOptimizationResult(
            initial_condition=initial_condition,
            alpha_iw=alpha_iw,
            alpha_ow=alpha_ow,
            t_oa_hist_mean_c=t_oa_hist_mean_c,
            t_in_hist_mean_c=t_in_hist_mean_c,
            objective_value=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
            n_iterations=int(getattr(result, "nit", 0)),
        )

    def build_initial_condition_from_history(
        self,
        t_in_0_c: float,
        t_oa_hist_mean_c: float,
        t_in_hist_mean_c: float,
        alpha_iw: float,
        alpha_ow: float,
    ) -> FourR2CInitialCondition:
        """
        Build initial condition from history means and alpha weights.
        """
        t_iw_0_c = alpha_iw * t_oa_hist_mean_c + (1.0 - alpha_iw) * t_in_hist_mean_c
        t_ow_0_c = alpha_ow * t_oa_hist_mean_c + (1.0 - alpha_ow) * t_in_hist_mean_c

        return FourR2CInitialCondition(
            t_in_0_c=float(t_in_0_c),
            t_iw_0_c=float(t_iw_0_c),
            t_ow_0_c=float(t_ow_0_c),
        )

    def _objective(
        self,
        x: np.ndarray,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        t_in_0_c: float,
        t_oa_hist_mean_c: float,
        t_in_hist_mean_c: float,
        dt_seconds: float,
    ) -> float:
        """
        Objective function: open-loop indoor-temperature RMSE over the segment.
        """
        alpha_iw = float(x[0])
        alpha_ow = float(x[1])

        ic = self.build_initial_condition_from_history(
            t_in_0_c=t_in_0_c,
            t_oa_hist_mean_c=t_oa_hist_mean_c,
            t_in_hist_mean_c=t_in_hist_mean_c,
            alpha_iw=alpha_iw,
            alpha_ow=alpha_ow,
        )

        y_pred = self._rollout_indoor_temperature(
            inputs=inputs,
            initial_condition=ic,
            dt_seconds=dt_seconds,
        )

        return compute_rmse(
            y_true=measurements_t_in_c,
            y_pred=y_pred,
        )

    def _rollout_indoor_temperature(
        self,
        inputs: list[FourR2CInput],
        initial_condition: FourR2CInitialCondition,
        dt_seconds: float,
    ) -> np.ndarray:
        """
        Roll out the model open-loop and return predicted indoor air temperature.
        """
        x = initial_condition.as_state()

        y_pred = np.zeros(len(inputs), dtype=float)
        y_pred[0] = x.t_in_c

        for k in range(1, len(inputs)):
            x = self.model.step(
                state=x,
                inp=inputs[k - 1],
                dt_seconds=dt_seconds,
            )
            y_pred[k] = x.t_in_c

        return y_pred


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))