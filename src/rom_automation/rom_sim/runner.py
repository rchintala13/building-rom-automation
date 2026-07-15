from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.types import (
    FourR2CInput,
    FourR2CParameters,
    FourR2CState,
)


@dataclass(frozen=True)
class SimulationOutput:
    """
    Predicted state trajectory from an open-loop simulation. All arrays have
    shape (n_steps,) and are aligned with the input timestamps.
    """

    t_in_pred_c: np.ndarray
    t_iw_pred_c: np.ndarray
    t_ow_pred_c: np.ndarray


@dataclass(frozen=True)
class PredictionOutput:
    """
    Predicted indoor temperature at each timestep for one_step / n_step modes.

    `t_in_pred_c` holds the predicted value aligned with the target timestep
    (the step being predicted), with NaN where a prediction is unavailable
    (start-of-window offset).
    """

    t_in_pred_c: np.ndarray


class FourR2CRunner:
    """
    Thin wrapper around FourR2CModel that provides three prediction modes:
    open-loop simulation, one-step-ahead, and n-step-ahead.

    For one_step / n_step modes: at each anchor timestep k, the runner takes
    the measured T_in at k and the walls from the open-loop simulation at k,
    then rolls the model forward 1 or n steps and records the predicted T_in.
    """

    def __init__(
        self,
        params: FourR2CParameters,
        dt_seconds: float,
    ) -> None:
        if dt_seconds <= 0:
            raise ValueError(f"dt_seconds must be positive. Got {dt_seconds}.")

        self.model = FourR2CModel(params=params)
        self.dt_seconds = dt_seconds

    def initialize_state(
        self,
        t_in_0_c: float,
        history_t_oa_c: np.ndarray,
        history_t_in_c: np.ndarray,
        alpha_iw: float,
        alpha_ow: float,
    ) -> FourR2CState:
        """
        Build the initial full state using history means and the user-supplied
        alpha weights (not optimized here).
        """
        history_t_oa_c = np.asarray(history_t_oa_c, dtype=float).reshape(-1)
        history_t_in_c = np.asarray(history_t_in_c, dtype=float).reshape(-1)

        if history_t_oa_c.size == 0 or history_t_in_c.size == 0:
            raise ValueError("History arrays must be non-empty.")

        t_oa_mean = float(np.mean(history_t_oa_c))
        t_in_mean = float(np.mean(history_t_in_c))

        t_iw_0 = alpha_iw * t_oa_mean + (1.0 - alpha_iw) * t_in_mean
        t_ow_0 = alpha_ow * t_oa_mean + (1.0 - alpha_ow) * t_in_mean

        return FourR2CState(
            t_in_c=float(t_in_0_c),
            t_iw_c=float(t_iw_0),
            t_ow_c=float(t_ow_0),
        )

    def simulate(
        self,
        initial_state: FourR2CState,
        inputs: list[FourR2CInput],
    ) -> SimulationOutput:
        """
        Open-loop rollout for `len(inputs)` steps.

        Returns predicted state trajectories aligned so that index k contains
        the state at time k. Index 0 holds the initial state.
        """
        n = len(inputs)
        if n == 0:
            raise ValueError("inputs must be non-empty.")

        t_in = np.zeros(n, dtype=float)
        t_iw = np.zeros(n, dtype=float)
        t_ow = np.zeros(n, dtype=float)

        t_in[0] = initial_state.t_in_c
        t_iw[0] = initial_state.t_iw_c
        t_ow[0] = initial_state.t_ow_c

        s = initial_state
        for k in range(n - 1):
            s = self.model.step(
                state=s,
                inp=inputs[k],
                dt_seconds=self.dt_seconds,
            )
            t_in[k + 1] = s.t_in_c
            t_iw[k + 1] = s.t_iw_c
            t_ow[k + 1] = s.t_ow_c

        return SimulationOutput(
            t_in_pred_c=t_in,
            t_iw_pred_c=t_iw,
            t_ow_pred_c=t_ow,
        )

    def one_step_predict(
        self,
        initial_state: FourR2CState,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
    ) -> PredictionOutput:
        """
        One-step-ahead prediction. Uses the open-loop simulation for walls;
        resets T_in to the measurement at each anchor step k, then predicts
        T_in at k+1.

        Output array length == len(inputs). Index 0 is NaN (no prediction
        available for the very first step).
        """
        measurements_t_in_c = np.asarray(measurements_t_in_c, dtype=float).reshape(-1)
        _check_measurement_length(measurements_t_in_c, inputs)

        sim = self.simulate(initial_state=initial_state, inputs=inputs)

        n = len(inputs)
        pred = np.full(n, np.nan, dtype=float)

        for k in range(n - 1):
            reset_state = FourR2CState(
                t_in_c=float(measurements_t_in_c[k]),
                t_iw_c=float(sim.t_iw_pred_c[k]),
                t_ow_c=float(sim.t_ow_pred_c[k]),
            )
            next_state = self.model.step(
                state=reset_state,
                inp=inputs[k],
                dt_seconds=self.dt_seconds,
            )
            pred[k + 1] = next_state.t_in_c

        return PredictionOutput(t_in_pred_c=pred)

    def n_step_predict(
        self,
        initial_state: FourR2CState,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        n_steps_ahead: int,
    ) -> PredictionOutput:
        """
        N-step-ahead prediction. Uses the open-loop simulation for walls;
        at each anchor step k, resets T_in to the measurement and rolls the
        model forward n_steps_ahead steps. Records the predicted T_in at
        k + n_steps_ahead.

        Output array length == len(inputs). The first `n_steps_ahead` entries
        are NaN.
        """
        if n_steps_ahead <= 0:
            raise ValueError(
                f"n_steps_ahead must be positive. Got {n_steps_ahead}."
            )

        measurements_t_in_c = np.asarray(measurements_t_in_c, dtype=float).reshape(-1)
        _check_measurement_length(measurements_t_in_c, inputs)

        sim = self.simulate(initial_state=initial_state, inputs=inputs)

        n = len(inputs)
        pred = np.full(n, np.nan, dtype=float)

        for k in range(n - n_steps_ahead):
            state = FourR2CState(
                t_in_c=float(measurements_t_in_c[k]),
                t_iw_c=float(sim.t_iw_pred_c[k]),
                t_ow_c=float(sim.t_ow_pred_c[k]),
            )
            for j in range(n_steps_ahead):
                state = self.model.step(
                    state=state,
                    inp=inputs[k + j],
                    dt_seconds=self.dt_seconds,
                )
            pred[k + n_steps_ahead] = state.t_in_c

        return PredictionOutput(t_in_pred_c=pred)


def _check_measurement_length(
    measurements_t_in_c: np.ndarray,
    inputs: list[FourR2CInput],
) -> None:
    if measurements_t_in_c.shape[0] != len(inputs):
        raise ValueError(
            "measurements_t_in_c length must match inputs length. "
            f"Got {measurements_t_in_c.shape[0]} and {len(inputs)}."
        )
