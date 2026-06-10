from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.types import FourR2CInput, FourR2CParameters, FourR2CState


@dataclass(frozen=True)
class AugmentedStateIndex:
    t_in_c: int = 0
    t_iw_c: int = 1
    t_ow_c: int = 2
    r_in_iw: int = 3
    r_iw_ow: int = 4
    r_ow_oa: int = 5
    r_in_oa: int = 6
    c_in: int = 7
    c_w: int = 8
    alpha_ghi_outer_wall: int = 9
    alpha_ghi_inner_wall: int = 10

    @property
    def n_states(self) -> int:
        return 11


@dataclass
class EKFResult:
    z_filtered: np.ndarray
    p_filtered: np.ndarray
    y_pred_one_step: np.ndarray


class AugmentedStateEKF:
    """
    Extended Kalman Filter for joint state-parameter estimation.

    Augmented state order
    ---------------------
    z = [
        T_in_C,
        T_iw_C,
        T_ow_C,
        r_in_iw,
        r_iw_ow,
        r_ow_oa,
        r_in_oa,
        c_in,
        c_w,
        alpha_ghi_outer_wall,
        alpha_ghi_inner_wall,
    ]
    """

    def __init__(
        self,
        q: np.ndarray,
        r: np.ndarray,
        state_index: AugmentedStateIndex | None = None,
    ) -> None:
        self.idx = state_index or AugmentedStateIndex()
        self.q = np.asarray(q, dtype=float)
        self.r = np.asarray(r, dtype=float)

        n = self.idx.n_states

        if self.q.shape != (n, n):
            raise ValueError(f"Q must have shape {(n, n)}, got {self.q.shape}")

        if self.r.shape != (1, 1):
            raise ValueError(f"R must have shape (1, 1), got {self.r.shape}")

    def run(
        self,
        inputs: list[FourR2CInput],
        measurements_t_in_c: np.ndarray,
        z0: np.ndarray,
        p0: np.ndarray,
        dt_seconds: float,
        z_lower: np.ndarray | None = None,
        z_upper: np.ndarray | None = None,
    ) -> EKFResult:
        """
        Run EKF over a sequence.

        Parameters
        ----------
        inputs
            List of FourR2CInput objects, one per timestep.
        measurements_t_in_c
            Array of measured indoor air temperatures, shape (N,).
        z0
            Initial augmented state, shape (11,).
        p0
            Initial covariance, shape (11, 11).
        dt_seconds
            Discrete timestep in seconds.
        z_lower, z_upper
            Optional per-state lower/upper bounds, shape (11,). When provided,
            the augmented state is clipped element-wise after each update step.
            Use -inf / +inf for entries that should not be bounded
            (e.g. temperatures).

        Returns
        -------
        EKFResult
        """
        measurements_t_in_c = np.asarray(measurements_t_in_c, dtype=float).reshape(-1)
        z = np.asarray(z0, dtype=float).reshape(-1)
        p = np.asarray(p0, dtype=float)

        n_steps = len(inputs)
        n_states = self.idx.n_states

        if measurements_t_in_c.shape[0] != n_steps:
            raise ValueError(
                "Number of measurements must equal number of inputs. "
                f"Got {measurements_t_in_c.shape[0]} measurements and {n_steps} inputs."
            )

        if z.shape[0] != n_states:
            raise ValueError(
                f"Initial augmented state must have length {n_states}, got {z.shape[0]}"
            )

        if p.shape != (n_states, n_states):
            raise ValueError(
                f"Initial covariance must have shape {(n_states, n_states)}, got {p.shape}"
            )

        bounds_active = z_lower is not None or z_upper is not None
        if bounds_active:
            if z_lower is None or z_upper is None:
                raise ValueError("z_lower and z_upper must both be provided together.")

            z_lower = np.asarray(z_lower, dtype=float).reshape(-1)
            z_upper = np.asarray(z_upper, dtype=float).reshape(-1)

            if z_lower.shape != (n_states,) or z_upper.shape != (n_states,):
                raise ValueError(
                    f"z_lower and z_upper must have shape ({n_states},), "
                    f"got {z_lower.shape} and {z_upper.shape}."
                )

            if np.any(z_lower > z_upper):
                raise ValueError("z_lower must be <= z_upper element-wise.")

        z_filtered = np.zeros((n_steps, n_states), dtype=float)
        p_filtered = np.zeros((n_steps, n_states, n_states), dtype=float)
        y_pred_one_step = np.zeros(n_steps, dtype=float)

        for k in range(n_steps):
            u_k = inputs[k]
            y_k = np.array([measurements_t_in_c[k]], dtype=float)

            # Predict
            z_pred = self._f(z, u_k, dt_seconds)
            f_jac = self._finite_difference_f_jacobian(z, u_k, dt_seconds)
            p_pred = f_jac @ p @ f_jac.T + self.q

            # Predicted measurement
            y_pred = self._h(z_pred)
            h_jac = self._h_jacobian()

            # Update
            innovation = y_k - y_pred
            s = h_jac @ p_pred @ h_jac.T + self.r
            k_gain = p_pred @ h_jac.T @ np.linalg.inv(s)

            z = z_pred + (k_gain @ innovation).reshape(-1)
            p = (np.eye(n_states) - k_gain @ h_jac) @ p_pred

            if bounds_active:
                z = np.clip(z, z_lower, z_upper)

            z_filtered[k, :] = z
            p_filtered[k, :, :] = p
            y_pred_one_step[k] = float(y_pred[0])

        return EKFResult(
            z_filtered=z_filtered,
            p_filtered=p_filtered,
            y_pred_one_step=y_pred_one_step,
        )

    def _f(
        self,
        z: np.ndarray,
        u: FourR2CInput,
        dt_seconds: float,
    ) -> np.ndarray:
        """
        Augmented transition function:

            temperatures evolve by 4R2C dynamics
            parameters evolve as random walk with mean identity
        """
        state = self._extract_state(z)
        params = self._extract_parameters(z)

        model = FourR2CModel(params=params)
        next_state = model.step(
            state=state,
            inp=u,
            dt_seconds=dt_seconds,
        )

        z_next = z.copy()
        z_next[self.idx.t_in_c] = next_state.t_in_c
        z_next[self.idx.t_iw_c] = next_state.t_iw_c
        z_next[self.idx.t_ow_c] = next_state.t_ow_c

        # Parameter states remain unchanged in the prediction mean
        # (random walk is represented through Q)
        return z_next

    def _h(self, z: np.ndarray) -> np.ndarray:
        """
        Measurement function: indoor air temperature only.
        """
        return np.array([z[self.idx.t_in_c]], dtype=float)

    def _h_jacobian(self) -> np.ndarray:
        """
        Measurement Jacobian for y = T_in.
        """
        h = np.zeros((1, self.idx.n_states), dtype=float)
        h[0, self.idx.t_in_c] = 1.0
        return h

    def _finite_difference_f_jacobian(
        self,
        z: np.ndarray,
        u: FourR2CInput,
        dt_seconds: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Finite-difference Jacobian of the augmented transition function.
        """
        n = self.idx.n_states
        f0 = self._f(z, u, dt_seconds)
        jac = np.zeros((n, n), dtype=float)

        for j in range(n):
            z_perturbed = z.copy()
            step = eps * max(1.0, abs(z[j]))
            z_perturbed[j] += step

            f1 = self._f(z_perturbed, u, dt_seconds)
            jac[:, j] = (f1 - f0) / step

        return jac

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