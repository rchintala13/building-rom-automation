from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from rom_automation.models.types import (
    FourR2CInitialCondition,
    FourR2CInput,
    FourR2CMeasurement,
    FourR2CParameters,
    FourR2CState,
)


@dataclass(frozen=True)
class DiscreteLinearModel:
    """
    Discrete-time linear state-space model:

        x[k+1] = A_d x[k] + B_d u[k]
        y[k]   = C x[k]

    where y is indoor air temperature.
    """

    a_d: np.ndarray
    b_d: np.ndarray
    c: np.ndarray


class FourR2CModel:
    """
    4R2C reduced-order thermal model with 3 states.

    State order
    -----------
    x = [T_in_C, T_iw_C, T_ow_C]

    Input order used internally
    ---------------------------
    u = [T_oa_C, G_ghi_kW_m2, P_int_kW, P_hvac_kW]

    Notes
    -----
    - p_sol_win_kw from FourR2CInput is currently not used, because it does not
      appear in the equations you provided.
    - Positive p_hvac_kw means net heating added to the zone.
    """

    def __init__(
        self,
        params: FourR2CParameters,
        discretization: str = "zoh",
    ) -> None:
        """
        Parameters
        ----------
        params
            4R2C physical parameters.
        discretization
            Discretization scheme used by `discretize()` and `step()`.
            Options:
              - "zoh"           : exact zero-order hold via matrix exponential.
              - "forward_euler" : first-order Euler, `A_d = I + A*dt`,
                                  `B_d = B*dt`. Cheaper (no expm) but only
                                  stable if `dt < 2 / max|eig(A)|`.
        """
        self.params = params
        self.discretization = discretization
        self._validate_parameters()
        self._validate_discretization()

    def continuous_state_matrix(self) -> np.ndarray:
        """
        Return continuous-time A matrix for:

            x_dot = A x + B u
        """
        p = self.params

        c_in = p.c_in
        c_iw = p.c_iw
        c_ow = p.c_ow

        a = np.array(
            [
                [
                    -(1.0 / (p.r_in_iw * c_in) + 1.0 / (p.r_in_oa * c_in)),
                    1.0 / (p.r_in_iw * c_in),
                    0.0,
                ],
                [
                    1.0 / (p.r_in_iw * c_iw),
                    -(1.0 / (p.r_in_iw * c_iw) + 1.0 / (p.r_iw_ow * c_iw)),
                    1.0 / (p.r_iw_ow * c_iw),
                ],
                [
                    0.0,
                    1.0 / (p.r_iw_ow * c_ow),
                    -(1.0 / (p.r_iw_ow * c_ow) + 1.0 / (p.r_ow_oa * c_ow)),
                ],
            ],
            dtype=float,
        )
        return a

    def continuous_input_matrix(self) -> np.ndarray:
        """
        Return continuous-time B matrix for internal input vector:

            u = [T_oa_C, G_ghi_kW_m2, P_int_kW, P_hvac_kW]
        """
        p = self.params

        c_in = p.c_in
        c_iw = p.c_iw
        c_ow = p.c_ow

        b = np.array(
            [
                [
                    1.0 / (p.r_in_oa * c_in),                 # T_oa_C
                    0.0,                                      # G_ghi_kW_m2
                    1.0 / c_in,                               # P_int_kW
                    1.0 / c_in,                               # P_hvac_kW
                ],
                [
                    0.0,                                      # T_oa_C
                    p.alpha_ghi_inner_wall / c_iw,            # G_ghi_kW_m2
                    1.0 / c_iw,                               # P_int_kW
                    0.0,                                      # P_hvac_kW
                ],
                [
                    1.0 / (p.r_ow_oa * c_ow),                 # T_oa_C
                    p.alpha_ghi_outer_wall / c_ow,            # G_ghi_kW_m2
                    0.0,                                      # P_int_kW
                    0.0,                                      # P_hvac_kW
                ],
            ],
            dtype=float,
        )
        return b

    def measurement_matrix(self) -> np.ndarray:
        """
        Indoor air temperature is the measured output:

            y = [1 0 0] x
        """
        return np.array([[1.0, 0.0, 0.0]], dtype=float)

    def measurement_function(self, state: FourR2CState) -> np.ndarray:
        """
        Return measurement vector y for the given state.
        """
        x = state.as_vector()
        return self.measurement_matrix() @ x

    def continuous_rhs(
        self,
        state: FourR2CState,
        inp: FourR2CInput,
    ) -> np.ndarray:
        """
        Evaluate continuous-time dynamics:

            x_dot = A x + B u
        """
        a = self.continuous_state_matrix()
        b = self.continuous_input_matrix()

        x = state.as_vector()
        u = self._input_vector(inp)

        return a @ x + b @ u

    def discretize(
        self,
        dt_seconds: float,
        method: str | None = None,
    ) -> DiscreteLinearModel:
        """
        Discretize the continuous-time model.

        Parameters
        ----------
        dt_seconds
            Discretization step in seconds. Must be positive. Converted
            internally to hours because the continuous-time A and B matrices
            are dimensioned in 1/hour (R in degC/kW, C in kWh/degC).
        method
            Discretization scheme:
              - "zoh"           : exact zero-order hold via matrix exponential.
              - "forward_euler" : first-order Euler approximation.
            If None (default), uses the model's `discretization` attribute set
            at construction time.
        """
        if dt_seconds <= 0:
            raise ValueError(f"dt_seconds must be positive. Got {dt_seconds}.")

        dt_hours = dt_seconds / 3600.0

        chosen = method if method is not None else self.discretization
        if chosen == "zoh":
            return self._discretize_zoh(dt_hours)
        if chosen == "forward_euler":
            return self._discretize_forward_euler(dt_hours)
        raise ValueError(
            f"Unknown discretization method: {chosen!r}. "
            "Must be 'zoh' or 'forward_euler'."
        )

    def _discretize_zoh(self, dt_hours: float) -> DiscreteLinearModel:
        """
        Exact zero-order-hold discretization via `expm` on the augmented
        [[A, B], [0, 0]] block matrix. A and B are in 1/hour, so dt is in
        hours to keep the exponent dimensionless.
        """
        a = self.continuous_state_matrix()
        b = self.continuous_input_matrix()
        c = self.measurement_matrix()

        n_states = a.shape[0]
        n_inputs = b.shape[1]

        aug = np.zeros((n_states + n_inputs, n_states + n_inputs), dtype=float)
        aug[:n_states, :n_states] = a
        aug[:n_states, n_states:] = b

        exp_aug = expm(aug * dt_hours)

        a_d = exp_aug[:n_states, :n_states]
        b_d = exp_aug[:n_states, n_states:]

        return DiscreteLinearModel(a_d=a_d, b_d=b_d, c=c)

    def _discretize_forward_euler(
        self,
        dt_hours: float,
    ) -> DiscreteLinearModel:
        """
        First-order Euler discretization:

            A_d = I + A * dt
            B_d = B * dt

        A and B are in 1/hour, so dt is in hours to keep A_d entries
        dimensionless. Stability requires dt < 2 / max|eig(A)|; for typical
        residential thermal time constants (hours) and a 5-min step
        (dt = 1/12 h), this holds comfortably. Cheaper than ZOH (no `expm`)
        but less accurate for stiff systems or large dt.
        """
        a = self.continuous_state_matrix()
        b = self.continuous_input_matrix()
        c = self.measurement_matrix()

        n_states = a.shape[0]

        a_d = np.eye(n_states, dtype=float) + a * dt_hours
        b_d = b * dt_hours

        return DiscreteLinearModel(a_d=a_d, b_d=b_d, c=c)

    def step(
        self,
        state: FourR2CState,
        inp: FourR2CInput,
        dt_seconds: float,
    ) -> FourR2CState:
        """
        Propagate the model one discrete step forward using the model's
        configured discretization method.
        """
        disc = self.discretize(dt_seconds)

        x_k = state.as_vector()
        u_k = self._input_vector(inp)

        x_next = disc.a_d @ x_k + disc.b_d @ u_k
        return FourR2CState.from_vector(x_next)

    def predict_measurement(
        self,
        state: FourR2CState,
    ) -> FourR2CMeasurement:
        """
        Return the predicted measurement object from the current state.
        """
        y = self.measurement_function(state).reshape(-1)
        return FourR2CMeasurement.from_vector(y)

    def state_jacobian_continuous(self) -> np.ndarray:
        """
        For this linear model, the continuous-time EKF state Jacobian is A.
        """
        return self.continuous_state_matrix()

    def state_jacobian_discrete(self, dt_seconds: float) -> np.ndarray:
        """
        For this linear model, the discrete-time EKF state Jacobian is A_d.
        """
        return self.discretize(dt_seconds).a_d

    def measurement_jacobian(self) -> np.ndarray:
        """
        For this linear model, the measurement Jacobian is C.
        """
        return self.measurement_matrix()

    def initial_state_from_conditions(
        self,
        ic: FourR2CInitialCondition,
    ) -> FourR2CState:
        return ic.as_state()

    def _input_vector(self, inp: FourR2CInput) -> np.ndarray:
        """
        Internal model input vector order:

            [T_oa_C, G_ghi_kW_m2, P_int_kW, P_hvac_kW]
        """
        return np.array(
            [
                inp.t_oa_c,
                inp.g_ghi_kw_m2,
                inp.p_int_kw,
                inp.p_hvac_kw,
            ],
            dtype=float,
        )

    def _validate_parameters(self) -> None:
        p = self.params

        positive_params = {
            "r_in_iw": p.r_in_iw,
            "r_iw_ow": p.r_iw_ow,
            "r_ow_oa": p.r_ow_oa,
            "r_in_oa": p.r_in_oa,
            "c_in": p.c_in,
            "c_w": p.c_w,
        }

        for name, value in positive_params.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive. Got {value}.")

        if p.c_iw <= 0 or p.c_ow <= 0:
            raise ValueError(
                f"Derived wall capacitances must be positive. "
                f"Got c_iw={p.c_iw}, c_ow={p.c_ow}."
            )

    def _validate_discretization(self) -> None:
        allowed = {"zoh", "forward_euler"}
        if self.discretization not in allowed:
            raise ValueError(
                f"discretization must be one of {sorted(allowed)}. "
                f"Got {self.discretization!r}."
            )