from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from rom_automation.mpc.config_types import ControlConfig, ObjectiveConfig
from rom_automation.mpc.disturbance import HorizonInputs
from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.types import FourR2CParameters


@dataclass(frozen=True)
class MpcSolution:
    """
    Result of one receding-horizon MPC solve.

    - `p_hvac_applied_kw` : the first control move, applied to the plant.
    - `p_hvac_plan_kw`    : the full planned control trajectory (H,).
    - `t_in_plan_c`       : predicted indoor temperature trajectory (H,) for
                            steps k=1..H (aligned with the planned controls).
    - `success`           : whether the LP solver converged.
    - `status`            : solver status message.
    """

    p_hvac_applied_kw: float
    p_hvac_plan_kw: np.ndarray
    t_in_plan_c: np.ndarray
    success: bool
    status: str


class MpcController:
    """
    Linear-program MPC using the 4R2C model as the prediction model.

    Decision variable is the HVAC thermal power `P_hvac` (kW, signed) over the
    horizon. The objective trades off time-of-use energy cost against a soft
    comfort-band penalty:

        min  w_energy * Σ rate[j]*(p_pos[j] + p_neg[j])*dt_h  +  w_comfort * Σ slack[k]
        s.t. x[k+1] = Ad x[k] + Bd_dist d[k] + bu (p_pos[k] - p_neg[k])
             lower[k] - slack[k] <= T_in[k] <= upper[k] + slack[k]
             0 <= p_pos, p_neg <= p_hvac_max,   slack >= 0

    The dynamics are condensed so `T_in[1..H] = f + Phi u`, where `f` folds in the
    known initial state and disturbances and `Phi` is the (lower-triangular)
    control-to-temperature map. Solved with scipy HiGHS.
    """

    def __init__(
        self,
        params: FourR2CParameters,
        control: ControlConfig,
        objective: ObjectiveConfig,
    ) -> None:
        self.control = control
        self.objective = objective
        self.dt_seconds = control.dt_seconds
        self.dt_hours = self.dt_seconds / 3600.0

        model = FourR2CModel(params=params)
        disc = model.discretize(self.dt_seconds)
        self.a_d = disc.a_d                    # (3, 3)
        b_d = disc.b_d                         # (3, 4): [T_oa, G_ghi, P_int, P_hvac]
        self.c_row = model.measurement_matrix()  # (1, 3)
        self.b_dist = b_d[:, :3]               # (3, 3) disturbance columns
        self.b_u = b_d[:, 3]                   # (3,)   P_hvac column

        self.n_states = self.a_d.shape[0]

    def solve(self, x0: np.ndarray, horizon: HorizonInputs) -> MpcSolution:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != self.n_states:
            raise ValueError(
                f"x0 must have length {self.n_states}. Got {x0.size}."
            )

        h = len(horizon.timestamps)
        d = horizon.disturbance_matrix()       # (H, 3)

        f, phi = self._build_condensed(x0, d, h)

        # Decision vector z = [p_pos (H), p_neg (H), slack (H)].
        n = 3 * h
        rate = horizon.tou_rate
        we = self.objective.w_energy
        wc = self.objective.w_comfort

        c = np.concatenate(
            [
                we * rate * self.dt_hours,   # p_pos
                we * rate * self.dt_hours,   # p_neg
                wc * np.ones(h),             # slack
            ]
        )

        eye = np.eye(h)
        # Upper band: Phi p_pos - Phi p_neg - slack <= upper - f
        a_upper = np.hstack([phi, -phi, -eye])
        b_upper = horizon.comfort_upper_c - f
        # Lower band: -Phi p_pos + Phi p_neg - slack <= f - lower
        a_lower = np.hstack([-phi, phi, -eye])
        b_lower = f - horizon.comfort_lower_c

        a_ub = np.vstack([a_upper, a_lower])
        b_ub = np.concatenate([b_upper, b_lower])

        p_max = self.control.p_hvac_max_kw
        bounds = (
            [(0.0, p_max)] * h        # p_pos
            + [(0.0, p_max)] * h      # p_neg
            + [(0.0, None)] * h       # slack
        )

        res = linprog(
            c=c,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            # Soft slack makes the program feasible; a failure is numerical.
            return MpcSolution(
                p_hvac_applied_kw=0.0,
                p_hvac_plan_kw=np.zeros(h),
                t_in_plan_c=f.copy(),
                success=False,
                status=str(res.message),
            )

        z = res.x
        p_pos = z[:h]
        p_neg = z[h : 2 * h]
        u = p_pos - p_neg

        t_in_plan = f + phi @ u

        return MpcSolution(
            p_hvac_applied_kw=float(u[0]),
            p_hvac_plan_kw=u,
            t_in_plan_c=t_in_plan,
            success=True,
            status=str(res.message),
        )

    def _build_condensed(
        self,
        x0: np.ndarray,
        d: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the condensed prediction of indoor temperature over k=1..H:

            T_in[k] = f[k] + sum_{j<k} Phi[k, j] u[j]

        Returns (f, Phi) with f shape (H,) and Phi shape (H, H), lower-triangular.
        Row index m = k-1 corresponds to predicted step k = m+1.
        """
        # Precompute Ad^0 .. Ad^H.
        powers = [np.eye(self.n_states)]
        for _ in range(h):
            powers.append(self.a_d @ powers[-1])

        c = self.c_row.reshape(-1)             # (3,)
        f = np.zeros(h)
        phi = np.zeros((h, h))

        for m in range(h):                     # predicted step k = m + 1
            k = m + 1
            # Free/disturbance response: C Ad^k x0 + sum_{j=0}^{m} C Ad^{m-j} Bd_dist d[j]
            f_val = c @ (powers[k] @ x0)
            for j in range(k):                 # j = 0..m
                cadmj = c @ powers[m - j]       # C Ad^{m-j}  (1x3 dot -> scalar ops)
                f_val += cadmj @ (self.b_dist @ d[j])
                phi[m, j] = cadmj @ self.b_u
            f[m] = f_val

        return f, phi
