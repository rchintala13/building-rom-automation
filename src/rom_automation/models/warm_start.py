from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rom_automation.models.four_r2c_model import FourR2CModel
from rom_automation.models.types import (
    FourR2CInitialCondition,
    FourR2CInput,
    FourR2CParameters,
    FourR2CState,
)


@dataclass(frozen=True)
class WarmStartResult:
    """
    Result of warm-starting the wall temperatures.

    - `initial_condition` : the full [T_in, T_iw, T_ow] state at the start of the
                            identification/control segment.
    - `seed_t_iw_c`, `seed_t_ow_c` : the steady-state wall seed used at the start
                            of the burn-in (kept for logging/inspection).
    """

    initial_condition: FourR2CInitialCondition
    seed_t_iw_c: float
    seed_t_ow_c: float


def conditional_wall_steady_state(
    model: FourR2CModel,
    inp: FourR2CInput,
    t_in_c: float,
) -> tuple[float, float]:
    """
    Steady-state wall temperatures [T_iw, T_ow] given a KNOWN indoor temperature
    and inputs, i.e. the equilibrium the walls would settle to if `T_in` and the
    inputs were held constant.

    From `dx/dt = A x + B u`, holding `T_in` fixed and setting the two wall rates
    to zero gives a 2x2 linear solve for the wall temperatures:

        A_ww [T_iw; T_ow] = -(A_w,in * T_in + (B u)_w)

    This uses the measured `T_in` rather than solving for it, so the seed is exact
    for steady conditions. Because the outer wall is slow, it sits near this
    equilibrium at all times, making it an excellent starting guess.
    """
    a = model.continuous_state_matrix()          # (3, 3)
    b = model.continuous_input_matrix()          # (3, 4): [T_oa, GHI, P_int, P_hvac]
    u = np.array(
        [inp.t_oa_c, inp.g_ghi_kw_m2, inp.p_int_kw, inp.p_hvac_kw],
        dtype=float,
    )
    bu = b @ u                                   # (3,)

    a_ww = a[1:3, 1:3]                            # wall-wall block
    a_w_in = a[1:3, 0]                            # wall coupling to T_in
    rhs = -(a_w_in * float(t_in_c) + bu[1:3])

    walls = np.linalg.solve(a_ww, rhs)
    return float(walls[0]), float(walls[1])


def warm_start_walls(
    params: FourR2CParameters,
    history_inputs: list[FourR2CInput],
    history_t_in_c: np.ndarray,
    t_in_0_c: float,
    dt_seconds: float,
) -> WarmStartResult:
    """
    Warm-start the wall temperatures for a segment that begins right after a
    measured history window.

    Procedure:
      1. Seed the walls at the steady-state value for the first history step's
         measured `T_in` and inputs (`conditional_wall_steady_state`).
      2. Anchored burn-in: propagate the model through the history, resetting
         `T_in` to the measurement each step and carrying the walls forward. The
         inner wall converges within a few hours; the outer wall is driven by the
         true indoor+outdoor trajectory. Pinning `T_in` removes the slow bulk mode,
         so the walls settle on the (short) anchored time constants.

    Returns the full state at segment start: `T_in = t_in_0_c` (the segment's
    first measurement) with the burn-in wall temperatures.
    """
    history_t_in_c = np.asarray(history_t_in_c, dtype=float).reshape(-1)
    n = len(history_inputs)
    if n == 0 or history_t_in_c.size == 0:
        raise ValueError("History must be non-empty to warm-start walls.")
    if history_t_in_c.size != n:
        raise ValueError(
            f"history_inputs ({n}) and history_t_in_c ({history_t_in_c.size}) "
            "must have the same length."
        )

    model = FourR2CModel(params=params)

    t_iw_seed, t_ow_seed = conditional_wall_steady_state(
        model=model, inp=history_inputs[0], t_in_c=float(history_t_in_c[0])
    )

    state = FourR2CState(
        t_in_c=float(history_t_in_c[0]),
        t_iw_c=t_iw_seed,
        t_ow_c=t_ow_seed,
    )
    for k in range(1, n):
        state = model.step(
            state=state,
            inp=history_inputs[k - 1],
            dt_seconds=dt_seconds,
        )
        # Re-anchor T_in to the measurement; keep the propagated walls.
        state = FourR2CState(
            t_in_c=float(history_t_in_c[k]),
            t_iw_c=state.t_iw_c,
            t_ow_c=state.t_ow_c,
        )

    return WarmStartResult(
        initial_condition=FourR2CInitialCondition(
            t_in_0_c=float(t_in_0_c),
            t_iw_0_c=state.t_iw_c,
            t_ow_0_c=state.t_ow_c,
        ),
        seed_t_iw_c=t_iw_seed,
        seed_t_ow_c=t_ow_seed,
    )
