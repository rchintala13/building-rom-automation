from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import brentq


def observability_matrix(a_d: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Observability matrix [C; C A; C A^2; ...; C A^{n-1}] for the pair (A_d, C).
    """
    a_d = np.asarray(a_d, dtype=float)
    c = np.asarray(c, dtype=float).reshape(1, -1)
    n = a_d.shape[0]
    return np.vstack([c @ np.linalg.matrix_power(a_d, i) for i in range(n)])


def is_observable(a_d: np.ndarray, c: np.ndarray) -> bool:
    """
    True iff (A_d, C) is observable (observability matrix has full rank n).
    """
    a_d = np.asarray(a_d, dtype=float)
    return np.linalg.matrix_rank(observability_matrix(a_d, c)) == a_d.shape[0]


def steady_state_kalman_gain(
    a_d: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    r: np.ndarray | float,
) -> np.ndarray:
    """
    Steady-state (stationary) Kalman filter gain for the discrete LTI system

        x[k+1] = A_d x[k] + w,   w ~ N(0, Q)
        y[k]   = C x[k]   + v,   v ~ N(0, R)

    Returned in the predict-then-update convention used by the EKF:

        x_post = x_pred + K (y - C x_pred)

    The stationary predicted covariance P_inf solves the filtering DARE

        P = A P A^T + Q - A P C^T (C P C^T + R)^{-1} C P A^T,

    which we obtain from scipy's control DARE by duality (A -> A^T, B -> C^T);
    then K = P C^T (C P C^T + R)^{-1}.

    The gain depends only on (A_d, C, Q, R) - not on any initial covariance P0,
    which affects only the (discarded) transient.

    Parameters
    ----------
    a_d : (n, n) discrete state-transition matrix.
    c   : (1, n) or (n,) measurement matrix (single output).
    q   : (n, n) process-noise covariance.
    r   : scalar or (1, 1) measurement-noise covariance.

    Returns
    -------
    (n, 1) steady-state Kalman gain.
    """
    a_d = np.asarray(a_d, dtype=float)
    c = np.asarray(c, dtype=float).reshape(1, -1)
    q = np.asarray(q, dtype=float)
    r = np.atleast_2d(np.asarray(r, dtype=float)).reshape(1, 1)

    n = a_d.shape[0]
    if a_d.shape != (n, n):
        raise ValueError(f"a_d must be square; got {a_d.shape}.")
    if c.shape != (1, n):
        raise ValueError(f"c must be (1, {n}); got {c.shape}.")
    if q.shape != (n, n):
        raise ValueError(f"q must be {(n, n)}; got {q.shape}.")

    if not is_observable(a_d, c):
        raise ValueError(
            "(a_d, c) is not observable; the steady-state Kalman gain is not "
            "well-defined. Check the model parameters and measurement matrix."
        )

    p_inf = _steady_state_prior_covariance(a_d, c, q, r)
    return _gain_from_prior(p_inf, c, r)


def _normalize(a_d, c, q, r):
    a_d = np.asarray(a_d, dtype=float)
    n = a_d.shape[0]
    c = np.asarray(c, dtype=float).reshape(1, n)
    q = np.asarray(q, dtype=float)
    r = np.atleast_2d(np.asarray(r, dtype=float)).reshape(1, 1)
    return a_d, c, q, r


def _steady_state_prior_covariance(a_d, c, q, r) -> np.ndarray:
    """Stationary predicted (prior) covariance P_inf from the filtering DARE."""
    return solve_discrete_are(a_d.T, c.T, q, r)


def _gain_from_prior(p_prior: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    s = c @ p_prior @ c.T + r
    return p_prior @ c.T @ np.linalg.inv(s)


def steady_state_innovation_variance(
    a_d: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    r: np.ndarray | float,
) -> float:
    """
    The filter's assumed stationary innovation variance S = C P_inf C^T + R,
    i.e. how large the one-step prediction residuals *should* be under (Q, R).
    """
    a_d, c, q, r = _normalize(a_d, c, q, r)
    p_inf = _steady_state_prior_covariance(a_d, c, q, r)
    return float(c @ p_inf @ c.T + r)


def innovation_consistent_gain(
    a_d: np.ndarray,
    c: np.ndarray,
    q_temp: np.ndarray,
    r: np.ndarray | float,
    target_innovation_var: float,
    inflation_bounds: tuple[float, float] = (1e-6, 1e12),
) -> tuple[np.ndarray, float]:
    """
    Scale the process noise (Q = f * q_temp) so the filter's assumed innovation
    variance S = C P_inf C^T + R matches an empirically observed innovation
    variance `target_innovation_var` (e.g. the sysid one-step RMSE squared).

    S is monotonically increasing in f, so a unique scaling exists whenever the
    target lies in the reachable range; otherwise f is clamped to the bounds.
    This makes the filter trust the measurement exactly as much as the model's
    real prediction error warrants -- more error => bigger gain -- and auto-relaxes
    as the model improves (smaller observed innovations => smaller gain).

    Returns (gain, inflation_factor).
    """
    a_d, c, q_temp, r = _normalize(a_d, c, q_temp, r)

    if not is_observable(a_d, c):
        raise ValueError("(a_d, c) is not observable; gain is undefined.")
    if target_innovation_var <= 0:
        raise ValueError(
            f"target_innovation_var must be positive. Got {target_innovation_var}."
        )

    lo, hi = np.log10(inflation_bounds[0]), np.log10(inflation_bounds[1])

    def s_gap(log_f: float) -> float:
        p = _steady_state_prior_covariance(a_d, c, (10.0**log_f) * q_temp, r)
        return float(c @ p @ c.T + r) - target_innovation_var

    gap_lo, gap_hi = s_gap(lo), s_gap(hi)
    if gap_lo >= 0:
        # Even minimal process noise implies larger innovations than observed:
        # the model is better than the target -> use the smallest inflation.
        log_f = lo
    elif gap_hi <= 0:
        log_f = hi
    else:
        log_f = brentq(s_gap, lo, hi, xtol=1e-6)

    f = float(10.0**log_f)
    p_inf = _steady_state_prior_covariance(a_d, c, f * q_temp, r)
    return _gain_from_prior(p_inf, c, r), f
