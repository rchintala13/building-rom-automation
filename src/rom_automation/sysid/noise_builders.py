from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rom_automation.models.types import FourR2CParameters
from rom_automation.sysid.ekf import AugmentedStateIndex


@dataclass(frozen=True)
class P0Spec:
    """
    Spec for per-candidate construction of the initial covariance P0.

    For each candidate, P0 = diag(std^2) where parameter stds scale with the
    candidate's initial guess; see `build_p0_from_ratios`.
    """

    temperature_std_c: dict[str, float]
    resistance_std_ratio: float
    capacitance_std_ratio: float
    alpha_std_ratio: float
    alpha_std_floor: float


@dataclass(frozen=True)
class ProcessNoiseSpec:
    """
    Spec for per-candidate construction of Q from unmodeled heat-gain stds.

    Q = V M V^T  (temperature block only; parameter block is zero);
    see `build_q_from_process_noise`.
    """

    q_in_std_kw: float
    q_iw_std_kw: float
    q_ow_std_kw: float


@dataclass(frozen=True)
class EKFNoiseConfig:
    """
    Bundle of noise/covariance settings consumed by the augmented-state EKF.

    R is the fixed scalar measurement-noise covariance. P0 and Q are built
    per-candidate by the trainer from `p0_spec` / `process_noise_spec`,
    since they depend on the candidate's initial guess and dt.
    """

    r: np.ndarray
    p0_spec: P0Spec
    process_noise_spec: ProcessNoiseSpec


def build_p0_from_ratios(
    z0: np.ndarray,
    temperature_std_c: dict[str, float],
    resistance_std_ratio: float,
    capacitance_std_ratio: float,
    alpha_std_ratio: float,
    alpha_std_floor: float,
    state_index: AugmentedStateIndex | None = None,
) -> np.ndarray:
    """
    Build initial EKF covariance diagonal P0 from standard deviation ratios.

    Parameters
    ----------
    z0
        Initial augmented state vector.
    temperature_std_c
        Fixed temperature standard deviations:
        {
            "t_in_c": 0.5,
            "t_iw_c": 2.0,
            "t_ow_c": 2.0,
        }
    resistance_std_ratio
        Relative std for resistance parameters, e.g. 0.30 means 30%.
    capacitance_std_ratio
        Relative std for capacitance parameters, e.g. 0.50 means 50%.
    alpha_std_ratio
        Relative std for alpha parameters.
    alpha_std_floor
        Minimum absolute std for alpha parameters.
    state_index
        Augmented state index mapping.

    Returns
    -------
    np.ndarray
        Diagonal covariance matrix P0.
    """
    idx = state_index or AugmentedStateIndex()
    z0 = np.asarray(z0, dtype=float).reshape(-1)

    if z0.size != idx.n_states:
        raise ValueError(f"Expected z0 length {idx.n_states}, got {z0.size}.")

    _validate_nonnegative("resistance_std_ratio", resistance_std_ratio)
    _validate_nonnegative("capacitance_std_ratio", capacitance_std_ratio)
    _validate_nonnegative("alpha_std_ratio", alpha_std_ratio)
    _validate_nonnegative("alpha_std_floor", alpha_std_floor)

    std = np.zeros(idx.n_states, dtype=float)

    std[idx.t_in_c] = float(temperature_std_c["t_in_c"])
    std[idx.t_iw_c] = float(temperature_std_c["t_iw_c"])
    std[idx.t_ow_c] = float(temperature_std_c["t_ow_c"])

    for j in [idx.r_in_iw, idx.r_iw_ow, idx.r_ow_oa, idx.r_in_oa]:
        std[j] = resistance_std_ratio * abs(z0[j])

    for j in [idx.c_in, idx.c_w]:
        std[j] = capacitance_std_ratio * abs(z0[j])

    for j in [idx.alpha_ghi_outer_wall, idx.alpha_ghi_inner_wall]:
        std[j] = max(alpha_std_floor, alpha_std_ratio * abs(z0[j]))

    if np.any(std <= 0):
        bad = np.where(std <= 0)[0].tolist()
        raise ValueError(f"All P0 standard deviations must be positive. Bad indices: {bad}")

    return np.diag(std**2)


def build_q_from_process_noise(
    params: FourR2CParameters,
    dt_seconds: float,
    q_in_std_kw: float,
    q_iw_std_kw: float,
    q_ow_std_kw: float,
    state_index: AugmentedStateIndex | None = None,
) -> np.ndarray:
    """
    Build augmented-state process noise covariance Q from heat-gain stds.

    Q = V M V^T  (temperature block only; parameter block is zero)

    where:
        M = diag(q_in_std_kw^2, q_iw_std_kw^2, q_ow_std_kw^2)
        V = dt_seconds * diag(1 / c_in, 1 / c_iw, 1 / c_ow)

    V is the Euler-step coupling from a steady unmodeled heat gain [kW]
    to a temperature change. Parameter states get zero process noise
    (they evolve as random walk only via the EKF update step; rely on
    parameter_bounds clipping to constrain drift).

    Parameters
    ----------
    params
        4R2C parameters for the current candidate (provides c_in, c_iw, c_ow).
    dt_seconds
        Discrete EKF timestep [s].
    q_in_std_kw, q_iw_std_kw, q_ow_std_kw
        Standard deviations of unmodeled heat gains into the indoor air,
        inner wall, and outer wall nodes [kW].
    state_index
        Augmented state index mapping.

    Returns
    -------
    np.ndarray
        Augmented Q matrix, shape (n_states, n_states).
    """
    idx = state_index or AugmentedStateIndex()

    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be positive. Got {dt_seconds}.")

    _validate_nonnegative("q_in_std_kw", q_in_std_kw)
    _validate_nonnegative("q_iw_std_kw", q_iw_std_kw)
    _validate_nonnegative("q_ow_std_kw", q_ow_std_kw)

    inv_c = np.array(
        [
            1.0 / params.c_in,
            1.0 / params.c_iw,
            1.0 / params.c_ow,
        ],
        dtype=float,
    )

    v_temp = dt_seconds * np.diag(inv_c)
    m = np.diag(
        np.array(
            [q_in_std_kw**2, q_iw_std_kw**2, q_ow_std_kw**2],
            dtype=float,
        )
    )

    q_temp_block = v_temp @ m @ v_temp.T

    q = np.zeros((idx.n_states, idx.n_states), dtype=float)
    q[idx.t_in_c, idx.t_in_c] = q_temp_block[0, 0]
    q[idx.t_iw_c, idx.t_iw_c] = q_temp_block[1, 1]
    q[idx.t_ow_c, idx.t_ow_c] = q_temp_block[2, 2]
    q[idx.t_in_c, idx.t_iw_c] = q_temp_block[0, 1]
    q[idx.t_in_c, idx.t_ow_c] = q_temp_block[0, 2]
    q[idx.t_iw_c, idx.t_in_c] = q_temp_block[1, 0]
    q[idx.t_iw_c, idx.t_ow_c] = q_temp_block[1, 2]
    q[idx.t_ow_c, idx.t_in_c] = q_temp_block[2, 0]
    q[idx.t_ow_c, idx.t_iw_c] = q_temp_block[2, 1]

    return q


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative. Got {value}.")
