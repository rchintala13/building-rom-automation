from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourR2CParameters:
    """
    Independent physical parameters for the 4R2C reduced-order model.

    State order
    -----------
    [T_in_C, T_iw_C, T_ow_C]

    Notes
    -----
    - Indoor air capacitance is modeled explicitly as c_in.
    - Total wall capacitance is c_w.
    - The wall capacitance is split equally:
          c_iw = c_w / 2
          c_ow = c_w / 2
    - alpha_ghi_outer_wall and alpha_ghi_inner_wall scale the GHI input
      applied to the outer-wall and inner-wall nodes, respectively.
    """

    r_in_iw: float
    r_iw_ow: float
    r_ow_oa: float
    r_in_oa: float

    c_in: float
    c_w: float

    alpha_ghi_outer_wall: float
    alpha_ghi_inner_wall: float

    @property
    def c_iw(self) -> float:
        return 0.5 * self.c_w

    @property
    def c_ow(self) -> float:
        return 0.5 * self.c_w


@dataclass(frozen=True)
class FourR2CState:
    """
    Dynamic state of the reduced-order model.

    State order
    -----------
    [T_in_C, T_iw_C, T_ow_C]
    """

    t_in_c: float
    t_iw_c: float
    t_ow_c: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.t_in_c, self.t_iw_c, self.t_ow_c],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "FourR2CState":
        x = np.asarray(x, dtype=float).reshape(-1)

        if x.size != 3:
            raise ValueError(f"Expected state vector of length 3, got {x.size}.")

        return cls(
            t_in_c=float(x[0]),
            t_iw_c=float(x[1]),
            t_ow_c=float(x[2]),
        )


@dataclass(frozen=True)
class FourR2CInput:
    """
    Input vector for the reduced-order model.

    Units
    -----
    t_oa_c       : degC
    g_ghi_kw_m2   : W/m2
    p_int_kw     : kW
    p_sol_win_kw : kW
    p_hvac_kw    : kW

    Convention
    ----------
    Positive p_hvac_kw means net heating added to the zone.
    """

    t_oa_c: float
    g_ghi_kw_m2: float
    p_int_kw: float
    p_sol_win_kw: float
    p_hvac_kw: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.t_oa_c,
                self.g_ghi_kw_m2,
                self.p_int_kw,
                self.p_sol_win_kw,
                self.p_hvac_kw,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, u: np.ndarray) -> "FourR2CInput":
        u = np.asarray(u, dtype=float).reshape(-1)

        if u.size != 5:
            raise ValueError(f"Expected input vector of length 5, got {u.size}.")

        return cls(
            t_oa_c=float(u[0]),
            g_ghi_kw_m2=float(u[1]),
            p_int_kw=float(u[2]),
            p_sol_win_kw=float(u[3]),
            p_hvac_kw=float(u[4]),
        )


@dataclass(frozen=True)
class FourR2CMeasurement:
    """
    Measurement vector used by estimation / evaluation.

    Current measurement:
    - indoor air temperature only
    """

    t_in_c: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.t_in_c], dtype=float)

    @classmethod
    def from_vector(cls, y: np.ndarray) -> "FourR2CMeasurement":
        y = np.asarray(y, dtype=float).reshape(-1)

        if y.size != 1:
            raise ValueError(f"Expected measurement vector of length 1, got {y.size}.")

        return cls(
            t_in_c=float(y[0]),
        )


@dataclass(frozen=True)
class FourR2CInitialCondition:
    """
    Initial condition for simulation / filtering.

    State order
    -----------
    [T_in_C, T_iw_C, T_ow_C]
    """

    t_in_0_c: float
    t_iw_0_c: float
    t_ow_0_c: float

    def as_state(self) -> FourR2CState:
        return FourR2CState(
            t_in_c=self.t_in_0_c,
            t_iw_c=self.t_iw_0_c,
            t_ow_c=self.t_ow_0_c,
        )

    def as_vector(self) -> np.ndarray:
        return self.as_state().as_vector()