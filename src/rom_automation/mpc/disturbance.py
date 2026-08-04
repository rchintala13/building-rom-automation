from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from rom_automation.mpc.config_types import ComfortConfig, TouConfig
from rom_automation.sysid.dataset_adapter import REQUIRED_COLUMNS


@dataclass(frozen=True)
class HorizonInputs:
    """
    Perfect-foresight exogenous signals over an MPC horizon of length H.

    Disturbance arrays are aligned with the model's internal input order
    (`[T_oa, G_ghi, P_int]`); `P_hvac` is the decision variable and is not here.
    All arrays have shape (H,).
    """

    timestamps: pd.DatetimeIndex
    t_oa_c: np.ndarray
    g_ghi_kw_m2: np.ndarray
    p_int_kw: np.ndarray
    tou_rate: np.ndarray
    comfort_lower_c: np.ndarray
    comfort_upper_c: np.ndarray

    def disturbance_matrix(self) -> np.ndarray:
        """
        Return the (H, 3) disturbance matrix in internal order [T_oa, G_ghi, P_int].
        """
        return np.column_stack([self.t_oa_c, self.g_ghi_kw_m2, self.p_int_kw])


class DisturbanceProvider:
    """
    Serves perfect-foresight disturbances (from processed_5min.csv), TOU rates,
    and comfort bounds over a receding MPC horizon.

    The processed CSV shares the weather/schedules of the live E+ run (same
    city/house/EPW), so slicing future disturbances from it is exact foresight.
    Horizon timestamps that fall outside the CSV coverage are clamped to the
    nearest available row (persistence tail near the data boundary).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dt_seconds: float,
        tou: TouConfig,
        comfort: ComfortConfig,
    ) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Disturbance CSV is missing required columns: {missing}"
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Disturbance CSV must have a DatetimeIndex.")

        self._df = df.sort_index()
        self._dt = pd.Timedelta(seconds=float(dt_seconds))
        self._tou = tou
        self._comfort = comfort

    def horizon(self, start_ts: datetime, horizon_steps: int) -> HorizonInputs:
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be positive. Got {horizon_steps}.")

        start = pd.Timestamp(start_ts)
        target_index = pd.DatetimeIndex(
            [start + i * self._dt for i in range(horizon_steps)]
        )

        # Clamp to CSV coverage, then nearest-align onto the 5-min grid.
        clamped = target_index.map(
            lambda t: min(max(t, self._df.index.min()), self._df.index.max())
        )
        rows = self._df.reindex(pd.DatetimeIndex(clamped), method="nearest")

        tou_rate = np.array(
            [self._tou.rate_at_hour(ts.hour) for ts in target_index],
            dtype=float,
        )
        comfort_lower = np.full(horizon_steps, self._comfort.lower_c, dtype=float)
        comfort_upper = np.full(horizon_steps, self._comfort.upper_c, dtype=float)

        return HorizonInputs(
            timestamps=target_index,
            t_oa_c=rows["T_oa_C"].to_numpy(dtype=float),
            g_ghi_kw_m2=rows["G_ghi_kW_m2"].to_numpy(dtype=float),
            p_int_kw=rows["P_int_kW"].to_numpy(dtype=float),
            tou_rate=tou_rate,
            comfort_lower_c=comfort_lower,
            comfort_upper_c=comfort_upper,
        )
