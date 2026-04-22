from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from three_r2c.models.types import FourR2CInput


REQUIRED_COLUMNS: tuple[str, ...] = (
    "T_oa_C",
    "T_zone_C",
    "P_int_kW",
    "P_cool_kW",
    "P_heat_kW",
    "G_ghi_W_m2",
)


@dataclass(frozen=True)
class SysIDDataset:
    """
    Training or testing segment prepared for sysid.
    """

    inputs: list[FourR2CInput]
    measurements_t_in_c: np.ndarray
    history_t_oa_c: np.ndarray
    history_t_in_c: np.ndarray
    timestamps: pd.DatetimeIndex


def build_sysid_dataset(
    df: pd.DataFrame,
    history_hours: float,
) -> SysIDDataset:
    """
    Build a sysid dataset from a processed dataframe.

    Parameters
    ----------
    df
        Processed dataframe with a DatetimeIndex and required columns.
        The dataframe must include both:
        - history period before the identification segment
        - the identification segment itself

    history_hours
        Number of hours of prior data to use for initialization history.

    Returns
    -------
    SysIDDataset
        Contains:
        - input sequence for the training segment
        - measured indoor temperatures for the training segment
        - outdoor and indoor history arrays prior to the segment
    """
    _validate_dataframe(df)

    timestep_seconds = _detect_timestep_seconds(df.index)
    history_steps = _compute_history_steps(
        history_hours=history_hours,
        timestep_seconds=timestep_seconds,
    )

    if len(df) <= history_steps:
        raise ValueError(
            f"Dataframe length {len(df)} is not sufficient for "
            f"{history_hours} hours of history ({history_steps} steps) "
            "plus at least one identification timestep."
        )

    history_df = df.iloc[:history_steps].copy()
    segment_df = df.iloc[history_steps:].copy()

    inputs = build_input_sequence(segment_df)
    measurements_t_in_c = segment_df["T_zone_C"].to_numpy(dtype=float)

    history_t_oa_c = history_df["T_oa_C"].to_numpy(dtype=float)
    history_t_in_c = history_df["T_zone_C"].to_numpy(dtype=float)

    return SysIDDataset(
        inputs=inputs,
        measurements_t_in_c=measurements_t_in_c,
        history_t_oa_c=history_t_oa_c,
        history_t_in_c=history_t_in_c,
        timestamps=segment_df.index,
    )


def build_input_sequence(df: pd.DataFrame) -> list[FourR2CInput]:
    """
    Convert a processed dataframe segment into a list of FourR2CInput objects.
    """
    _validate_dataframe(df)

    p_hvac_kw = df["P_heat_kW"].to_numpy(dtype=float) - df["P_cool_kW"].to_numpy(dtype=float)

    inputs: list[FourR2CInput] = []

    for i in range(len(df)):
        inputs.append(
            FourR2CInput(
                t_oa_c=float(df["T_oa_C"].iloc[i]),
                g_ghi_w_m2=float(df["G_ghi_W_m2"].iloc[i]),
                p_int_kw=float(df["P_int_kW"].iloc[i]),
                p_sol_win_kw=0.0,
                p_hvac_kw=float(p_hvac_kw[i]),
            )
        )

    return inputs


def split_history_and_segment(
    df: pd.DataFrame,
    history_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into:
    - history period
    - identification segment

    Useful if you want direct access to both pieces.
    """
    _validate_dataframe(df)

    timestep_seconds = _detect_timestep_seconds(df.index)
    history_steps = _compute_history_steps(
        history_hours=history_hours,
        timestep_seconds=timestep_seconds,
    )

    if len(df) <= history_steps:
        raise ValueError(
            f"Dataframe length {len(df)} is not sufficient for "
            f"{history_hours} hours of history."
        )

    history_df = df.iloc[:history_steps].copy()
    segment_df = df.iloc[history_steps:].copy()

    return history_df, segment_df


def _compute_history_steps(
    history_hours: float,
    timestep_seconds: float,
) -> int:
    if history_hours <= 0:
        raise ValueError(f"history_hours must be positive. Got {history_hours}.")

    if timestep_seconds <= 0:
        raise ValueError(
            f"timestep_seconds must be positive. Got {timestep_seconds}."
        )

    history_seconds = history_hours * 3600.0
    steps = history_seconds / timestep_seconds

    rounded_steps = int(round(steps))

    if not np.isclose(steps, rounded_steps):
        raise ValueError(
            f"history_hours={history_hours} does not align with "
            f"timestep_seconds={timestep_seconds}. "
            f"Computed steps={steps}."
        )

    return rounded_steps


def _detect_timestep_seconds(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Dataframe index must be a DatetimeIndex.")

    deltas = index.to_series().diff().dropna()

    if deltas.empty:
        raise ValueError("Cannot detect timestep from fewer than 2 timestamps.")

    timestep = deltas.mode().iloc[0]
    return float(timestep.total_seconds())


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required processed dataframe columns: "
            + ", ".join(missing)
        )