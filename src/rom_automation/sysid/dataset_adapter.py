from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rom_automation.models.types import FourR2CInput


REQUIRED_COLUMNS: tuple[str, ...] = (
    "T_oa_C",
    "T_zone_C",
    "P_int_kW",
    "P_cool_kW",
    "P_heat_kW",
    "G_ghi_kW_m2",
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


@dataclass(frozen=True)
class DatasetSegment:
    """
    A single contiguous evaluation segment (train, val, or test).
    """

    inputs: list[FourR2CInput]
    measurements_t_in_c: np.ndarray
    timestamps: pd.DatetimeIndex


@dataclass(frozen=True)
class SysIDSplitDataset:
    """
    Dataset split into train / optional val / test segments, plus shared
    history used to initialize wall temperatures.
    """

    train: DatasetSegment
    val: DatasetSegment | None
    test: DatasetSegment
    history_t_oa_c: np.ndarray
    history_t_in_c: np.ndarray
    timestep_seconds: float


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


def build_sysid_split_dataset(
    df: pd.DataFrame,
    history_hours: float,
    train_fraction: float,
    test_fraction: float,
    val_fraction: float = 0.0,
) -> SysIDSplitDataset:
    """
    Build a train / val / test split dataset from a processed dataframe.

    Fractions are taken relative to the full dataframe length (including
    the history window). Splits are sequential in time:

        history window  ⊂  train portion (carved from train's head)
        then val portion, then test portion.

    Parameters
    ----------
    df
        Processed dataframe with a DatetimeIndex and required columns.
    history_hours
        Number of hours of prior data used to initialize wall temperatures.
        Must fit inside the train portion.
    train_fraction, val_fraction, test_fraction
        Fractions of the full dataframe assigned to each split. Must each be
        in [0, 1] and sum to <= 1.0. val_fraction may be 0 (no validation set).

    Returns
    -------
    SysIDSplitDataset
    """
    _validate_dataframe(df)
    _validate_split_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    timestep_seconds = _detect_timestep_seconds(df.index)
    history_steps = _compute_history_steps(
        history_hours=history_hours,
        timestep_seconds=timestep_seconds,
    )

    n_total = len(df)
    train_total_steps = int(n_total * train_fraction)
    val_steps = int(n_total * val_fraction)
    test_steps = int(n_total * test_fraction)

    if train_total_steps <= history_steps:
        raise ValueError(
            f"Train fraction yields {train_total_steps} steps, which is not "
            f"larger than the history window of {history_steps} steps. "
            "Increase train_fraction or reduce history_hours."
        )

    if test_steps <= 0:
        raise ValueError(
            f"Test fraction yields {test_steps} steps; test segment must be non-empty."
        )

    if val_fraction > 0 and val_steps <= 0:
        raise ValueError(
            f"Val fraction {val_fraction} yields {val_steps} steps; "
            "val segment must be non-empty when val_fraction > 0."
        )

    history_end = history_steps
    train_end = train_total_steps
    val_end = train_end + val_steps
    test_end = val_end + test_steps

    if test_end > n_total:
        raise ValueError(
            f"Computed split end index {test_end} exceeds dataframe length {n_total}."
        )

    history_df = df.iloc[:history_end]
    train_df = df.iloc[history_end:train_end]
    val_df = df.iloc[train_end:val_end] if val_steps > 0 else None
    test_df = df.iloc[val_end:test_end]

    history_t_oa_c = history_df["T_oa_C"].to_numpy(dtype=float)
    history_t_in_c = history_df["T_zone_C"].to_numpy(dtype=float)

    train_segment = _build_segment(train_df)
    val_segment = _build_segment(val_df) if val_df is not None else None
    test_segment = _build_segment(test_df)

    return SysIDSplitDataset(
        train=train_segment,
        val=val_segment,
        test=test_segment,
        history_t_oa_c=history_t_oa_c,
        history_t_in_c=history_t_in_c,
        timestep_seconds=timestep_seconds,
    )


def _build_segment(df: pd.DataFrame) -> DatasetSegment:
    if len(df) == 0:
        raise ValueError("Cannot build a DatasetSegment from an empty dataframe.")

    inputs = build_input_sequence(df)
    measurements_t_in_c = df["T_zone_C"].to_numpy(dtype=float)
    return DatasetSegment(
        inputs=inputs,
        measurements_t_in_c=measurements_t_in_c,
        timestamps=df.index,
    )


def _validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    for name, value in (
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ):
        if value < 0 or value > 1:
            raise ValueError(f"{name} must lie in [0, 1]. Got {value}.")

    total = train_fraction + val_fraction + test_fraction
    if total > 1.0 + 1e-9:
        raise ValueError(
            f"Split fractions must sum to <= 1.0. "
            f"Got train+val+test = {total}."
        )

    if train_fraction <= 0:
        raise ValueError(f"train_fraction must be positive. Got {train_fraction}.")

    if test_fraction <= 0:
        raise ValueError(f"test_fraction must be positive. Got {test_fraction}.")


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
                g_ghi_kw_m2=float(df["G_ghi_kW_m2"].iloc[i]),
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