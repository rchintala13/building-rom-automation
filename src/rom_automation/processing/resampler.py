from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def prepare_datetime_index(
    df: pd.DataFrame,
    year: int,
    datetime_column: str = "Date/Time",
) -> pd.DataFrame:
    """
    Convert the EnergyPlus 'Date/Time' column into a clean DatetimeIndex.

    Parameters
    ----------
    df
        Raw or partially processed dataframe containing an EnergyPlus Date/Time column.
    year
        Calendar year to assign to the timestamps.
    datetime_column
        Name of the EnergyPlus datetime column.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe indexed by DatetimeIndex.
    """
    if datetime_column not in df.columns:
        raise ValueError(
            f"Datetime column {datetime_column!r} not found in dataframe."
        )

    out = df.copy()
    out.index = out[datetime_column].apply(lambda x: _parse_energyplus_datetime(x, year))
    out = out.drop(columns=[datetime_column])
    out.index.name = "timestamp"

    return out.sort_index()


def detect_timestep_seconds(df: pd.DataFrame) -> float:
    """
    Detect the dominant timestep in seconds from the dataframe index.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex to detect timestep.")

    deltas = df.index.to_series().diff().dropna()

    if deltas.empty:
        raise ValueError("Cannot detect timestep from fewer than 2 timestamps.")

    timestep = deltas.mode().iloc[0]
    return timestep.total_seconds()


def validate_timestep_seconds(
    df: pd.DataFrame,
    expected_seconds: float,
    tolerance: float = 1e-6,
) -> None:
    """
    Raise an error if the dataframe timestep does not match the expected value.
    """
    detected_seconds = detect_timestep_seconds(df)

    if abs(detected_seconds - expected_seconds) > tolerance:
        raise ValueError(
            f"Expected timestep {expected_seconds} seconds, "
            f"but detected {detected_seconds} seconds."
        )


def resample_processed_data(
    df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    Resample processed data using mean aggregation.

    Assumes:
    - temperatures are state/rate-like
    - irradiance is rate-like
    - power terms are already converted from energy to kW

    Therefore mean aggregation is appropriate for all columns here.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex before resampling.")

    return df.resample(freq).mean().dropna(how="all")


def write_processed_csv(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Write processed dataframe to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True)


def _parse_energyplus_datetime(value: str, year: int) -> pd.Timestamp:
    """
    Parse EnergyPlus Date/Time strings like:
    - '01/01  00:05:00'
    - '01/01  24:00:00'

    EnergyPlus may use 24:00:00 to indicate the end of the day.
    This function converts 24:00:00 to 00:00:00 of the next day.
    """
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    match = re.fullmatch(r"(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{2}):(\d{2})", text)
    if not match:
        raise ValueError(
            f"Could not parse EnergyPlus datetime string: {value!r}"
        )

    month, day, hour, minute, second = map(int, match.groups())

    if hour == 24:
        base = pd.Timestamp(year=year, month=month, day=day)
        return base + pd.Timedelta(days=1)

    return pd.Timestamp(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )