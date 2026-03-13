from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    raw_variable_name: str
    short_name: str


COLUMN_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        raw_variable_name="Site Outdoor Air Drybulb Temperature",
        short_name="T_oa_C",
    ),
    ColumnSpec(
        raw_variable_name="Site Direct Solar Radiation Rate per Area",
        short_name="G_dir_W_m2",
    ),
    ColumnSpec(
        raw_variable_name="Site Diffuse Solar Radiation Rate per Area",
        short_name="G_dif_W_m2",
    ),
    ColumnSpec(
        raw_variable_name="Zone Total Internal Convective Heating Energy",
        short_name="Q_int_J",
    ),
    # ColumnSpec(
    #     raw_variable_name="Zone Windows Total Transmitted Solar Radiation Energy",
    #     short_name="Q_sol_win_J",
    # ),
    ColumnSpec(
        raw_variable_name="Zone Air System Sensible Cooling Energy",
        short_name="Q_cool_sens_J",
    ),
    ColumnSpec(
        raw_variable_name="Zone Air System Sensible Heating Energy",
        short_name="Q_heat_sens_J",
    ),
    ColumnSpec(
        raw_variable_name="Zone Thermostat Cooling Setpoint Temperature",
        short_name="T_csp_C",
    ),
    ColumnSpec(
        raw_variable_name="Zone Mean Air Temperature",
        short_name="T_zone_C",
    ),
)


def rename_energyplus_columns(
    df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Rename EnergyPlus output columns to compact standardized names.

    Parameters
    ----------
    df
        Raw EnergyPlus dataframe.
    strict
        If True, raise an error when any required variable is not found.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with renamed columns.
    """
    rename_map = build_rename_map(df.columns)

    if strict:
        missing = get_missing_required_variables(df.columns)
        if missing:
            raise ValueError(
                "Missing required EnergyPlus output variable(s): "
                + ", ".join(missing)
            )

    return df.rename(columns=rename_map).copy()


def build_rename_map(columns: pd.Index) -> dict[str, str]:
    """
    Build a rename map from raw EnergyPlus CSV headers to standardized names.
    """
    rename_map: dict[str, str] = {}

    for column in columns:
        matched_short_name = _match_short_name(column)
        if matched_short_name is not None:
            rename_map[str(column)] = matched_short_name

    return rename_map


def get_missing_required_variables(columns: pd.Index) -> list[str]:
    """
    Return the raw variable labels that were not found in the dataframe columns.
    """
    missing: list[str] = []

    for spec in COLUMN_SPECS:
        found = any(_contains_variable_label(col, spec.raw_variable_name) for col in columns)
        if not found:
            missing.append(spec.raw_variable_name)

    return missing


def _match_short_name(column_name: str) -> str | None:
    """
    Match a raw EnergyPlus column name to a standardized short name.

    Example raw EnergyPlus columns:
    - Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)
    - Conditioned Space:Zone Mean Air Temperature [C](TimeStep)
    """
    normalized_column = _normalize_column_name(column_name)

    for spec in COLUMN_SPECS:
        if _contains_variable_label(normalized_column, spec.raw_variable_name):
            return spec.short_name

    return None


def _contains_variable_label(column_name: str, raw_variable_name: str) -> bool:
    """
    Check whether an EnergyPlus CSV header contains the target variable label.
    """
    normalized_column = _normalize_column_name(column_name)
    normalized_variable = _normalize_column_name(raw_variable_name)

    return normalized_variable in normalized_column


def _normalize_column_name(name: str) -> str:
    """
    Normalize strings for more robust matching.
    """
    name = str(name).strip().lower()

    # Collapse repeated whitespace
    name = re.sub(r"\s+", " ", name)

    return name