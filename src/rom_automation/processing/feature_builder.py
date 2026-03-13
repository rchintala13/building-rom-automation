from __future__ import annotations

import pandas as pd


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "T_oa_C",
    "G_dir_W_m2",
    "G_dif_W_m2",
    "Q_int_J",
    #"Q_sol_win_J",
    "Q_cool_sens_J",
    "Q_heat_sens_J",
    "T_csp_C",
    "T_zone_C",
)


FINAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "T_oa_C",
    "T_zone_C",
    "T_csp_C",
    "G_ghi_kW_m2",
    "P_int_kW",
    #"P_sol_win_kW",
    "P_cool_kW",
    "P_heat_kW",
    "P_hvac_kW"
)


def build_features(
    df: pd.DataFrame,
    timestep_seconds: float,
) -> pd.DataFrame:
    """
    Build processed RC-model features from standardized EnergyPlus columns.

    Parameters
    ----------
    df
        Dataframe with standardized column names from column_mapper.
    timestep_seconds
        Simulation reporting timestep in seconds. Used to convert
        EnergyPlus energy outputs [J] into power inputs [kW].

    Returns
    -------
    pd.DataFrame
        Dataframe containing final processed feature columns.
    """
    _validate_required_columns(df)
    _validate_timestep_seconds(timestep_seconds)

    out = df.copy()

    # Solar irradiance
    out["G_ghi_kW_m2"] = (1./1000) * (out["G_dir_W_m2"] + out["G_dif_W_m2"])

    # Convert EnergyPlus energy outputs [J over timestep]
    # to average power over timestep [kW]
    j_to_kw = 1.0 / (timestep_seconds * 1000.0)

    out["P_int_kW"] = out["Q_int_J"] * j_to_kw
    #out["P_sol_win_kW"] = out["Q_sol_win_J"] * j_to_kw
    out["P_cool_kW"] = out["Q_cool_sens_J"] * j_to_kw
    out["P_heat_kW"] = out["Q_heat_sens_J"] * j_to_kw
    out["P_hvac_kW"] = out["P_heat_kW"] - out["P_cool_kW"]

    return out.loc[:, FINAL_OUTPUT_COLUMNS].copy()


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for feature building: "
            + ", ".join(missing)
        )


def _validate_timestep_seconds(timestep_seconds: float) -> None:
    if timestep_seconds <= 0:
        raise ValueError(
            f"timestep_seconds must be positive. Got {timestep_seconds}."
        )