from __future__ import annotations

import math
from typing import Any

from rom_automation.agent.state import AgentState

# Air properties
_RHO_AIR = 1.2        # kg/m³
_CP_AIR = 1005.0      # J/kgK

# Convection coefficients (W/m²K)
_H_INT = 3.0          # interior natural convection (wall ↔ air)
_H_EXT = 17.0         # exterior forced convection (wall ↔ outdoor air)

# Logspace grid half-spans in decades (initial_guess × 10^±span)
_R_SPAN = 2.0         # thermal resistance: broader because geometry estimation is rough
_C_SPAN = 1.5         # thermal capacitance: better constrained by material data
_ALPHA_SPAN = 1.5     # solar gain coefficient
_N_POINTS = 5         # points per dimension (replace with a trimmed value later)


def estimate_rc(state: AgentState) -> dict[str, Any]:
    feat = state["building_features"]
    prior = state.get("distilled_prior", {})
    prior_params: dict[str, float] = prior.get("fitted_params", {})

    guess = _physics_guess(feat)

    # If a close prior exists, blend toward its values (geometric mean)
    if prior_params:
        guess = _blend_with_prior(guess, prior_params, weight=0.4)

    rc_ranges = _build_logspace_ranges(guess)

    return {
        "initial_guess": guess,
        "rc_ranges": rc_ranges,
    }


# ---------------------------------------------------------------------------
# Physics-based initial guess
# ---------------------------------------------------------------------------

def _physics_guess(feat: dict[str, Any]) -> dict[str, float]:
    vol = feat.get("zone_volume_m3") or 250.0
    floor_area = feat.get("zone_floor_area_m2") or 100.0
    ext_wall_area = feat.get("ext_wall_area_m2") or 80.0
    int_surface_area = feat.get("int_surface_area_m2") or (2.5 * floor_area)
    window_area = feat.get("window_area_m2") or (0.15 * ext_wall_area)
    wall_r = feat.get("wall_r_value_m2k_per_w") or 2.0      # m²K/W
    wall_mass = feat.get("wall_thermal_mass_j_per_k_m2") or 20_000.0  # J/K/m²
    window_u = feat.get("window_u_w_m2k") or 3.0             # W/m²K
    window_shgc = feat.get("window_shgc") or 0.4
    ach = feat.get("infiltration_ach") or 0.35
    tcm = feat.get("temp_capacity_multiplier") or 1.0
    solar_abs = feat.get("solar_absorptance_exterior") or 0.7

    # --- R parameters (K/W) ---

    # R_in_iw: indoor air → inner wall node (interior convection)
    #   R = 1 / (h_int × A_int_surfaces)
    A_int = max(int_surface_area, floor_area * 2)
    r_in_iw = 1.0 / (_H_INT * A_int)

    # R_iw_ow: inner wall → outer wall node (wall conduction, excluding air films)
    #   R = (R_wall_m2k_per_w - 0.17) / A_ext_wall
    r_wall_pure = max(wall_r - 0.17, 0.1)   # strip air films added in load_idf
    r_iw_ow = r_wall_pure / ext_wall_area

    # R_ow_oa: outer wall → outdoor air (exterior convection film)
    #   R = 1 / (h_ext × A_ext_wall)
    r_ow_oa = 1.0 / (_H_EXT * ext_wall_area)

    # R_in_oa: indoor air → outdoor air (infiltration + window conduction in parallel)
    #   G_inf = ACH × V × rho × Cp / 3600
    #   G_win = U_window × A_window
    g_inf = ach * vol * _RHO_AIR * _CP_AIR / 3600.0
    g_win = window_u * window_area
    r_in_oa = 1.0 / max(g_inf + g_win, 1e-3)

    # --- C parameters (J/K) ---

    # C_in: indoor air thermal mass (including furniture / contents multiplier)
    c_in = _RHO_AIR * _CP_AIR * vol * tcm

    # C_w: total wall thermal mass (both sides of the 4R2C wall lumped together)
    c_w = wall_mass * ext_wall_area

    # --- Alpha parameters (effective solar collection area, m²) ---

    # alpha_ghi_outer_wall: solar absorbed directly by exterior wall surface
    #   ≈ absorptance × A_ext_wall × average_projection_factor
    alpha_outer = solar_abs * ext_wall_area * 0.3

    # alpha_ghi_inner_wall: solar entering through windows and hitting internal mass
    #   ≈ SHGC × A_window (solar transmitted to interior)
    alpha_inner = window_shgc * window_area

    return {
        "r_in_iw": _clamp_positive(r_in_iw),
        "r_iw_ow": _clamp_positive(r_iw_ow),
        "r_ow_oa": _clamp_positive(r_ow_oa),
        "r_in_oa": _clamp_positive(r_in_oa),
        "c_in": _clamp_positive(c_in),
        "c_w": _clamp_positive(c_w),
        "alpha_ghi_outer_wall": _clamp_positive(alpha_outer),
        "alpha_ghi_inner_wall": _clamp_positive(alpha_inner),
    }


# ---------------------------------------------------------------------------
# Logspace range builder
# ---------------------------------------------------------------------------

def _build_logspace_ranges(guess: dict[str, float]) -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for param, value in guess.items():
        if param.startswith("r_"):
            span = _R_SPAN
        elif param.startswith("c_"):
            span = _C_SPAN
        else:
            span = _ALPHA_SPAN

        log_center = math.log10(value)
        ranges[param] = {
            "log_min": log_center - span,
            "log_max": log_center + span,
            "n_points": _N_POINTS,
            "center": value,
        }
    return ranges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend_with_prior(
    guess: dict[str, float],
    prior: dict[str, float],
    weight: float,
) -> dict[str, float]:
    """Geometric-mean blend: guess^(1-w) × prior^w."""
    blended = {}
    for k, v in guess.items():
        if k in prior and prior[k] > 0:
            log_blended = (1 - weight) * math.log(v) + weight * math.log(prior[k])
            blended[k] = math.exp(log_blended)
        else:
            blended[k] = v
    return blended


def _clamp_positive(value: float, minimum: float = 1e-6) -> float:
    return max(value, minimum)
