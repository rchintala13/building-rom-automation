from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rom_automation.agent.state import AgentState

_MODEL = "claude-sonnet-4-6"
_SYSTEM_PROMPT = """\
You are an expert in building thermal physics and reduced-order RC models.
Your job is to narrow the search range for 4R2C system identification parameters
based on the physical properties of a specific building.

The 4R2C model has:
  - r_in_iw  : thermal resistance, indoor air → inner wall node  (K/W)
  - r_iw_ow  : thermal resistance, inner wall → outer wall node  (K/W)
  - r_ow_oa  : thermal resistance, outer wall → outdoor air      (K/W)
  - r_in_oa  : thermal resistance, indoor air → outdoor air      (K/W)
              (parallel path: infiltration + window conduction)
  - c_in     : indoor air thermal capacitance                    (J/K)
  - c_w      : total wall thermal capacitance                    (J/K)
  - alpha_ghi_outer_wall : effective solar collection area on exterior wall surface (m²)
  - alpha_ghi_inner_wall : effective solar collection area entering via windows     (m²)

All R and C parameters are in logspace. You will receive:
1. Building features extracted from the EnergyPlus IDF file.
2. Physics-based initial guesses and logspace ranges [log10_min, log10_max] for each parameter.

Return a JSON object with the SAME keys, each containing:
  {
    "log_min": <tighter lower bound in log10>,
    "log_max": <tighter upper bound in log10>,
    "n_points": <suggested number of grid points, integer 3-7>,
    "center": <best single initial guess, positive float>
  }

Rules:
- You may ONLY tighten ranges — never widen them beyond the input bounds.
- Use physical reasoning: e.g., a well-insulated wall constrains r_iw_ow to higher values;
  a large volume raises c_in; low WWR limits alpha_ghi_inner_wall.
- Assign more grid points (5-7) to parameters you are most uncertain about.
- Assign fewer (3) to parameters well-constrained by geometry.
- Output ONLY valid JSON with no extra commentary.\
"""


def llm_trim(state: AgentState) -> dict[str, Any]:
    features = state["building_features"]
    initial_guess = state["initial_guess"]
    rc_ranges = state["rc_ranges"]

    user_content = _build_user_message(features, initial_guess, rc_ranges)

    llm = ChatAnthropic(model=_MODEL, temperature=0)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response: AIMessage = llm.invoke(messages)
    trimmed_ranges = _parse_response(response.content, rc_ranges)

    return {
        "trimmed_ranges": trimmed_ranges,
        "messages": [*state.get("messages", []), HumanMessage(content=user_content), response],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(
    features: dict[str, Any],
    initial_guess: dict[str, float],
    rc_ranges: dict[str, dict],
) -> str:
    feat_summary = {
        "zone_floor_area_m2": round(features.get("zone_floor_area_m2") or 0, 1),
        "zone_volume_m3": round(features.get("zone_volume_m3") or 0, 1),
        "ext_wall_area_m2": round(features.get("ext_wall_area_m2") or 0, 1),
        "window_area_m2": round(features.get("window_area_m2") or 0, 1),
        "wwr": round(features.get("wwr") or 0, 3),
        "wall_r_value_m2k_per_w": round(features.get("wall_r_value_m2k_per_w") or 0, 3),
        "wall_thermal_mass_j_per_k_m2": round(
            features.get("wall_thermal_mass_j_per_k_m2") or 0, 1
        ),
        "window_u_w_m2k": round(features.get("window_u_w_m2k") or 0, 2),
        "window_shgc": round(features.get("window_shgc") or 0, 3),
        "infiltration_ach": round(features.get("infiltration_ach") or 0, 3),
        "temp_capacity_multiplier": features.get("temp_capacity_multiplier", 1.0),
        "solar_absorptance_exterior": round(
            features.get("solar_absorptance_exterior") or 0, 2
        ),
    }

    ranges_summary = {
        k: {
            "log_min": round(v["log_min"], 3),
            "log_max": round(v["log_max"], 3),
            "n_points": v["n_points"],
            "center": _fmt(v["center"]),
        }
        for k, v in rc_ranges.items()
    }

    return (
        "## Building Features\n"
        f"```json\n{json.dumps(feat_summary, indent=2)}\n```\n\n"
        "## Physics-based initial guesses\n"
        f"```json\n{json.dumps({k: _fmt(v) for k, v in initial_guess.items()}, indent=2)}\n```\n\n"
        "## Current logspace ranges\n"
        f"```json\n{json.dumps(ranges_summary, indent=2)}\n```\n\n"
        "Please return a tightened JSON object for the ranges."
    )


def _parse_response(
    content: str,
    fallback: dict[str, dict],
) -> dict[str, dict]:
    """Extract JSON from the LLM response; fall back to the input ranges on parse error."""
    raw = content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first { ... } block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return fallback
        else:
            return fallback

    # Validate and clamp: trimmed must not exceed the original bounds
    trimmed: dict[str, dict] = {}
    for param, original in fallback.items():
        if param not in parsed:
            trimmed[param] = original
            continue
        entry = parsed[param]
        try:
            log_min = max(float(entry["log_min"]), original["log_min"])
            log_max = min(float(entry["log_max"]), original["log_max"])
            if log_min >= log_max:
                log_min = original["log_min"]
                log_max = original["log_max"]
            trimmed[param] = {
                "log_min": log_min,
                "log_max": log_max,
                "n_points": int(entry.get("n_points", original["n_points"])),
                "center": float(entry.get("center", original["center"])),
            }
        except (KeyError, ValueError, TypeError):
            trimmed[param] = original

    return trimmed


def _fmt(v: float) -> str:
    """Scientific notation string for readability."""
    return f"{v:.3e}"
