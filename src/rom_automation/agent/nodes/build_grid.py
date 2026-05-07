from __future__ import annotations

from typing import Any

import numpy as np

from rom_automation.agent.state import AgentState
from rom_automation.sysid.parameter_grid import ParameterCandidateGrid


def build_grid(state: AgentState) -> dict[str, Any]:
    ranges = state["trimmed_ranges"]

    grid = ParameterCandidateGrid(
        r_in_iw=_logspace(ranges["r_in_iw"]),
        r_iw_ow=_logspace(ranges["r_iw_ow"]),
        r_ow_oa=_logspace(ranges["r_ow_oa"]),
        r_in_oa=_logspace(ranges["r_in_oa"]),
        c_in=_logspace(ranges["c_in"]),
        c_w=_logspace(ranges["c_w"]),
        alpha_ghi_outer_wall=_logspace(ranges["alpha_ghi_outer_wall"]),
        alpha_ghi_inner_wall=_logspace(ranges["alpha_ghi_inner_wall"]),
    )

    return {"parameter_grid": grid}


def _logspace(entry: dict[str, Any]) -> list[float]:
    log_min: float = entry["log_min"]
    log_max: float = entry["log_max"]
    n: int = int(entry.get("n_points", 5))

    # Always include the center point
    center: float | None = entry.get("center")
    points = np.logspace(log_min, log_max, num=n).tolist()

    if center is not None and center > 0:
        log_c = float(np.log10(center))
        if log_min < log_c < log_max:
            points.append(center)
            points = sorted(set(round(p, 12) for p in points))

    return points
