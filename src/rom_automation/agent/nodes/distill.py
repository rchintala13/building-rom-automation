from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rom_automation.agent.knowledge_base import (
    append_record,
    load_knowledge_base,
    save_knowledge_base,
)
from rom_automation.agent.state import AgentState
from rom_automation.models.types import FourR2CParameters


def distill(
    state: AgentState,
    fitted_params: FourR2CParameters,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Call this node after EKF sysid completes to persist the result.

    Parameters
    ----------
    state : AgentState
        Current agent state (provides building features and KB path).
    fitted_params : FourR2CParameters
        Best-fit parameters returned by EKFSysIDTrainer.
    metrics : dict
        Sysid quality metrics, e.g. {"rmse_one_step_c": 0.3, "rmse_n_step_c": 1.2}.

    Returns
    -------
    dict with updated "distilled_prior" so future runs in the same session benefit immediately.
    """
    kb_path = state.get("knowledge_base_path", "data/agent_knowledge_base.json")
    features = state["building_features"]

    params_dict = asdict(fitted_params)

    records = load_knowledge_base(kb_path)
    records = append_record(
        records=records,
        features=features,
        fitted_params=params_dict,
        metrics=metrics,
    )
    save_knowledge_base(kb_path, records)

    return {
        "distilled_prior": {
            "features": {
                k: features.get(k)
                for k in [
                    "floor_area_m2",
                    "zone_volume_m3",
                    "wall_r_value_m2k_per_w",
                    "wwr",
                    "wall_thermal_mass_j_per_k_m2",
                ]
            },
            "fitted_params": params_dict,
            "metrics": metrics,
        }
    }
