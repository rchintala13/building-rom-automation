from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Inputs
    idf_path: str
    idd_path: str
    knowledge_base_path: str

    # Extracted from IDF
    building_features: dict[str, Any]

    # Physics-based initial guess and logspace ranges per parameter
    # rc_ranges shape: {param_name: {"log_min": float, "log_max": float, "n_points": int}}
    initial_guess: dict[str, float]
    rc_ranges: dict[str, dict[str, Any]]

    # After LLM trimming
    trimmed_ranges: dict[str, dict[str, Any]]

    # Final grid fed to sysid
    parameter_grid: Any  # ParameterCandidateGrid

    # Nearest prior from knowledge base (may be empty)
    distilled_prior: dict[str, Any]

    # LLM conversation history
    messages: list[BaseMessage]
