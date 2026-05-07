from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rom_automation.agent.nodes.build_grid import build_grid
from rom_automation.agent.nodes.estimate_rc import estimate_rc
from rom_automation.agent.nodes.llm_trim import llm_trim
from rom_automation.agent.nodes.load_idf import load_idf
from rom_automation.agent.state import AgentState


def build_sysid_prior_graph() -> StateGraph:
    """
    Assemble the LangGraph that produces a ParameterCandidateGrid from an IDF file.

    Graph flow
    ----------
    load_idf → estimate_rc → llm_trim → build_grid → END

    The distill node is intentionally NOT wired into this graph — it is called
    externally after EKF sysid completes (it requires the fitted params and
    metrics that come from outside this graph).
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_idf", load_idf)
    graph.add_node("estimate_rc", estimate_rc)
    graph.add_node("llm_trim", llm_trim)
    graph.add_node("build_grid", build_grid)

    graph.add_edge(START, "load_idf")
    graph.add_edge("load_idf", "estimate_rc")
    graph.add_edge("estimate_rc", "llm_trim")
    graph.add_edge("llm_trim", "build_grid")
    graph.add_edge("build_grid", END)

    return graph.compile()


def run_sysid_prior_agent(
    idf_path: str,
    idd_path: str,
    knowledge_base_path: str = "data/agent_knowledge_base.json",
) -> AgentState:
    """
    Convenience wrapper: run the full graph and return the final state.

    The returned state contains:
    - building_features   : dict of IDF-extracted features
    - initial_guess       : dict of physics-based RC parameter estimates
    - rc_ranges           : dict of logspace ranges before LLM trimming
    - trimmed_ranges      : dict of logspace ranges after LLM trimming
    - parameter_grid      : ParameterCandidateGrid ready for EKFSysIDTrainer
    - distilled_prior     : nearest prior from the knowledge base (may be empty)
    """
    app = build_sysid_prior_graph()

    initial_state: AgentState = {
        "idf_path": idf_path,
        "idd_path": idd_path,
        "knowledge_base_path": knowledge_base_path,
        "building_features": {},
        "initial_guess": {},
        "rc_ranges": {},
        "trimmed_ranges": {},
        "parameter_grid": None,
        "distilled_prior": {},
        "messages": [],
    }

    return app.invoke(initial_state)
