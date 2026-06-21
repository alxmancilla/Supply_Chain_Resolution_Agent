"""StateGraph definition for the Supply Chain Resolution Agent.

Topology — an intent router narrows a five-way retrieval fan-out per turn,
a Think & Plan step refines the per-turn retrieval plan, retrievers run in
parallel, and a Reflection Agent decides whether the evidence is sufficient
or whether one rescue retrieval pass should run before answering. The
planning + execution stage then proposes and (with human approval) commits
a booking draft before persisting memory:

    START -> classify_intent -> think_and_plan ─┬─> retrieve_ltm ────────┐
                                                ├─> retrieve_episodes ───┤
                                                ├─> retrieve_procedures ─┤
                                                ├─> retrieve_rag ────────┼─> reflect_on_evidence
                                                └─> retrieve_kg ─────────┘
              -> {think_and_plan (loop, capped) | generate_response}
              -> validate_citations -> plan_action -> execute_action
              -> save_memory -> END
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from core.schemas import ALL_BRANCHES

from .memory import get_checkpointer, get_store
from .nodes import (
    AgentState,
    classify_intent,
    execute_action,
    generate_response,
    plan_action,
    reflect_on_evidence,
    retrieve_episodes,
    retrieve_kg,
    retrieve_ltm,
    retrieve_procedures,
    retrieve_rag,
    save_memory,
    think_and_plan,
    validate_citations,
)


_BRANCH_TO_NODE = {
    "ltm": "retrieve_ltm",
    "episodes": "retrieve_episodes",
    "procedures": "retrieve_procedures",
    "rag": "retrieve_rag",
    "kg": "retrieve_kg",
}


def _route_to_retrievers(state: AgentState) -> list[str]:
    plan = state.get("plan") or {}
    branches = plan.get("branches") or (state.get("routing") or {}).get("branches") or list(ALL_BRANCHES)
    selected = [_BRANCH_TO_NODE[b] for b in branches if b in _BRANCH_TO_NODE]
    return selected or [_BRANCH_TO_NODE[b] for b in ALL_BRANCHES]


def _route_after_reflection(state: AgentState) -> str:
    """Loop back to think_and_plan when reflection asked for a rescue pass."""
    reflection = state.get("reflection_eval") or {}
    if reflection.get("sufficient") is False:
        return "think_and_plan"
    return "generate_response"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("think_and_plan", think_and_plan)
    builder.add_node("retrieve_ltm", retrieve_ltm)
    builder.add_node("retrieve_episodes", retrieve_episodes)
    builder.add_node("retrieve_procedures", retrieve_procedures)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("retrieve_kg", retrieve_kg)
    builder.add_node("reflect_on_evidence", reflect_on_evidence)
    builder.add_node("generate_response", generate_response)
    builder.add_node("validate_citations", validate_citations)
    builder.add_node("plan_action", plan_action)
    builder.add_node("execute_action", execute_action)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "think_and_plan")
    builder.add_conditional_edges(
        "think_and_plan",
        _route_to_retrievers,
        list(_BRANCH_TO_NODE.values()),
    )
    for retriever in _BRANCH_TO_NODE.values():
        builder.add_edge(retriever, "reflect_on_evidence")
    builder.add_conditional_edges(
        "reflect_on_evidence",
        _route_after_reflection,
        ["think_and_plan", "generate_response"],
    )
    builder.add_edge("generate_response", "validate_citations")
    builder.add_edge("validate_citations", "plan_action")
    builder.add_edge("plan_action", "execute_action")
    builder.add_edge("execute_action", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(
        checkpointer=get_checkpointer(),
        store=get_store(),
    )


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
