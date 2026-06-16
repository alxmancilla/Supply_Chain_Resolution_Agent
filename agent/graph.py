"""StateGraph definition for the Supply Chain Resolution Agent.

Topology — an intent router narrows a five-way retrieval fan-out per turn,
then a planning + execution stage proposes and (with human approval) commits
a booking draft before persisting memory:

    START -> classify_intent ─┬─> retrieve_ltm ────────┐
                              ├─> retrieve_episodes ───┤
                              ├─> retrieve_procedures ─┤
                              ├─> retrieve_rag ────────┼─> generate_response
                              └─> retrieve_kg ─────────┘
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
    retrieve_episodes,
    retrieve_kg,
    retrieve_ltm,
    retrieve_procedures,
    retrieve_rag,
    save_memory,
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
    routing = state.get("routing") or {}
    branches = routing.get("branches") or list(ALL_BRANCHES)
    selected = [_BRANCH_TO_NODE[b] for b in branches if b in _BRANCH_TO_NODE]
    return selected or [_BRANCH_TO_NODE[b] for b in ALL_BRANCHES]


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_ltm", retrieve_ltm)
    builder.add_node("retrieve_episodes", retrieve_episodes)
    builder.add_node("retrieve_procedures", retrieve_procedures)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("retrieve_kg", retrieve_kg)
    builder.add_node("generate_response", generate_response)
    builder.add_node("validate_citations", validate_citations)
    builder.add_node("plan_action", plan_action)
    builder.add_node("execute_action", execute_action)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        _route_to_retrievers,
        list(_BRANCH_TO_NODE.values()),
    )
    for retriever in _BRANCH_TO_NODE.values():
        builder.add_edge(retriever, "generate_response")
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
