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
              -> {review_draft (opt-in) | validate_citations}
              -> validate_citations -> plan_action -> execute_action
              -> save_memory -> END
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from core.schemas import ALL_BRANCHES
from core.settings import get_settings

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
    review_draft,
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


def _route_after_writer(_state: AgentState) -> str:
    """Route the Writer's draft through the reviewer when the flag is on."""
    if get_settings().review_draft_enabled:
        return "review_draft"
    return "validate_citations"


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
    builder.add_node("review_draft", review_draft)
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
    builder.add_conditional_edges(
        "generate_response",
        _route_after_writer,
        ["review_draft", "validate_citations"],
    )
    builder.add_edge("review_draft", "validate_citations")
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


# --- P2 #8: retry-from-failed-node helpers -------------------------------

_RETRIEVER_NODES: frozenset[str] = frozenset(_BRANCH_TO_NODE.values())

# Markers shaped `"<retriever>: <ExcType>: <msg>"` come from `core.resilience.safe_retrieve`.
_SAFE_RETRIEVE_RE = re.compile(r"^(?P<node>[a-z_]+):\s+[A-Za-z][\w.]*:\s+")

# Markers shaped `"structured_failed:<node>"` come from `_append_marker` in nodes.py.
_STRUCTURED_FAILED_RE = re.compile(r"^structured_failed:(?P<node>[a-z_]+)$")


def parse_failure_marker(marker: str) -> str | None:
    """Return the node name a degraded marker came from, or `None` if not retryable.

    Only failures that re-running the node could plausibly fix are mapped:
    structured-output exhaustion (`structured_failed:<node>`), retriever
    exceptions (`<retrieve_*>: <ExcType>: <msg>` from `safe_retrieve`), the
    classifier (`classify_intent: ...`), and `reflection_failed` (retried by
    re-running `save_memory`). Informational markers (`chat_fallback:*`,
    `structured_retry:*`, `cost_extracted_via_fallback`, `citations_missing`,
    `evidence_insufficient`, `draft_*`) are not retryable.
    """
    if not isinstance(marker, str) or not marker:
        return None
    if marker == "reflection_failed":
        return "save_memory"
    m = _STRUCTURED_FAILED_RE.match(marker)
    if m:
        return m.group("node")
    m = _SAFE_RETRIEVE_RE.match(marker)
    if m:
        node = m.group("node")
        if node == "classify_intent" or node in _RETRIEVER_NODES:
            return node
    return None


def retryable_failures(degraded: list[str] | None) -> list[tuple[str, str]]:
    """Return `(marker, node)` pairs for retryable failures in `degraded`.

    Dedupes by target node (keeps the first marker), preserving order so the
    UI surfaces one retry button per failed node.
    """
    seen: dict[str, str] = {}
    for marker in degraded or []:
        node = parse_failure_marker(marker)
        if node and node not in seen:
            seen[node] = marker
    return [(marker, node) for node, marker in seen.items()]


def find_retry_checkpoint(graph, config: dict[str, Any], target_node: str) -> dict[str, Any] | None:
    """Locate the checkpoint config that re-runs `target_node` when resumed.

    Walks `graph.get_state_history(config)` (newest first) and returns the
    `config` of the most recent snapshot whose `next` tuple contains
    `target_node` — i.e. the state captured *before* that node ran. Resuming
    `graph.stream(None, returned_config)` then replays the graph from that
    node forward. Returns `None` if no such snapshot exists for the thread.
    """
    try:
        history = graph.get_state_history(config)
    except Exception:  # pragma: no cover - defensive: checkpointer not configured
        return None
    for snapshot in history:
        nxt = getattr(snapshot, "next", None) or ()
        if target_node in tuple(nxt):
            return snapshot.config
    return None
