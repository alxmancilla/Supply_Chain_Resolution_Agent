"""Minimal one-turn demo of the Supply Chain Resolution Agent.

Runs a single user question through the full graph and prints the
router's decision, per-branch hit counts and latencies, and the agent's
reply.

Usage:
    .venv/bin/python -m tools.demo
    .venv/bin/python -m tools.demo "Which carriers serve TX-AZ under 18000 lbs?"
"""
from __future__ import annotations

import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

from agent.graph import get_graph
from core.settings import AgentContext, get_settings


DEFAULT_QUESTION = (
    "I need to ship 15,000 lbs Austin to Dallas — what's my best option?"
)

_BRANCH_HIT_KEYS = {
    "ltm": "ltm_hits",
    "episodes": "episode_hits",
    "procedures": "procedure_hits",
    "rag": "rag_hits",
    "kg": "kg_hits",
}

_BRANCH_LATENCY_KEYS = {
    "ltm": "ltm_ms",
    "episodes": "episodes_ms",
    "procedures": "procedures_ms",
    "rag": "rag_ms",
    "kg": "kg_ms",
}


def _print_result(question: str, final_state: dict) -> None:
    routing = final_state.get("routing") or {}
    latency = final_state.get("latency_ms", {})
    branches = routing.get("branches", [])

    print(f"\nQ: {question}\n")
    print(
        f"routing: intent={routing.get('intent_label', '?')} "
        f"branches=[{', '.join(branches)}] "
        f"({latency.get('router_ms', 0):.1f} ms)"
    )
    print(f"rationale: {routing.get('rationale', '')!r}")

    degraded = final_state.get("degraded", [])
    if degraded:
        print(f"⚠️  degraded branches: {degraded}")

    print("\nretrieval:")
    for branch in branches:
        hits = final_state.get(_BRANCH_HIT_KEYS[branch], [])
        ms = latency.get(_BRANCH_LATENCY_KEYS[branch], 0)
        print(f"  {branch:<11} {len(hits):>2} hits   ({ms:.0f} ms)")
    for branch in _BRANCH_HIT_KEYS:
        if branch not in branches:
            print(f"  {branch:<11}  - skipped by router")

    print(
        f"\nllm: {latency.get('llm_ms', 0):.0f} ms "
        f"(ttft {latency.get('llm_ttft_ms', 0):.0f} ms)   "
        f"plan: {latency.get('plan_ms', 0):.0f} ms   "
        f"exec: {latency.get('execute_ms', 0):.0f} ms   "
        f"save: {latency.get('save_ms', 0):.0f} ms"
    )

    usage = final_state.get("usage") or {}
    if usage:
        print(
            f"usage: {int(usage.get('tokens_in', 0))} in + "
            f"{int(usage.get('tokens_out', 0))} out tokens "
            f"across {int(usage.get('calls', 0))} call(s)   "
            f"cost: ${usage.get('cost_usd', 0.0):.4f}"
        )

    reflection = final_state.get("reflection") or {}
    if reflection:
        print(
            f"reflection: clusters_found={reflection.get('clusters_found', 0)} "
            f"canonical_written={reflection.get('canonical_written', 0)} "
            f"tombstoned={reflection.get('tombstoned', 0)}"
        )

    plan = final_state.get("action_plan") or {}
    if plan.get("action_type") and plan.get("action_type") != "none":
        print(
            f"\naction_plan: {plan.get('action_type')} carrier={plan.get('carrier')} "
            f"lane={plan.get('lane')} est=${plan.get('estimated_cost_usd')} "
            f"requires_approval={plan.get('requires_approval')}"
        )
    draft = final_state.get("booking_draft") or {}
    if draft:
        print(
            f"booking_draft: {draft.get('draft_id')} status={draft.get('status')}"
        )

    reply = final_state["messages"][-1]
    body = reply.content if isinstance(reply.content, str) else str(reply.content)
    if not getattr(_print_result, "_streamed", False):
        print("\nA:")
        print(body)


def _stream_turn(graph, payload: dict | Command, config: dict) -> dict:
    """Stream a single graph invocation, printing token deltas live; return final state."""
    final_state: dict = {}
    printed_header = False
    for mode, event in graph.stream(payload, config=config, stream_mode=["values", "custom"]):
        if mode == "values":
            final_state = event
        elif mode == "custom" and isinstance(event, dict) and event.get("delta"):
            if not printed_header:
                print("\nA: ", end="", flush=True)
                printed_header = True
            sys.stdout.write(event["delta"])
            sys.stdout.flush()
    if printed_header:
        print()
        _print_result._streamed = True  # type: ignore[attr-defined]
    return final_state


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    settings = get_settings()
    graph = get_graph()
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content=question)],
            "context": AgentContext.from_settings(settings),
        },
        config,
    )
    while final_state.get("__interrupt__"):
        payload = final_state["__interrupt__"][0].value if final_state["__interrupt__"] else {}
        print(f"\n⏸  human approval requested: {payload} → auto-approving for demo")
        final_state = _stream_turn(graph, Command(resume={"approved": True}), config)

    _print_result(question, final_state)


if __name__ == "__main__":
    main()
