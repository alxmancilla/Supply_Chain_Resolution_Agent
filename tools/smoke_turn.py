"""End-to-end smoke test: run one agent turn, then verify cross-session LTM recall.

Usage:
    .venv/bin/python -m tools.smoke_turn
"""
from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from langgraph.types import Command

from agent.graph import get_graph
from agent.memory import get_store, memory_namespace
from core.memory import get_semantic_memory
from core.settings import AgentContext, get_settings

_SETTINGS = get_settings()
REALM = _SETTINGS.realm_id
USER = _SETTINGS.user_id


def _print_turn_summary(label: str, final_state: dict) -> None:
    print(f"\n===== {label} =====")
    routing = final_state.get("routing") or {}
    if routing:
        branches = ", ".join(routing.get("branches", []))
        print(
            f"  routing: intent={routing.get('intent_label', '?')} "
            f"branches=[{branches}] rationale={routing.get('rationale', '')!r}"
        )
    degraded = final_state.get("degraded", [])
    if degraded:
        print(f"  ⚠️ degraded: {degraded}")
    latency = final_state.get("latency_ms", {})
    for k in ("router_ms", "ltm_ms", "episodes_ms", "procedures_ms", "rag_ms", "kg_ms", "llm_ttft_ms", "llm_ms", "save_ms"):
        if k in latency:
            print(f"  {k}: {latency[k]:.1f} ms")
    usage = final_state.get("usage") or {}
    if usage:
        print(
            f"  usage: {int(usage.get('tokens_in', 0))} in + {int(usage.get('tokens_out', 0))} out tokens "
            f"across {int(usage.get('calls', 0))} call(s) (cost ${usage.get('cost_usd', 0.0):.4f})"
        )
    reflection = final_state.get("reflection") or {}
    if reflection:
        print(
            f"  reflection: clusters_found={reflection.get('clusters_found', 0)} "
            f"canonical_written={reflection.get('canonical_written', 0)} "
            f"tombstoned={reflection.get('tombstoned', 0)}"
        )
    print(f"  ltm_hits (semantic): {len(final_state.get('ltm_hits', []))}")
    for h in final_state.get("ltm_hits", []):
        print(f"    - [{h.get('score'):.3f}] {h.get('content')[:120]}")
    print(f"  episode_hits (episodic): {len(final_state.get('episode_hits', []))}")
    for h in final_state.get("episode_hits", []):
        print(
            f"    - [{h.get('score'):.3f}] lane={h.get('lane')} {h.get('summary', '')[:100]} "
            f"(rec: {h.get('recommendation', '-')})"
        )
    print(f"  procedure_hits (procedural): {len(final_state.get('procedure_hits', []))}")
    for h in final_state.get("procedure_hits", []):
        print(f"    - {h.get('rule_id')} ({h.get('category')}): {h.get('rule', '')[:100]}")
    print(f"  rag_hits: {len(final_state.get('rag_hits', []))}")
    for h in final_state.get("rag_hits", []):
        print(f"    - [{h.get('score'):.3f}] {h.get('doc_type')} :: {h.get('source')}")
    print(f"  kg_hits: {len(final_state.get('kg_hits', []))}")
    for h in final_state.get("kg_hits", []):
        print(f"    - {h.get('fact', '')[:160]}")
    plan = final_state.get("action_plan") or {}
    if plan.get("action_type") and plan.get("action_type") != "none":
        print(
            f"  action_plan: type={plan.get('action_type')} carrier={plan.get('carrier')} "
            f"lane={plan.get('lane')} weight_lb={plan.get('weight_lb')} "
            f"est=${plan.get('estimated_cost_usd')} requires_approval={plan.get('requires_approval')}"
        )
    draft = final_state.get("booking_draft") or {}
    if draft:
        print(
            f"  booking_draft: id={draft.get('draft_id')} status={draft.get('status')} "
            f"carrier={draft.get('carrier')} cost=${draft.get('estimated_cost_usd')}"
        )
    reply = final_state["messages"][-1]
    body = reply.content if isinstance(reply.content, str) else str(reply.content)
    print("  agent reply (first 400 chars):")
    print("    " + body[:400].replace("\n", "\n    "))


def _stream_invocation(graph, payload, config) -> dict:
    """Drive one graph invocation in streaming mode and return the final values event."""
    final_state: dict = {}
    deltas = 0
    for mode, event in graph.stream(payload, config=config, stream_mode=["values", "custom"]):
        if mode == "values":
            final_state = event
        elif mode == "custom" and isinstance(event, dict) and event.get("delta"):
            deltas += 1
    if deltas:
        print(f"  streamed {deltas} token deltas")
    return final_state


def run_turn(graph, thread_id: str, user_text: str) -> dict:
    """Run one turn, auto-approving any human-approval interrupt for the demo."""
    config = {"configurable": {"thread_id": thread_id}}
    state_in = {
        "messages": [HumanMessage(content=user_text)],
        "context": AgentContext.from_settings(_SETTINGS),
    }
    result = _stream_invocation(graph, state_in, config)
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value if result["__interrupt__"] else {}
        print(f"  ⏸  interrupt: {payload} → auto-approving for demo")
        result = _stream_invocation(graph, Command(resume={"approved": True}), config)
    return result


def main() -> None:
    graph = get_graph()

    thread1 = f"thread-{uuid.uuid4().hex[:8]}"
    q1 = "I need to ship 15,000 lbs Austin to Dallas — what's my best option?"
    final1 = run_turn(graph, thread1, q1)
    _print_turn_summary("Turn 1 (session A)", final1)

    namespace = memory_namespace(REALM, USER)
    raw_total = len(list(get_store().search(namespace, query="carrier preferences", limit=50)))
    live = list(get_semantic_memory().search(REALM, USER, query="carrier preferences", limit=10))
    print(f"\n[store] semantic LTM after turn 1: {len(live)} live (of {raw_total} raw; tombstoned rows filtered)")
    for f in live:
        marker = "canonical" if f.key.startswith("canon_") else "fact"
        print(f"  - [{marker}] {f.key}: {f.content[:120]}")

    thread2 = f"thread-{uuid.uuid4().hex[:8]}"
    q2 = "What carrier did I prefer last time on this lane?"
    final2 = run_turn(graph, thread2, q2)
    _print_turn_summary("Turn 2 (new session, cross-session recall)", final2)

    thread3 = f"thread-{uuid.uuid4().hex[:8]}"
    q3 = "Which carriers serve TX-AZ with no fuel surcharge and a weight threshold above 18000 lbs?"
    final3 = run_turn(graph, thread3, q3)
    _print_turn_summary("Turn 3 (KG multi-constraint)", final3)

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
