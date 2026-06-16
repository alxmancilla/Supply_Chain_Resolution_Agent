"""Streamlit UI for the Supply Chain Resolution Agent demo.

Two-panel layout:
  * Left  — agent chat (per-turn expanders for LTM, RAG, response, latency).
  * Right — live memory inspector reading the agent_memories collection.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

load_dotenv()

from agent.graph import get_graph
from agent.memory import (
    DB_NAME,
    EPISODES_COLLECTION,
    KG_CARRIERS_COLLECTION,
    KG_LANES_COLLECTION,
    KG_SERVES_COLLECTION,
    KG_SLAS_COLLECTION,
    MEMORIES_COLLECTION,
    PROCEDURES_COLLECTION,
    get_mongo_client,
    memory_namespace,
)
from core.memory import reset_memory_cache
from core.settings import AgentContext, get_settings

_SETTINGS = get_settings()
REALM_ID = _SETTINGS.realm_id
USER_ID = _SETTINGS.user_id
AGENT_ID = _SETTINGS.agent_id

LTM_GREEN_MS, LTM_RED_MS = 50.0, 500.0
RAG_GREEN_MS, RAG_RED_MS = 100.0, 500.0

st.set_page_config(page_title="Supply Chain Resolution Agent", layout="wide")


def _color_for(value: float, green: float, red: float) -> str:
    if value <= green:
        return "#1a7f37"  # green
    if value >= red:
        return "#cf222e"  # red
    return "#bf8700"  # amber


DEFAULT_THREAD_ID = "session-demo-live"

DEMO_PROMPTS: list[tuple[str, str, str]] = [
    (
        "Recommend (low-cost)",
        "I need to ship 15,000 lbs Austin to Dallas — what's my best option?",
        "Router → recommend; all branches fire; booking draft auto-executes (under $10k).",
    ),
    (
        "Propose: hot-lane rule",
        "Going forward, always prefer Carrier A on the Austin-Dallas hot lane and escalate any switch to a different carrier for that lane.",
        "Router → propose_procedure; HIL approval card commits to `agent_procedures`.",
    ),
    (
        "Propose: $7.5K threshold",
        "From now on, escalate any booking above $7,500 for Acme Manufacturing, not the default $10,000.",
        "Second proposed rule; both rules then compose on the next booking.",
    ),
    (
        "Compose: $8.2K Carrier C",
        "Book 18,000 lbs Austin → Phoenix on Carrier C for next Tuesday — estimated cost $8,200.",
        "Both new rules fire: pushes back on the carrier switch AND interrupts (new $7.5k ceiling).",
    ),
    (
        "Recall preference",
        "What carrier did I prefer last time on TX-AZ?",
        "Cross-session LTM recall — runs after turn 1 has written facts.",
    ),
    (
        "KG multi-constraint",
        "Which carriers serve TX-AZ with no fuel surcharge and a weight threshold above 18,000 lbs?",
        "Structured lookup — KG hop-1 + hop-2 with SLA filters.",
    ),
    (
        "Policy lookup (RAG)",
        "What are the hours-of-service limits I need to plan around for a same-day Austin–Dallas turn?",
        "RAG-only branch; cites `policies/hours_of_service.pdf`.",
    ),
    (
        "High-cost booking (interrupt)",
        "Book 42,000 lbs Austin → Phoenix on Carrier A for next Tuesday — estimated cost $14,500.",
        "Triggers the >$10k human-approval interrupt; resume Approve/Reject.",
    ),
]


def _init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = DEFAULT_THREAD_ID
    if "turns" not in st.session_state:
        st.session_state.turns = []  # list of dicts: user, ai, ltm_hits, rag_hits, latency_ms
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None  # dict | None
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None  # str | None


def _new_session() -> None:
    st.session_state.thread_id = f"session-demo-{uuid.uuid4().hex[:8]}"
    st.session_state.turns = []
    st.session_state.pending_approval = None
    st.session_state.queued_prompt = None
    reset_memory_cache()


def _ns_filter() -> dict:
    return {"namespace": list(memory_namespace(REALM_ID, USER_ID))}


def _fetch_semantic() -> pd.DataFrame:
    coll = get_mongo_client()[DB_NAME][MEMORIES_COLLECTION]
    cursor = coll.find(_ns_filter(), {"_id": 0, "key": 1, "value": 1, "updated_at": 1}).sort("updated_at", -1)
    rows = []
    for doc in cursor:
        value = doc.get("value") or {}
        rows.append({"content": value.get("content", str(value)), "key": doc.get("key")})
    return pd.DataFrame(rows, columns=["content", "key"])


def _fetch_episodes() -> pd.DataFrame:
    coll = get_mongo_client()[DB_NAME][EPISODES_COLLECTION]
    cursor = coll.find(_ns_filter(), {"_id": 0, "key": 1, "value": 1, "updated_at": 1}).sort("updated_at", -1)
    rows = []
    for doc in cursor:
        value = doc.get("value") or {}
        rows.append(
            {
                "summary": value.get("summary", ""),
                "lane": value.get("lane", ""),
                "recommendation": value.get("recommendation", ""),
                "outcome": value.get("outcome", ""),
                "occurred_at": value.get("occurred_at", ""),
                "key": doc.get("key"),
            }
        )
    return pd.DataFrame(
        rows, columns=["summary", "lane", "recommendation", "outcome", "occurred_at", "key"]
    )


def _fetch_procedures() -> pd.DataFrame:
    coll = get_mongo_client()[DB_NAME][PROCEDURES_COLLECTION]
    cursor = coll.find(
        {"realm_id": REALM_ID, "agent_id": AGENT_ID, "active": True},
        {"_id": 0, "rule_id": 1, "category": 1, "rule": 1},
    ).sort("rule_id", 1)
    return pd.DataFrame(list(cursor), columns=["rule_id", "category", "rule"])


def _fetch_kg_counts() -> dict[str, int]:
    db = get_mongo_client()[DB_NAME]
    filt = {"realm_id": REALM_ID}
    return {
        "carriers": db[KG_CARRIERS_COLLECTION].count_documents(filt),
        "lanes": db[KG_LANES_COLLECTION].count_documents(filt),
        "slas": db[KG_SLAS_COLLECTION].count_documents(filt),
        "serves": db[KG_SERVES_COLLECTION].count_documents(filt),
    }


def _fetch_kg_serves() -> pd.DataFrame:
    db = get_mongo_client()[DB_NAME]
    rows = list(db[KG_SERVES_COLLECTION].find(
        {"realm_id": REALM_ID},
        {"_id": 0, "carrier_id": 1, "lane_id": 1, "priority": 1, "since": 1},
    ).sort("lane_id", 1))
    return pd.DataFrame(rows, columns=["carrier_id", "lane_id", "priority", "since"])


def _fetch_kg_slas() -> pd.DataFrame:
    db = get_mongo_client()[DB_NAME]
    rows = list(db[KG_SLAS_COLLECTION].find(
        {"realm_id": REALM_ID},
        {"_id": 0, "sla_id": 1, "carrier_id": 1, "lane_id": 1,
         "surcharge_rate": 1, "weight_threshold_lb": 1, "transit_hours": 1},
    ).sort("sla_id", 1))
    return pd.DataFrame(
        rows,
        columns=["sla_id", "carrier_id", "lane_id", "surcharge_rate", "weight_threshold_lb", "transit_hours"],
    )


def _stream_invocation(graph, payload, config, placeholder, buffer: list[str]) -> dict:
    """Stream one graph invocation; append token deltas to `buffer` and live-update `placeholder`."""
    final_state: dict = {}
    for mode, event in graph.stream(payload, config=config, stream_mode=["values", "custom"]):
        if mode == "values":
            final_state = event
        elif mode == "custom" and isinstance(event, dict) and event.get("delta"):
            buffer.append(event["delta"])
            placeholder.markdown("".join(buffer) + " ▍")
    return final_state


def _finalize_turn(user_input: str, result: dict, interrupts: list[dict]) -> None:
    ai_message = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            ai_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    st.session_state.turns.append(
        {
            "user": user_input,
            "ai": ai_message,
            "ltm_hits": result.get("ltm_hits", []),
            "rag_hits": result.get("rag_hits", []),
            "episode_hits": result.get("episode_hits", []),
            "procedure_hits": result.get("procedure_hits", []),
            "kg_hits": result.get("kg_hits", []),
            "action_plan": result.get("action_plan", {}),
            "booking_draft": result.get("booking_draft", {}),
            "procedure_proposal": result.get("procedure_proposal", {}),
            "interrupts": interrupts,
            "latency_ms": result.get("latency_ms", {}),
            "usage": result.get("usage", {}),
            "degraded": result.get("degraded", []),
            "routing": result.get("routing", {}),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }
    )


def _drive_turn(payload, *, user_input: str, prior_interrupts: list[dict]) -> None:
    """Stream until completion or until an interrupt fires.

    On interrupt: stash mid-turn state in `session_state.pending_approval`
    (the LangGraph checkpointer holds the real graph state) and stop, so
    the next render can show the approval card. On completion: append the
    finalized turn record.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    stream_box = st.empty()
    stream_box.markdown("_streaming…_")
    buffer: list[str] = []
    result = _stream_invocation(graph, payload, config, stream_box, buffer)
    interrupts = list(prior_interrupts)
    if result.get("__interrupt__"):
        intr = result["__interrupt__"][0]
        payload_value = intr.value if hasattr(intr, "value") else intr
        interrupts.append(payload_value if isinstance(payload_value, dict) else {"value": payload_value})
        st.session_state.pending_approval = {
            "user_input": user_input,
            "interrupts": interrupts,
            "request": payload_value,
            "partial_reply": "".join(buffer),
        }
        stream_box.empty()
        return
    stream_box.empty()
    _finalize_turn(user_input, result, interrupts)


def _run_turn(user_input: str) -> None:
    _drive_turn(
        {
            "messages": [HumanMessage(content=user_input)],
            "context": AgentContext.from_settings(_SETTINGS),
        },
        user_input=user_input,
        prior_interrupts=[],
    )


def _resume_turn(*, approved: bool, approver: str) -> None:
    pending = st.session_state.pending_approval
    if not pending:
        return
    st.session_state.pending_approval = None
    _drive_turn(
        Command(resume={"approved": approved, "approver": approver}),
        user_input=pending["user_input"],
        prior_interrupts=pending["interrupts"],
    )


def _render_approval_card() -> None:
    pending = st.session_state.pending_approval
    if not pending:
        return
    req = pending.get("request") or {}
    kind = req.get("kind", "approval_request")
    with st.container(border=True):
        if kind == "procedure_proposal":
            _render_procedure_proposal_card(req, pending)
        else:
            _render_booking_approval_card(req, pending)


def _render_booking_approval_card(req: dict, pending: dict) -> None:
    st.markdown("### ⏸ Human approval required")
    st.caption(f"Draft `{req.get('draft_id', '?')}`")
    cols = st.columns(3)
    cols[0].metric("Carrier", req.get("carrier") or "—")
    cols[1].metric("Lane", req.get("lane") or "—")
    cost = req.get("estimated_cost_usd")
    cols[2].metric(
        "Est. cost",
        f"${cost:,.0f}" if isinstance(cost, (int, float)) else "—",
    )
    weight = req.get("weight_lb")
    if isinstance(weight, (int, float)):
        st.caption(f"Weight: {weight:,.0f} lb · Rationale: {req.get('rationale', '—')}")
    else:
        st.caption(f"Rationale: {req.get('rationale', '—')}")
    approver = st.text_input(
        "Approver", value=os.environ.get("USER", "ops-demo"), key="approval_approver"
    )
    partial = pending.get("partial_reply") or ""
    with st.expander("Streamed draft reply", expanded=False):
        st.caption("Tokens emitted before the interrupt fired.")
        st.markdown(partial or "_(no tokens streamed before interrupt)_")
    with st.expander("Raw proposal JSON", expanded=False):
        st.caption("Payload passed to `interrupt()` from `execute_action`.")
        st.code(json.dumps(req, indent=2, default=str), language="json")
    action_cols = st.columns(2)
    if action_cols[0].button("✅ Approve & execute", type="primary", use_container_width=True):
        _resume_turn(approved=True, approver=approver)
        st.rerun()
    if action_cols[1].button("❌ Reject", use_container_width=True):
        _resume_turn(approved=False, approver=approver)
        st.rerun()


def _render_procedure_proposal_card(req: dict, pending: dict) -> None:
    st.markdown("### ⏸ New operating rule proposed")
    st.caption(f"Proposal `{req.get('proposal_id', '?')}` · category: `{req.get('category', 'general')}`")
    st.markdown(f"> **{req.get('rule', '(empty rule)')}**")
    rationale = req.get("rationale") or ""
    if rationale:
        st.caption(f"Rationale: {rationale}")
    st.info(
        "Approving writes this rule to `agent_procedures` (active=True). It "
        "will be injected into the system prompt on every future turn for "
        "this tenant. Rejecting marks the staged proposal as `rejected` and "
        "leaves the active rule set untouched."
    )
    approver = st.text_input(
        "Approver", value=os.environ.get("USER", "ops-demo"), key="approval_approver"
    )
    partial = pending.get("partial_reply") or ""
    with st.expander("Streamed draft reply", expanded=False):
        st.markdown(partial or "_(no tokens streamed before interrupt)_")
    with st.expander("Raw proposal JSON", expanded=False):
        st.code(json.dumps(req, indent=2, default=str), language="json")
    action_cols = st.columns(2)
    if action_cols[0].button("✅ Approve & commit rule", type="primary", use_container_width=True):
        _resume_turn(approved=True, approver=approver)
        st.rerun()
    if action_cols[1].button("❌ Reject", use_container_width=True):
        _resume_turn(approved=False, approver=approver)
        st.rerun()


def _render_latency(latency: dict) -> None:
    total = sum(latency.values()) if latency else 0.0
    router = latency.get("router_ms", 0.0)
    ltm = latency.get("ltm_ms", 0.0)
    eps = latency.get("episodes_ms", 0.0)
    proc = latency.get("procedures_ms", 0.0)
    rag = latency.get("rag_ms", 0.0)
    kg = latency.get("kg_ms", 0.0)
    llm = latency.get("llm_ms", 0.0)
    ttft = latency.get("llm_ttft_ms")
    save = latency.get("save_ms", 0.0)
    cols = st.columns(8)
    cols[0].markdown(f"**Router** {router:.0f} ms")
    cols[1].markdown(
        f"**Semantic** <span style='color:{_color_for(ltm, LTM_GREEN_MS, LTM_RED_MS)}'>{ltm:.0f} ms</span>",
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f"**Episodic** <span style='color:{_color_for(eps, LTM_GREEN_MS, LTM_RED_MS)}'>{eps:.0f} ms</span>",
        unsafe_allow_html=True,
    )
    cols[3].markdown(f"**Procedural** {proc:.0f} ms")
    cols[4].markdown(
        f"**RAG** <span style='color:{_color_for(rag, RAG_GREEN_MS, RAG_RED_MS)}'>{rag:.0f} ms</span>",
        unsafe_allow_html=True,
    )
    cols[5].markdown(
        f"**KG** <span style='color:{_color_for(kg, RAG_GREEN_MS, RAG_RED_MS)}'>{kg:.0f} ms</span>",
        unsafe_allow_html=True,
    )
    ttft_txt = f" (ttft {ttft:.0f} ms)" if isinstance(ttft, (int, float)) else ""
    cols[6].markdown(f"**LLM** {llm:.0f} ms{ttft_txt}")
    cols[7].markdown(f"**Total** {total:.0f} ms (save {save:.0f} ms)")


def _render_usage(usage: dict) -> None:
    if not usage:
        return
    tokens_in = int(usage.get("tokens_in", 0))
    tokens_out = int(usage.get("tokens_out", 0))
    calls = int(usage.get("calls", 0))
    cost = float(usage.get("cost_usd", 0.0))
    cols = st.columns(4)
    cols[0].markdown(f"**Tokens in** {tokens_in:,}")
    cols[1].markdown(f"**Tokens out** {tokens_out:,}")
    cols[2].markdown(f"**LLM calls** {calls}")
    cols[3].markdown(f"**Cost** ${cost:.4f}")


def _routing_caption(turn: dict, branch: str) -> str | None:
    routing = turn.get("routing") or {}
    branches = routing.get("branches")
    if branches is not None and branch not in branches:
        return "Skipped by intent router for this turn."
    return None


def _render_turn(turn: dict, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        routing = turn.get("routing") or {}
        if routing:
            label = routing.get("intent_label", "?")
            branches = routing.get("branches") or []
            rationale = routing.get("rationale", "")
            st.markdown(
                f"🎯 **intent:** `{label}` · branches: "
                + ", ".join(f"`{b}`" for b in branches)
                + (f"  \n_{rationale}_" if rationale else "")
            )
        if turn.get("degraded"):
            st.warning("⚠️ Degraded retrieval: " + " · ".join(turn["degraded"]))
        with st.expander(f"Semantic LTM ({len(turn['ltm_hits'])} memories)", expanded=False):
            skip = _routing_caption(turn, "ltm")
            if skip:
                st.caption(skip)
            elif turn["ltm_hits"]:
                for hit in turn["ltm_hits"]:
                    score = hit.get("score")
                    score_txt = f" — score {score:.3f}" if isinstance(score, (int, float)) else ""
                    st.markdown(f"- **{hit.get('key', '?')}**{score_txt}\n\n  {hit.get('content', '')}")
            else:
                st.caption("No semantic memories retrieved for this turn.")
        with st.expander(f"Episodic LTM ({len(turn['episode_hits'])} episodes)", expanded=False):
            skip = _routing_caption(turn, "episodes")
            if skip:
                st.caption(skip)
            elif turn["episode_hits"]:
                for hit in turn["episode_hits"]:
                    score = hit.get("score")
                    score_txt = f" — score {score:.3f}" if isinstance(score, (int, float)) else ""
                    st.markdown(
                        f"- **{hit.get('key', '?')}**{score_txt} · lane `{hit.get('lane', '?')}` "
                        f"· {hit.get('occurred_at', '?')}\n\n  {hit.get('summary', '')}\n\n"
                        f"  _rec_: {hit.get('recommendation', '-')} · _outcome_: {hit.get('outcome', '-')}"
                    )
            else:
                st.caption("No prior episodes retrieved for this turn.")
        with st.expander(f"Procedural LTM ({len(turn['procedure_hits'])} active rules)", expanded=False):
            skip = _routing_caption(turn, "procedures")
            if skip:
                st.caption(skip)
            elif turn["procedure_hits"]:
                for hit in turn["procedure_hits"]:
                    st.markdown(
                        f"- **{hit.get('rule_id', '?')}** ({hit.get('category', 'general')}): "
                        f"{hit.get('rule', '')}"
                    )
            else:
                st.caption("No procedural rules active.")
        with st.expander(f"RAG retrieved (top {len(turn['rag_hits'])} chunks)", expanded=False):
            skip = _routing_caption(turn, "rag")
            if skip:
                st.caption(skip)
            elif turn["rag_hits"]:
                for hit in turn["rag_hits"][:3]:
                    st.markdown(
                        f"- **{hit.get('doc_type', '?')}** :: `{hit.get('source', '?')}` "
                        f"— score {hit.get('score', 0):.3f}\n\n  {hit.get('text', '')}"
                    )
            else:
                st.caption("No RAG chunks retrieved for this turn.")
        kg_hits = turn.get("kg_hits", [])
        with st.expander(f"Knowledge graph ({len(kg_hits)} structured facts)", expanded=False):
            skip = _routing_caption(turn, "kg")
            if skip:
                st.caption(skip)
            elif kg_hits:
                for hit in kg_hits:
                    st.markdown(f"- {hit.get('fact', '')}")
            else:
                st.caption("No graph facts retrieved for this turn.")
        with st.expander("Response", expanded=True):
            st.markdown(turn["ai"] or "_(empty response)_")
        _render_latency(turn["latency_ms"])
        _render_usage(turn.get("usage", {}))


def _render_demo_sidebar() -> None:
    pending = bool(st.session_state.get("pending_approval"))
    with st.sidebar:
        st.markdown("### 🎬 Demo prompts")
        st.caption("One click → queues the prompt and runs it as the next turn.")
        for label, prompt, hint in DEMO_PROMPTS:
            if st.button(label, key=f"demo_{label}", use_container_width=True, disabled=pending):
                st.session_state.queued_prompt = prompt
                st.rerun()
            st.caption(hint)
        st.divider()
        st.caption(
            "Order matters for **Recall preference** (needs a prior turn that wrote LTM)."
        )


def main() -> None:
    _init_state()
    st.title("Supply Chain Resolution Agent")
    st.caption(
        f"Realm `{REALM_ID}` · User `{USER_ID}` · Thread `{st.session_state.thread_id}` · "
        "One Atlas cluster · Semantic + Episodic + Procedural LTM + RAG + KG in `by_genai`"
    )

    _render_demo_sidebar()

    left, right = st.columns([2, 1], gap="large")

    with left:
        controls = st.columns([1, 1, 4])
        if controls[0].button("🔄 New Session", help="Reset thread_id (LTM persists)"):
            _new_session()
            st.rerun()
        controls[1].markdown(f"**Turns:** {len(st.session_state.turns)}")
        for idx, turn in enumerate(st.session_state.turns):
            _render_turn(turn, idx)
        pending = st.session_state.pending_approval
        if pending:
            with st.chat_message("user"):
                st.markdown(pending["user_input"])
            with st.chat_message("assistant"):
                _render_approval_card()
        user_input = st.chat_input(
            "Resolve the pending approval to continue..." if pending else "Ask the supply chain agent...",
            disabled=bool(pending),
        )
        queued = st.session_state.queued_prompt
        if queued and not pending:
            st.session_state.queued_prompt = None
            user_input = queued
        if user_input and not pending:
            with st.spinner("Running agent (parallel retrieve → LLM → save)..."):
                _run_turn(user_input)
            st.rerun()

    with right:
        st.subheader("🧠 Long-Term Memory Inspector")
        st.caption(f"Namespace: `{REALM_ID}` / `{USER_ID}` — refreshed every turn")
        tab_sem, tab_epi, tab_proc, tab_kg = st.tabs(
            ["Semantic", "Episodic", "Procedural", "Knowledge Graph"]
        )

        with tab_sem:
            df = _fetch_semantic()
            st.metric("Semantic memories", len(df))
            if df.empty:
                st.info("Empty. Seed with `python -m data.seed_memories`.")
            else:
                st.dataframe(df, hide_index=True, use_container_width=True, height=520)

        with tab_epi:
            df = _fetch_episodes()
            st.metric("Episodes recorded", len(df))
            if df.empty:
                st.info("Empty. Seed with `python -m data.seed_episodes`.")
            else:
                st.dataframe(df, hide_index=True, use_container_width=True, height=520)

        with tab_proc:
            df = _fetch_procedures()
            st.metric("Active rules", len(df))
            if df.empty:
                st.info("Empty. Seed with `python -m data.seed_procedures`.")
            else:
                st.dataframe(df, hide_index=True, use_container_width=True, height=520)

        with tab_kg:
            counts = _fetch_kg_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Carriers", counts["carriers"])
            m2.metric("Lanes", counts["lanes"])
            m3.metric("SLAs", counts["slas"])
            m4.metric("Serves edges", counts["serves"])
            if sum(counts.values()) == 0:
                st.info("Empty. Seed with `python -m data.seed_kg`.")
            else:
                st.markdown("**Serves edges** (carrier → lane)")
                st.dataframe(_fetch_kg_serves(), hide_index=True, use_container_width=True, height=220)
                st.markdown("**SLAs** (per carrier × lane)")
                st.dataframe(_fetch_kg_slas(), hide_index=True, use_container_width=True, height=260)


if __name__ == "__main__":
    main()
