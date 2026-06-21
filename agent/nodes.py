"""LangGraph nodes for the Supply Chain Resolution Agent.

Each node records its own latency in `state["latency_ms"]` so the Streamlit
UI can render per-turn metrics with color-coded thresholds.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

try:
    from langgraph.config import get_stream_writer as _get_stream_writer
except ImportError:  # pragma: no cover - older langgraph
    _get_stream_writer = None

from core.latency import timed
from core.kg import get_entity_extractor, get_knowledge_graph
from core.memory import (
    get_episodic_memory,
    get_procedural_memory,
    get_semantic_memory,
)
from core.providers.chat.retry import invoke_typed_with_retry
from core.providers.registry import get_chat_provider
from core.rag.mongo import get_knowledge_retriever
from core.resilience import safe_retrieve
from core.router import get_intent_router
from core.schemas import (
    ALL_BRANCHES,
    BookingProposal,
    EvidenceReflection,
    ProcedureProposal,
    ResearchPlan,
)
from core.settings import get_settings
from core.usage import extract_usage_metadata, merge_usage, usage_payload


def _merge_latency(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    """Reducer for the latency_ms channel — preserves entries from prior nodes."""
    merged: dict[str, float] = {}
    if left:
        merged.update(left)
    if right:
        merged.update(right)
    return merged


_DEGRADED_RESET = "__RESET__"


def _merge_degraded(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for the degraded channel.

    Concatenates failure markers within a turn but supports a `__RESET__`
    sentinel emitted by the entry node so accumulated markers from prior
    turns on the same thread don't leak forward. Duplicate markers within
    the same turn collapse to one (preserves first-seen order).
    """
    right = right or []
    if _DEGRADED_RESET in right:
        idx = right.index(_DEGRADED_RESET)
        base: list[str] = list(right[idx + 1 :])
    else:
        base = [*(left or []), *right]
    seen: dict[str, None] = {}
    for marker in base:
        if marker != _DEGRADED_RESET:
            seen.setdefault(marker, None)
    return list(seen.keys())

from core.settings import AgentContext

from core.schemas import MemoryExtraction

from .memory import EMBEDDING_DIMS, get_booking_drafts_collection, get_embeddings  # noqa: F401  (re-exported for callers)
from .prompts import (
    ACTION_PLANNING_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    PROCEDURE_PROPOSAL_PROMPT,
    REFLECTION_PROMPT,
    SYSTEM_PROMPT,
)

RAG_TOP_K = 5
LTM_TOP_K = 5
EPISODES_TOP_K = 3
KG_TOP_K = 8
AGENT_ID = "supply-chain-resolution-agent"
MAX_REPLANS = 1
_GROUNDING_INTENTS = frozenset({"recommend_shipment", "lookup_policy", "structured_lookup", "fallback"})

_TURN_COUNTERS: dict[tuple[str, str], int] = {}


def _reset_turn_counters() -> None:
    """Test hook: clear the per-process reflection counters."""
    _TURN_COUNTERS.clear()


def _run_reflection_pass(realm_id: str, user_id: str, threshold: float) -> dict[str, int]:
    """Run `LLMMemoryReflector` against semantic + episodic namespaces.

    Factored out so unit tests can monkeypatch around the live Atlas/LLM
    calls. Returns a flat summary that gets merged into `state['reflection']`.
    """
    from agent.memory import (
        DB_NAME,
        EPISODES_COLLECTION,
        MEMORIES_COLLECTION,
        get_mongo_client,
    )
    from core.memory.reflector import LLMMemoryReflector, MongoMemoryAdmin
    from core.providers.registry import get_embedding_provider

    chat = get_chat_provider()
    embeddings = get_embedding_provider()
    db = get_mongo_client()[DB_NAME]
    summary = {"clusters_found": 0, "canonical_written": 0, "tombstoned": 0}
    for coll_name, field in (
        (MEMORIES_COLLECTION, "content"),
        (EPISODES_COLLECTION, "summary"),
    ):
        admin = MongoMemoryAdmin(collection=db[coll_name], content_field=field, embeddings=embeddings)
        reflector = LLMMemoryReflector(admin=admin, chat=chat, similarity_threshold=threshold)
        report = reflector.reflect(realm_id, user_id)
        summary["clusters_found"] += report.clusters_found
        summary["canonical_written"] += len(report.canonical_written)
        summary["tombstoned"] += len(report.tombstoned_keys)
    return summary


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    context: AgentContext
    routing: dict[str, Any]
    plan: dict[str, Any]
    reflection_eval: dict[str, Any]
    ltm_context: str
    rag_context: str
    episodic_context: str
    procedural_context: str
    kg_context: str
    ltm_hits: list[dict[str, Any]]
    rag_hits: list[dict[str, Any]]
    episode_hits: list[dict[str, Any]]
    procedure_hits: list[dict[str, Any]]
    kg_hits: list[dict[str, Any]]
    action_plan: dict[str, Any]
    booking_draft: dict[str, Any]
    procedure_proposal: dict[str, Any]
    latency_ms: Annotated[dict[str, float], _merge_latency]
    degraded: Annotated[list[str], _merge_degraded]
    usage: Annotated[dict[str, float], merge_usage]
    reflection: dict[str, int]


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Backward-compat shim: returns the raw langchain client behind the active ChatProvider.

    The graph's `generate_response` node needs to invoke the LLM with a full
    `BaseMessage` list (system + history) which is not on the protocol; this
    escape hatch keeps that flow working while new code (router, extractors)
    talks to `get_chat_provider()` directly.
    """
    return get_chat_provider().underlying()


def _last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _query_for(state: AgentState) -> str:
    """Return the active retrieval query — `plan.subquery` if set, else the last user text.

    Lets the Reflection / Think & Plan loop substitute a refined subquery for
    a rescue retrieval pass without changing retriever call sites.
    """
    plan = state.get("plan") or {}
    subquery = plan.get("subquery")
    if isinstance(subquery, str) and subquery.strip():
        return subquery
    return _last_user_text(state["messages"])


def _record_chat_fallback(chat: Any, out: dict[str, Any]) -> None:
    """Append `chat_fallback:<provider>` to `out['degraded']` when the chat
    provider just failed over to a secondary. No-op for non-fallback providers.
    """
    fallback = getattr(chat, "last_fallback", None)
    if not fallback:
        return
    marker = f"chat_fallback:{fallback}"
    markers = list(out.get("degraded") or [])
    if marker not in markers:
        markers.append(marker)
    out["degraded"] = markers


def _append_marker(out: dict[str, Any], marker: str) -> None:
    """Append `marker` to `out['degraded']` (deduped, order-preserving)."""
    markers = list(out.get("degraded") or [])
    if marker not in markers:
        markers.append(marker)
    out["degraded"] = markers


def _record_structured_retry(chat: Any, out: dict[str, Any], *, node: str) -> None:
    """Append `structured_retry:<node>` when invoke_typed needed >1 attempt."""
    attempts = getattr(chat, "last_structured_attempts", 1) or 1
    if attempts > 1:
        _append_marker(out, f"structured_retry:{node}")


_ROUTER_FALLBACK = {
    "intent_label": "fallback",
    "branches": list(ALL_BRANCHES),
    "rationale": "router unavailable",
}


@timed("router_ms")
@safe_retrieve("classify_intent", routing=_ROUTER_FALLBACK)
def classify_intent(state: AgentState) -> dict[str, Any]:
    query = _last_user_text(state["messages"])
    reset: dict[str, Any] = {"degraded": [_DEGRADED_RESET]}
    if not query:
        return {**reset, "routing": dict(_ROUTER_FALLBACK)}
    decision = get_intent_router().route(query)
    routing = decision.model_dump()
    if not routing.get("branches"):
        routing["branches"] = list(ALL_BRANCHES)
    return {**reset, "routing": routing}


_REPLAN_BRANCHES: tuple[str, ...] = ("rag", "kg", "procedures")


@timed("plan_ms")
def think_and_plan(state: AgentState) -> dict[str, Any]:
    """Process-reflection step that produces the per-turn retrieval plan.

    First pass: mirrors the router's branches into `plan` with no subquery
    rewrite (zero LLM cost). On a replan pass — triggered when the
    Reflection Agent emits `sufficient=False` with a `followup_subquery` —
    narrows to grounding-oriented branches and substitutes the refined
    subquery so the retrievers run a rescue pass against tighter terms.
    """
    routing = state.get("routing") or {}
    router_branches = list(routing.get("branches") or ALL_BRANCHES)
    prior_plan = state.get("plan") or {}
    reflection = state.get("reflection_eval") or {}
    replan_count = int(prior_plan.get("replan_count") or 0)

    needs_replan = (
        bool(prior_plan)
        and reflection.get("sufficient") is False
        and replan_count < MAX_REPLANS
    )

    if not needs_replan:
        plan = ResearchPlan(
            branches=router_branches,
            subquery=None,
            rationale="first-pass plan mirrors router branches",
            replan_count=0,
        )
        return {"plan": plan.model_dump()}

    followup = reflection.get("followup_subquery")
    subquery = followup.strip() if isinstance(followup, str) and followup.strip() else None
    narrowed = [b for b in router_branches if b in _REPLAN_BRANCHES] or list(_REPLAN_BRANCHES)
    plan = ResearchPlan(
        branches=narrowed,
        subquery=subquery,
        rationale="replan: narrowed to grounding branches with refined subquery",
        replan_count=replan_count + 1,
    )
    return {"plan": plan.model_dump()}


@timed("ltm_ms")
@safe_retrieve("retrieve_ltm", ltm_context="(retrieval degraded)", ltm_hits=[])
def retrieve_ltm(state: AgentState) -> dict[str, Any]:
    query = _query_for(state)
    ctx = state["context"]
    hits: list[dict[str, Any]] = []
    context = "(no prior memories found)"
    if query:
        facts = get_semantic_memory().search(ctx.realm_id, ctx.user_id, query=query, limit=LTM_TOP_K)
        hits = [f.model_dump() for f in facts]
        if hits:
            context = "\n".join(f"- {h['content']}" for h in hits)
    return {"ltm_context": context, "ltm_hits": hits}


@timed("rag_ms")
@safe_retrieve("retrieve_rag", rag_context="(retrieval degraded)", rag_hits=[])
def retrieve_rag(state: AgentState) -> dict[str, Any]:
    query = _query_for(state)
    hits: list[dict[str, Any]] = []
    context = "(no knowledge chunks retrieved)"
    if query:
        results = get_knowledge_retriever().query(
            realm_id=state["context"].realm_id, text=query, k=RAG_TOP_K
        )
        hits = [h.model_dump() for h in results]
        if hits:
            context = "\n\n".join(
                f"[{h['doc_type']} :: {h['source']} :: score={h['score']:.3f}]\n{h['text']}"
                for h in hits
            )
    return {"rag_context": context, "rag_hits": hits}


@timed("episodes_ms")
@safe_retrieve("retrieve_episodes", episodic_context="(retrieval degraded)", episode_hits=[])
def retrieve_episodes(state: AgentState) -> dict[str, Any]:
    query = _query_for(state)
    ctx = state["context"]
    hits: list[dict[str, Any]] = []
    context = "(no prior episodes found)"
    if query:
        episodes = get_episodic_memory().search(
            ctx.realm_id, ctx.user_id, query=query, limit=EPISODES_TOP_K
        )
        hits = [e.model_dump() for e in episodes]
        if hits:
            context = "\n".join(
                f"- [{h.get('occurred_at', '?')} | lane={h.get('lane', '?')}] {h['summary']} "
                f"(rec: {h.get('recommendation', '-')}; outcome: {h.get('outcome', '-')})"
                for h in hits
            )
    return {"episodic_context": context, "episode_hits": hits}


@timed("procedures_ms")
@safe_retrieve("retrieve_procedures", procedural_context="(retrieval degraded)", procedure_hits=[])
def retrieve_procedures(state: AgentState) -> dict[str, Any]:
    ctx = state["context"]
    rules = get_procedural_memory().list_active(ctx.realm_id, ctx.agent_id)
    hits = [r.model_dump() for r in rules]
    if hits:
        context = "\n".join(f"- ({h.get('category', 'general')}) {h['rule']}" for h in hits)
    else:
        context = "(no procedural rules configured)"
    return {"procedural_context": context, "procedure_hits": hits}


@timed("kg_ms")
@safe_retrieve("retrieve_kg", kg_context="(retrieval degraded)", kg_hits=[])
def retrieve_kg(state: AgentState) -> dict[str, Any]:
    query = _query_for(state)
    ctx = state["context"]
    if not query:
        return {"kg_context": "(no query)", "kg_hits": []}
    spec = get_entity_extractor().extract(query)
    if not spec.lanes and not spec.carriers:
        return {"kg_context": "(no entities resolved from query)", "kg_hits": []}
    subgraph = get_knowledge_graph().query(ctx.realm_id, spec, limit=KG_TOP_K)
    hits = [
        {**e.model_dump(), "fact": fact}
        for e, fact in zip(subgraph.edges, subgraph.facts)
    ]
    context = (
        "\n".join(subgraph.facts)
        if subgraph.facts
        else "(graph traversal returned no matching facts)"
    )
    return {"kg_context": context, "kg_hits": hits}


def _evidence_summary(state: AgentState) -> str:
    parts: list[str] = []
    for label, field in (
        ("rag", "rag_context"),
        ("kg", "kg_context"),
        ("procedures", "procedural_context"),
        ("episodes", "episodic_context"),
        ("ltm", "ltm_context"),
    ):
        text = state.get(field) or ""
        text = text.strip()
        if text and not text.startswith("("):
            parts.append(f"[{label}] {text[:280]}")
    return "\n".join(parts) if parts else "(no grounding evidence retrieved)"


@timed("reflect_ms")
def reflect_on_evidence(state: AgentState) -> dict[str, Any]:
    """Data-reflection step that decides whether retrieved evidence is sufficient.

    Non-grounding intents (recall, list, propose) short-circuit to sufficient.
    For grounding intents, any RAG / KG / procedure hits count as sufficient.
    When evidence is thin and the replan budget is exhausted, the node forwards
    anyway and flags `evidence_insufficient` so the response can degrade
    gracefully. When budget remains, the LLM is asked for a structured verdict
    naming the gaps and a refined follow-up subquery.
    """
    routing = state.get("routing") or {}
    intent = routing.get("intent_label") or "fallback"
    plan = state.get("plan") or {}
    replan_count = int(plan.get("replan_count") or 0)

    if intent not in _GROUNDING_INTENTS:
        verdict = EvidenceReflection(sufficient=True, rationale="non-grounding intent")
        return {"reflection_eval": verdict.model_dump()}

    hits = (
        len(state.get("rag_hits") or [])
        + len(state.get("kg_hits") or [])
        + len(state.get("procedure_hits") or [])
    )
    if hits > 0:
        verdict = EvidenceReflection(sufficient=True, rationale="retrieval returned grounded hits")
        return {"reflection_eval": verdict.model_dump()}

    if replan_count >= MAX_REPLANS:
        verdict = EvidenceReflection(
            sufficient=True,
            missing=["grounded sources"],
            rationale="replan budget exhausted; forwarding with degraded flag",
        )
        return {
            "reflection_eval": verdict.model_dump(),
            "degraded": ["evidence_insufficient"],
        }

    user_text = _last_user_text(state["messages"])
    chat = get_chat_provider()
    settings = get_settings()
    prompt = REFLECTION_PROMPT.format(
        user_message=user_text,
        evidence_summary=_evidence_summary(state),
    )
    structured_failed = False
    try:
        verdict = invoke_typed_with_retry(
            chat, prompt, EvidenceReflection,
            max_attempts=settings.structured_retry_max_attempts,
        )
    except ValueError:
        structured_failed = True
        verdict = EvidenceReflection(
            sufficient=False,
            missing=["grounded sources"],
            followup_subquery=user_text,
            rationale="reflection LLM parse failed; defaulting to one rescue pass",
        )
    assert isinstance(verdict, EvidenceReflection)
    out: dict[str, Any] = {"reflection_eval": verdict.model_dump()}
    usage = getattr(chat, "last_usage", None)
    if usage:
        out["usage"] = usage_payload(usage["input_tokens"], usage["output_tokens"], settings)
    _record_structured_retry(chat, out, node="reflect_on_evidence")
    if structured_failed:
        _append_marker(out, "structured_failed:reflect_on_evidence")
    _record_chat_fallback(chat, out)
    return out


_SKIPPED_NOTE = "(not retrieved this turn — intent router skipped this branch)"


def _ctx_for(state: AgentState, branch: str, field: str) -> str:
    plan = state.get("plan") or {}
    branches = plan.get("branches") or (state.get("routing") or {}).get("branches")
    if branches is not None and branch not in branches:
        return _SKIPPED_NOTE
    return state.get(field, "")


def _resolve_stream_writer():
    """Return the active LangGraph custom-stream writer, or None when unavailable."""
    if _get_stream_writer is None:
        return None
    try:
        return _get_stream_writer()
    except Exception:
        return None


@timed("llm_ms")
def generate_response(state: AgentState) -> dict[str, Any]:
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(
            ltm_context=_ctx_for(state, "ltm", "ltm_context"),
            episodic_context=_ctx_for(state, "episodes", "episodic_context"),
            procedural_context=_ctx_for(state, "procedures", "procedural_context"),
            rag_context=_ctx_for(state, "rag", "rag_context"),
            kg_context=_ctx_for(state, "kg", "kg_context"),
        )
    )
    prompt: list[BaseMessage] = [system, *state["messages"]]
    writer = _resolve_stream_writer()
    llm = get_llm()
    stream_fn = getattr(llm, "stream", None)
    started = time.perf_counter()
    ttft_ms: float | None = None
    usage: dict[str, int] | None = None
    if stream_fn is None:
        reply = llm.invoke(prompt)
        body = reply.content if isinstance(reply.content, str) else str(reply.content)
        if body:
            ttft_ms = (time.perf_counter() - started) * 1000.0
            if writer is not None:
                writer({"node": "generate_response", "delta": body})
        usage = extract_usage_metadata(reply)
        out: dict[str, Any] = {"messages": [reply]}
        if ttft_ms is not None:
            out["latency_ms"] = {"llm_ttft_ms": ttft_ms}
        if usage is not None:
            out["usage"] = usage_payload(usage["input_tokens"], usage["output_tokens"], get_settings())
        return out
    parts: list[str] = []
    for chunk in stream_fn(prompt):
        delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        chunk_usage = extract_usage_metadata(chunk)
        if chunk_usage is not None:
            usage = chunk_usage
        if not delta:
            continue
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - started) * 1000.0
        parts.append(delta)
        if writer is not None:
            writer({"node": "generate_response", "delta": delta})
    reply = AIMessage(content="".join(parts))
    out = {"messages": [reply]}
    if ttft_ms is not None:
        out["latency_ms"] = {"llm_ttft_ms": ttft_ms}
    if usage is not None:
        out["usage"] = usage_payload(usage["input_tokens"], usage["output_tokens"], get_settings())
    return out


def _citable_tokens_from(state: AgentState) -> list[str]:
    """Collect candidate source strings from RAG hits and KG fact lines."""
    tokens: list[str] = []
    for hit in state.get("rag_hits") or []:
        src = hit.get("source") if isinstance(hit, dict) else None
        if isinstance(src, str) and src:
            tokens.append(src)
            base = src.rsplit("/", 1)[-1]
            if base and base != src:
                tokens.append(base)
    for hit in state.get("kg_hits") or []:
        fact = hit.get("fact") if isinstance(hit, dict) else None
        if not isinstance(fact, str):
            continue
        for chunk in fact.replace("[sources:", " ").replace("]", " ").split():
            stripped = chunk.strip(".,;:()[]")
            if "/" in stripped and "." in stripped:
                tokens.append(stripped)
                base = stripped.rsplit("/", 1)[-1]
                if base and base != stripped:
                    tokens.append(base)
    return tokens


@timed("validate_ms")
def validate_citations(state: AgentState) -> dict[str, Any]:
    """Flag replies that don't cite any retrieved RAG / KG source.

    Skipped when the reply is empty or when no groundable sources were
    retrieved this turn (recall- or policy-only answers, or a fully
    degraded retrieval phase). On violation, appends `"citations_missing"`
    to `state['degraded']` without blocking the rest of the graph.
    """
    reply_text = ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            reply_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if not reply_text:
        return {}
    citable = _citable_tokens_from(state)
    if not citable:
        return {}
    haystack = reply_text.lower()
    if any(tok.lower() in haystack for tok in citable):
        return {}
    return {"degraded": ["citations_missing"]}


_NO_ACTION: dict[str, Any] = {"action_type": "none"}


@timed("plan_ms")
def plan_action(state: AgentState) -> dict[str, Any]:
    """Extract a structured action proposal from the agent reply.

    Two action types share this node:
    - `propose_procedure` when the router classified the turn as a request to
      add a new operating rule (extracted via `PROCEDURE_PROPOSAL_PROMPT`).
    - `create_booking_draft` for shipment recommendations (the default path).
    """
    messages = state["messages"]
    user_text = _last_user_text(messages)
    agent_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            agent_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if not (user_text and agent_text):
        return {"action_plan": dict(_NO_ACTION)}
    chat = get_chat_provider()
    settings = get_settings()
    routing = state.get("routing") or {}
    structured_failed = False
    if routing.get("intent_label") == "propose_procedure":
        prompt = PROCEDURE_PROPOSAL_PROMPT.format(
            user_message=user_text, agent_message=agent_text
        )
        try:
            proposal = invoke_typed_with_retry(
                chat, prompt, ProcedureProposal,
                max_attempts=settings.structured_retry_max_attempts,
            )
        except ValueError:
            out: dict[str, Any] = {"action_plan": dict(_NO_ACTION)}
            structured_failed = True
        else:
            assert isinstance(proposal, ProcedureProposal)
            if proposal.action_type != "propose_procedure" or not proposal.rule.strip():
                out = {"action_plan": dict(_NO_ACTION)}
            else:
                out = {"action_plan": proposal.model_dump()}
    else:
        prompt = ACTION_PLANNING_PROMPT.format(
            user_message=user_text, agent_message=agent_text
        )
        try:
            booking = invoke_typed_with_retry(
                chat, prompt, BookingProposal,
                max_attempts=settings.structured_retry_max_attempts,
            )
        except ValueError:
            out = {"action_plan": dict(_NO_ACTION)}
            structured_failed = True
        else:
            assert isinstance(booking, BookingProposal)
            if booking.requires_approval is False and "[REQUIRES HUMAN APPROVAL]" in agent_text:
                booking = booking.model_copy(update={"requires_approval": True})
            out = {"action_plan": booking.model_dump()}
    usage = getattr(chat, "last_usage", None)
    if usage:
        out["usage"] = usage_payload(usage["input_tokens"], usage["output_tokens"], settings)
    _record_structured_retry(chat, out, node="plan_action")
    if structured_failed:
        _append_marker(out, "structured_failed:plan_action")
    _record_chat_fallback(chat, out)
    return out


def _draft_id_for(ctx, plan: dict[str, Any]) -> str:
    """Derive a deterministic draft_id so the node is idempotent across interrupt/resume."""
    seed = "|".join(
        str(part)
        for part in (
            getattr(ctx, "correlation_id", None) or "",
            getattr(ctx, "realm_id", "") or "",
            getattr(ctx, "user_id", "") or "",
            plan.get("carrier") or "",
            plan.get("lane") or "",
            plan.get("weight_lb") or "",
            plan.get("estimated_cost_usd") or "",
        )
    )
    return "draft_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _execute_procedure_proposal(state: AgentState, plan: dict[str, Any]) -> dict[str, Any]:
    """Stage a procedural-rule proposal, interrupt for approval, and commit on yes.

    The agent never writes directly to `agent_procedures`. Proposals land in
    the `procedure_proposals` staging collection; `commit()` is only called
    after a human approval resumes the graph from `interrupt()`.
    """
    ctx = state["context"]
    proc_mem = get_procedural_memory()
    proposal_doc = proc_mem.propose(
        ctx.realm_id,
        ctx.agent_id,
        rule=plan.get("rule") or "",
        category=plan.get("category") or "general",
        rationale=plan.get("rationale") or "",
        proposed_by=getattr(ctx, "user_id", None),
        correlation_id=getattr(ctx, "correlation_id", None),
    )
    proposal_id = proposal_doc["proposal_id"]
    decision = interrupt(
        {
            "kind": "procedure_proposal",
            "proposal_id": proposal_id,
            "rule_id": proposal_doc["rule_id"],
            "rule": proposal_doc["rule"],
            "category": proposal_doc.get("category", "general"),
            "rationale": proposal_doc.get("rationale", ""),
        }
    )
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        approver = decision.get("approver")
    else:
        approved = bool(decision)
        approver = None
    if approved:
        proc_mem.commit(ctx.realm_id, ctx.agent_id, proposal_id, approver=approver)
        status = "approved"
    else:
        proc_mem.reject(ctx.realm_id, ctx.agent_id, proposal_id, approver=approver)
        status = "rejected"
    return {
        "procedure_proposal": {
            "proposal_id": proposal_id,
            "rule_id": proposal_doc["rule_id"],
            "rule": proposal_doc["rule"],
            "category": proposal_doc.get("category", "general"),
            "rationale": proposal_doc.get("rationale", ""),
            "status": status,
            "approver": approver,
        }
    }


@timed("execute_ms")
def execute_action(state: AgentState) -> dict[str, Any]:
    """Persist a booking draft (or procedural-rule proposal) and gate on approval."""
    plan = state.get("action_plan") or {}
    action_type = plan.get("action_type")
    if action_type == "propose_procedure":
        return _execute_procedure_proposal(state, plan)
    if action_type != "create_booking_draft":
        return {}
    ctx = state["context"]
    drafts = get_booking_drafts_collection()
    draft_id = _draft_id_for(ctx, plan)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    requires_approval = bool(plan.get("requires_approval"))
    base: dict[str, Any] = {
        "draft_id": draft_id,
        "realm_id": ctx.realm_id,
        "user_id": ctx.user_id,
        "agent_id": ctx.agent_id,
        "correlation_id": getattr(ctx, "correlation_id", None),
        "carrier": plan.get("carrier"),
        "lane": plan.get("lane"),
        "origin": plan.get("origin"),
        "destination": plan.get("destination"),
        "weight_lb": plan.get("weight_lb"),
        "estimated_cost_usd": plan.get("estimated_cost_usd"),
        "rationale": plan.get("rationale", ""),
        "updated_at": now,
    }
    drafts.update_one(
        {"draft_id": draft_id},
        {
            "$set": {**base, "status": "pending_approval" if requires_approval else "approved"},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    if requires_approval:
        decision = interrupt(
            {
                "kind": "approval_request",
                "draft_id": draft_id,
                "carrier": base["carrier"],
                "lane": base["lane"],
                "weight_lb": base["weight_lb"],
                "estimated_cost_usd": base["estimated_cost_usd"],
                "rationale": base["rationale"],
            }
        )
        if isinstance(decision, dict):
            approved = bool(decision.get("approved"))
            approver = decision.get("approver")
        else:
            approved = bool(decision)
            approver = None
        new_status = "executed" if approved else "rejected"
        update_fields: dict[str, Any] = {
            "status": new_status,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if approver:
            update_fields["approver"] = approver
        drafts.update_one({"draft_id": draft_id}, {"$set": update_fields})
        base["status"] = new_status
        if approver:
            base["approver"] = approver
    else:
        drafts.update_one(
            {"draft_id": draft_id},
            {"$set": {"status": "executed", "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}},
        )
        base["status"] = "executed"
    return {"booking_draft": base}


@timed("save_ms")
def save_memory(state: AgentState) -> dict[str, Any]:
    messages = state["messages"]
    user_text = _last_user_text(messages)
    agent_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            agent_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if not (user_text and agent_text):
        return {}

    ctx = state["context"]
    chat = get_chat_provider()
    settings = get_settings()
    prompt = MEMORY_EXTRACTION_PROMPT.format(user_message=user_text, agent_message=agent_text)
    try:
        extraction = invoke_typed_with_retry(
            chat, prompt, MemoryExtraction,
            max_attempts=settings.structured_retry_max_attempts,
        )
    except ValueError:
        out: dict[str, Any] = {}
        _record_structured_retry(chat, out, node="save_memory")
        _append_marker(out, "structured_failed:save_memory")
        _record_chat_fallback(chat, out)
        return out
    assert isinstance(extraction, MemoryExtraction)
    usage = getattr(chat, "last_usage", None)

    semantic_mem = get_semantic_memory()
    for fact in extraction.facts:
        line = fact.strip()
        if not line:
            continue
        digest = hashlib.sha1(line.encode("utf-8")).hexdigest()[:16]
        semantic_mem.put(ctx.realm_id, ctx.user_id, key=f"mem_{digest}", content=line)

    episode = extraction.episode
    if episode is not None and episode.summary.strip():
        record: dict[str, str] = {"summary": episode.summary.strip()}
        for field in ("lane", "recommendation", "outcome"):
            value = getattr(episode, field)
            if value:
                record[field] = value.strip()
        record["occurred_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        digest = hashlib.sha1(record["summary"].encode("utf-8")).hexdigest()[:16]
        get_episodic_memory().put(ctx.realm_id, ctx.user_id, key=f"ep_{digest}", episode=record)

    out = {}
    if usage:
        out["usage"] = usage_payload(usage["input_tokens"], usage["output_tokens"], settings)
    _record_structured_retry(chat, out, node="save_memory")
    _record_chat_fallback(chat, out)

    every = settings.reflect_every_n_turns
    if every > 0:
        key = (ctx.realm_id, ctx.user_id)
        count = _TURN_COUNTERS.get(key, 0) + 1
        _TURN_COUNTERS[key] = count
        if count % every == 0:
            try:
                out["reflection"] = _run_reflection_pass(
                    ctx.realm_id, ctx.user_id, settings.reflect_threshold
                )
            except Exception:
                markers = list(out.get("degraded") or [])
                if "reflection_failed" not in markers:
                    markers.append("reflection_failed")
                out["degraded"] = markers
    return out


__all__ = [
    "AgentState",
    "classify_intent",
    "retrieve_ltm",
    "retrieve_rag",
    "retrieve_episodes",
    "retrieve_procedures",
    "retrieve_kg",
    "generate_response",
    "validate_citations",
    "plan_action",
    "execute_action",
    "save_memory",
    "EMBEDDING_DIMS",
]
