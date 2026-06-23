"""Unit tests for the four retriever nodes.

Nodes are exercised against in-memory fakes so the suite runs without
Atlas, Voyage, or Grove credentials.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.nodes import (
    execute_action,
    generate_response,
    plan_action,
    reflect_on_evidence,
    retrieve_episodes,
    retrieve_ltm,
    retrieve_procedures,
    retrieve_rag,
    save_memory,
    think_and_plan,
    validate_citations,
)
from tests.fakes import (
    FakeChatProvider,
    FakeEpisodicMemory,
    FakeKnowledgeRetriever,
    FakeProceduralMemory,
    FakeSemanticMemory,
)


def _state(context, text: str = "hello"):
    return {"messages": [HumanMessage(content=text)], "context": context}


def test_retrieve_ltm_empty(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_semantic_memory", lambda: FakeSemanticMemory([]))
    out = retrieve_ltm(_state(context))
    assert out["ltm_hits"] == []
    assert "no prior memories" in out["ltm_context"]
    assert "ltm_ms" in out["latency_ms"]


def test_retrieve_ltm_hits(monkeypatch, context):
    facts = [{"key": "m1", "content": "User prefers Carrier A.", "score": 0.91}]
    monkeypatch.setattr(nodes, "get_semantic_memory", lambda: FakeSemanticMemory(facts))
    out = retrieve_ltm(_state(context, "which carrier?"))
    assert len(out["ltm_hits"]) == 1
    assert out["ltm_hits"][0]["content"] == "User prefers Carrier A."
    assert "User prefers Carrier A." in out["ltm_context"]


def test_retrieve_episodes_hits(monkeypatch, context):
    episodes = [{
        "key": "ep1",
        "summary": "shipped El Paso to Phoenix",
        "lane": "TX-AZ",
        "recommendation": "Carrier A",
        "outcome": "booked",
        "occurred_at": "2026-04-08T14:22:00+00:00",
        "score": 0.82,
    }]
    monkeypatch.setattr(nodes, "get_episodic_memory", lambda: FakeEpisodicMemory(episodes))
    out = retrieve_episodes(_state(context, "El Paso shipment"))
    assert len(out["episode_hits"]) == 1
    assert out["episode_hits"][0]["lane"] == "TX-AZ"
    assert out["episode_hits"][0]["recommendation"] == "Carrier A"
    assert "shipped El Paso to Phoenix" in out["episodic_context"]


def test_retrieve_procedures_filters_by_realm_and_agent(monkeypatch, context):
    docs = [
        {"realm_id": "realm-test", "agent_id": "agent-test", "active": True,
         "rule_id": "p1", "rule": "always emit lb + kg", "category": "units"},
        {"realm_id": "realm-other", "agent_id": "agent-test", "active": True,
         "rule_id": "p2", "rule": "should not match", "category": "units"},
    ]
    monkeypatch.setattr(nodes, "get_procedural_memory", lambda: FakeProceduralMemory(docs))
    out = retrieve_procedures(_state(context))
    assert len(out["procedure_hits"]) == 1
    assert out["procedure_hits"][0]["rule_id"] == "p1"
    assert "always emit lb + kg" in out["procedural_context"]


def test_retrieve_procedures_empty(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_procedural_memory", lambda: FakeProceduralMemory([]))
    out = retrieve_procedures(_state(context))
    assert out["procedure_hits"] == []
    assert "no procedural rules" in out["procedural_context"]


def test_retrieve_rag(monkeypatch, context):
    hits = [{
        "doc_type": "route_guide",
        "source": "tx_az_lane.pdf",
        "text": "Carrier A preferred for TX-AZ.",
        "score": 0.77,
        "metadata": {},
    }]
    monkeypatch.setattr(nodes, "get_knowledge_retriever", lambda: FakeKnowledgeRetriever(hits))
    out = retrieve_rag(_state(context, "shipping question?"))
    assert len(out["rag_hits"]) == 1
    assert "Carrier A preferred for TX-AZ." in out["rag_context"]
    assert "rag_ms" in out["latency_ms"]


def test_retrieve_ltm_no_query_returns_empty(monkeypatch, context):
    facts = [{"key": "x", "content": "y"}]
    monkeypatch.setattr(nodes, "get_semantic_memory", lambda: FakeSemanticMemory(facts))
    # No HumanMessage → _last_user_text returns "" → no search
    state = {"messages": [], "context": context}
    out = retrieve_ltm(state)
    assert out["ltm_hits"] == []


def _boom(*_args, **_kwargs):
    raise RuntimeError("backend unavailable")


def test_retrieve_ltm_degrades_on_backend_failure(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_semantic_memory", _boom)
    out = retrieve_ltm(_state(context, "anything"))
    assert out["ltm_hits"] == []
    assert "retrieval degraded" in out["ltm_context"]
    assert any("retrieve_ltm" in m for m in out.get("degraded", []))
    assert "ltm_ms" in out["latency_ms"]


def test_retrieve_rag_degrades_on_retriever_failure(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_knowledge_retriever", _boom)
    out = retrieve_rag(_state(context, "anything"))
    assert out["rag_hits"] == []
    assert "retrieval degraded" in out["rag_context"]
    assert any("retrieve_rag" in m for m in out.get("degraded", []))


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text


class _StreamingLLM:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    def stream(self, _prompt):
        for d in self._deltas:
            yield _Chunk(d)


class _NonStreamingLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def invoke(self, _prompt):
        return AIMessage(content=self._reply)


def _llm_state(context):
    return {"messages": [HumanMessage(content="hello")], "context": context}


def test_generate_response_records_ttft_on_streaming_path(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_llm", lambda: _StreamingLLM(["", "Hello ", "world."]))
    out = generate_response(_llm_state(context))
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == "Hello world."
    latency = out["latency_ms"]
    assert "llm_ttft_ms" in latency
    assert latency["llm_ttft_ms"] >= 0.0


def test_generate_response_records_ttft_on_non_streaming_path(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_llm", lambda: _NonStreamingLLM("ok"))
    out = generate_response(_llm_state(context))
    assert out["messages"][0].content == "ok"
    assert "llm_ttft_ms" in out["latency_ms"]


def test_generate_response_skips_ttft_when_reply_is_empty(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_llm", lambda: _NonStreamingLLM(""))
    out = generate_response(_llm_state(context))
    assert out["messages"][0].content == ""
    assert "llm_ttft_ms" not in out.get("latency_ms", {})


class _CapturingLLM:
    """Non-streaming LLM that records the last prompt it received."""

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return AIMessage(content=self._reply)


def _system_text_from(llm: _CapturingLLM) -> str:
    from langchain_core.messages import SystemMessage

    for msg in llm.last_prompt or []:
        if isinstance(msg, SystemMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def test_build_system_prompt_includes_only_supplied_branches():
    from agent.prompts import SYSTEM_PROMPT_BASE, build_system_prompt

    text = build_system_prompt({"rag": "chunk-text"})
    assert text.startswith(SYSTEM_PROMPT_BASE)
    assert "Retrieved knowledge corpus chunks" in text
    assert "chunk-text" in text
    for header in (
        "Semantic memory",
        "Episodic memory",
        "Procedural memory",
        "Structured facts",
    ):
        assert header not in text


def test_build_system_prompt_drops_empty_and_whitespace_branches():
    from agent.prompts import build_system_prompt

    text = build_system_prompt({"rag": "  ", "kg": "fact-line"})
    assert "Retrieved knowledge corpus chunks" not in text
    assert "Structured facts" in text
    assert "fact-line" in text


def test_build_system_prompt_orders_sections_canonically():
    from agent.prompts import build_system_prompt

    text = build_system_prompt({
        "kg": "k", "rag": "r", "procedures": "p", "episodes": "e", "ltm": "l",
    })
    order = [
        text.index("Semantic memory"),
        text.index("Episodic memory"),
        text.index("Procedural memory"),
        text.index("Retrieved knowledge corpus chunks"),
        text.index("Structured facts"),
    ]
    assert order == sorted(order)


def test_build_system_prompt_returns_base_only_when_no_sections():
    from agent.prompts import SYSTEM_PROMPT_BASE, build_system_prompt

    assert build_system_prompt({}) == SYSTEM_PROMPT_BASE
    assert build_system_prompt({"ltm": "", "rag": "   "}) == SYSTEM_PROMPT_BASE


def test_branch_contexts_filters_by_router_branches(context):
    state = {
        "messages": [HumanMessage(content="q")],
        "context": context,
        "routing": {"intent_label": "recall_history", "branches": ["ltm", "episodes"]},
        "ltm_context": "L",
        "episodic_context": "E",
        "procedural_context": "P",
        "rag_context": "R",
        "kg_context": "K",
    }
    out = nodes._branch_contexts(state)
    assert set(out) == {"ltm", "episodes"}


def test_branch_contexts_prefers_plan_branches_over_routing(context):
    state = {
        "messages": [HumanMessage(content="q")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": list(nodes.ALL_BRANCHES)},
        "plan": {"branches": ["rag", "kg"], "replan_count": 1},
        "ltm_context": "L",
        "rag_context": "R",
        "kg_context": "K",
    }
    out = nodes._branch_contexts(state)
    assert set(out) == {"rag", "kg"}


def test_branch_contexts_drops_empty_payloads_even_when_branch_selected(context):
    state = {
        "messages": [HumanMessage(content="q")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": list(nodes.ALL_BRANCHES)},
        "rag_context": "chunk",
        "kg_context": "   ",
        "ltm_context": "",
    }
    out = nodes._branch_contexts(state)
    assert out == {"rag": "chunk"}


def test_branch_contexts_returns_all_populated_when_no_routing(context):
    state = {
        "messages": [HumanMessage(content="q")],
        "context": context,
        "rag_context": "R",
        "ltm_context": "L",
    }
    out = nodes._branch_contexts(state)
    assert set(out) == {"ltm", "rag"}


def test_generate_response_omits_skipped_branch_sections(monkeypatch, context):
    llm = _CapturingLLM("ok")
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    state = {
        "messages": [HumanMessage(content="who do we prefer for TX-AZ?")],
        "context": context,
        "routing": {"intent_label": "recall_history", "branches": ["ltm"]},
        "ltm_context": "User prefers Carrier A for TX-AZ.",
        "rag_context": "should-not-leak-into-prompt",
        "kg_context": "should-not-leak-either",
    }
    generate_response(state)
    sys_text = _system_text_from(llm)
    assert "Semantic memory" in sys_text
    assert "User prefers Carrier A for TX-AZ." in sys_text
    assert "Retrieved knowledge corpus chunks" not in sys_text
    assert "Structured facts" not in sys_text
    assert "should-not-leak-into-prompt" not in sys_text
    assert "should-not-leak-either" not in sys_text


def _citation_state(reply: str, *, rag=None, kg=None, context=None):
    state = {
        "messages": [HumanMessage(content="q"), AIMessage(content=reply)],
        "rag_hits": rag or [],
        "kg_hits": kg or [],
    }
    if context is not None:
        state["context"] = context
    return state


def test_validate_citations_passes_when_reply_cites_rag_basename():
    state = _citation_state(
        "Per carrier_a_2026.pdf the TX-AZ surcharge applies above 22000 lb.",
        rag=[{"source": "carrier_agreements/carrier_a_2026.pdf", "text": "..."}],
    )
    assert "degraded" not in validate_citations(state)


def test_validate_citations_passes_when_reply_cites_kg_source_doc():
    fact = (
        "- Carrier A serves lane TX-AZ (Austin->Phoenix); priority=1 "
        "[sources: route_guides/tx_az_lane.pdf, carrier_agreements/carrier_a_2026.pdf]"
    )
    state = _citation_state(
        "See route_guides/tx_az_lane.pdf for the lane policy.",
        kg=[{"fact": fact}],
    )
    assert "degraded" not in validate_citations(state)


def test_validate_citations_flags_missing_citation():
    state = _citation_state(
        "Carrier A is your best option for TX-AZ.",
        rag=[{"source": "carrier_agreements/carrier_a_2026.pdf", "text": "..."}],
    )
    out = validate_citations(state)
    assert out.get("degraded") == ["citations_missing"]


def test_validate_citations_skips_when_no_groundable_sources():
    state = _citation_state("You preferred Carrier A last time.", rag=[], kg=[])
    assert "degraded" not in validate_citations(state)


def test_validate_citations_skips_when_reply_is_empty():
    state = _citation_state(
        "",
        rag=[{"source": "carrier_agreements/carrier_a_2026.pdf"}],
    )
    assert "degraded" not in validate_citations(state)


def test_validate_citations_attaches_citation_spans_from_rag():
    reply = (
        "Carrier A serves the Austin to Dallas same-day lane with no fuel surcharge "
        "under 20,000 lb. Typical 15,000 lb dry-van quotes price between $410 and $475 all-in."
    )
    rag = [
        {
            "doc_type": "route_guide",
            "source": "route_guides/austin_dallas_hot_lane.pdf",
            "text": (
                "Primary carrier: Carrier A dry van with no fuel surcharge under "
                "the 20,000 lb TX-TX threshold. Typical 15,000 lb dry-van quotes "
                "price between $410 and $475 all-in."
            ),
            "score": 0.71,
        },
    ]
    out = validate_citations(_citation_state(reply, rag=rag))
    spans = out.get("citations")
    assert isinstance(spans, list) and len(spans) >= 1
    # Both sentences share enough tokens with the chunk to bind to it.
    assert all(s["source"] == "route_guides/austin_dallas_hot_lane.pdf" for s in spans)
    assert all(s["kind"] == "rag" for s in spans)
    assert all(s["overlap"] >= 2 for s in spans)
    # Offsets are valid and ordered.
    assert all(0 <= s["start"] < s["end"] <= len(reply) for s in spans)


def test_validate_citations_no_citations_when_overlap_below_floor():
    reply = "Hello there."
    rag = [{"source": "x.pdf", "doc_type": "policy", "text": "Carrier A dry van TX-TX lane.", "score": 0.5}]
    out = validate_citations(_citation_state(reply, rag=rag))
    assert "citations" not in out


def _save_memory_state(context, agent_reply: str = "ok"):
    return {
        "messages": [HumanMessage(content="how do I ship TX-AZ?"), AIMessage(content=agent_reply)],
        "context": context,
    }


def _stub_save_memory_deps(monkeypatch, *, every: int):
    """Wire fakes so `save_memory` runs without Atlas / Voyage / Grove."""
    from core.settings import Settings

    extraction_json = '{"facts": ["User prefers Carrier A."], "episode": null}'
    chat = FakeChatProvider(reply=extraction_json)
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    monkeypatch.setattr(nodes, "get_semantic_memory", lambda: FakeSemanticMemory([]))
    monkeypatch.setattr(nodes, "get_episodic_memory", lambda: FakeEpisodicMemory([]))
    settings = Settings(
        mongodb_uri="mongodb://localhost:27017/test",
        reflect_every_n_turns=every,
    )
    monkeypatch.setattr(nodes, "get_settings", lambda: settings)


def test_save_memory_skips_reflection_when_disabled(monkeypatch, context):
    nodes._reset_turn_counters()
    _stub_save_memory_deps(monkeypatch, every=0)
    calls: list[tuple] = []
    monkeypatch.setattr(
        nodes, "_run_reflection_pass",
        lambda r, u, t: calls.append((r, u, t)) or {"clusters_found": 0},
    )
    out = save_memory(_save_memory_state(context))
    assert "reflection" not in out
    assert calls == []


def test_save_memory_triggers_reflection_every_n_turns(monkeypatch, context):
    nodes._reset_turn_counters()
    _stub_save_memory_deps(monkeypatch, every=2)
    calls: list[tuple] = []
    monkeypatch.setattr(
        nodes, "_run_reflection_pass",
        lambda r, u, t: calls.append((r, u, t)) or {
            "clusters_found": 1, "canonical_written": 1, "tombstoned": 2,
        },
    )

    first = save_memory(_save_memory_state(context))
    assert "reflection" not in first
    assert calls == []

    second = save_memory(_save_memory_state(context))
    assert second["reflection"] == {
        "clusters_found": 1, "canonical_written": 1, "tombstoned": 2,
    }
    assert calls == [(context.realm_id, context.user_id, 0.88)]

    third = save_memory(_save_memory_state(context))
    assert "reflection" not in third
    assert len(calls) == 1


def test_save_memory_degrades_when_reflection_raises(monkeypatch, context):
    nodes._reset_turn_counters()
    _stub_save_memory_deps(monkeypatch, every=1)

    def _boom_reflection(*_args, **_kwargs):
        raise RuntimeError("atlas down")

    monkeypatch.setattr(nodes, "_run_reflection_pass", _boom_reflection)
    out = save_memory(_save_memory_state(context))
    assert "reflection" not in out
    assert out.get("degraded") == ["reflection_failed"]


def _propose_state(context, *, user="going forward, always prefer Carrier A on TX-AZ",
                   agent="Got it — I will propose this rule for approval."):
    return {
        "messages": [HumanMessage(content=user), AIMessage(content=agent)],
        "context": context,
        "routing": {"intent_label": "propose_procedure", "branches": ["procedures"]},
    }


def test_plan_action_uses_procedure_prompt_when_routing_proposes(monkeypatch, context):
    chat = FakeChatProvider(
        reply='{"action_type": "propose_procedure", "rule": "Always prefer Carrier A on TX-AZ.", "category": "policy", "rationale": "user preference"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = plan_action(_propose_state(context))
    assert out["action_plan"]["action_type"] == "propose_procedure"
    assert out["action_plan"]["rule"] == "Always prefer Carrier A on TX-AZ."
    assert out["action_plan"]["category"] == "policy"
    assert "User: going forward" in chat.calls[0]
    assert "PROCEDURE EXTRACTOR" in chat.calls[0]


def test_plan_action_drops_empty_procedure_proposal(monkeypatch, context):
    chat = FakeChatProvider(reply='{"action_type": "propose_procedure", "rule": "", "category": "general", "rationale": ""}')
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = plan_action(_propose_state(context))
    assert out["action_plan"] == {"action_type": "none"}


def test_plan_action_keeps_booking_path_when_intent_not_propose(monkeypatch, context):
    chat = FakeChatProvider(
        reply='{"action_type": "create_booking_draft", "carrier": "A", "lane": "TX-AZ", "weight_lb": 1000, "estimated_cost_usd": 500, "requires_approval": false, "rationale": "r"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="ship 1000 lb to Phoenix"), AIMessage(content="Carrier A.")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": list(nodes.ALL_BRANCHES)},
    }
    out = plan_action(state)
    assert out["action_plan"]["action_type"] == "create_booking_draft"
    assert out["action_plan"]["carrier"] == "A"
    assert "ACTION PLANNER" in chat.calls[0]


def _booking_state(context, *, agent_text: str, rag_context: str = ""):
    return {
        "messages": [HumanMessage(content="ship 15,000 lb Austin to Dallas"), AIMessage(content=agent_text)],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": list(nodes.ALL_BRANCHES)},
        "rag_context": rag_context,
    }


def test_plan_action_cost_fallback_takes_range_upper_bound(monkeypatch, context):
    """When the LLM omits estimated_cost_usd, the regex fallback picks the
    upper bound of the first '$X-$Y' range found in the agent reply.
    """
    chat = FakeChatProvider(
        reply='{"action_type": "create_booking_draft", "carrier": "A", "lane": "Austin-Dallas", "weight_lb": 15000, "estimated_cost_usd": null, "requires_approval": false, "rationale": "r"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    agent_text = "Typical 15,000 lb dry-van quote is $410–$475 all-in. $475 < $10,000."
    out = plan_action(_booking_state(context, agent_text=agent_text))
    assert out["action_plan"]["estimated_cost_usd"] == 475.0
    assert out["action_plan"]["requires_approval"] is False
    assert "cost_extracted_via_fallback" in out.get("degraded", [])


def test_plan_action_cost_fallback_does_not_override_llm_value(monkeypatch, context):
    """A planner-supplied cost must not be touched by the regex fallback,
    even when the agent reply also contains a different '$X' literal.
    """
    chat = FakeChatProvider(
        reply='{"action_type": "create_booking_draft", "carrier": "A", "lane": "Austin-Dallas", "weight_lb": 15000, "estimated_cost_usd": 500, "requires_approval": false, "rationale": "r"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = plan_action(_booking_state(context, agent_text="Quote $410-$475 typical."))
    assert out["action_plan"]["estimated_cost_usd"] == 500
    assert "cost_extracted_via_fallback" not in out.get("degraded", [])


def test_plan_action_flips_approval_when_fallback_cost_above_threshold(monkeypatch, context):
    """When the regex fallback yields a cost above $10k, requires_approval
    must flip True even though the LLM returned False.
    """
    chat = FakeChatProvider(
        reply='{"action_type": "create_booking_draft", "carrier": "A", "lane": "TX-CA", "weight_lb": 30000, "estimated_cost_usd": null, "requires_approval": false, "rationale": "r"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = plan_action(_booking_state(context, agent_text="Estimated all-in $12,500 for this lane."))
    assert out["action_plan"]["estimated_cost_usd"] == 12500.0
    assert out["action_plan"]["requires_approval"] is True


def test_extract_cost_fallback_helper_variants():
    """Direct unit test for the cost-extraction regex across common phrasings."""
    f = nodes._extract_cost_fallback
    assert f("Typical quote is $410-$475 all-in.") == 475.0
    assert f("Typical quote is $410–$475 all-in.") == 475.0  # en-dash
    assert f("Range: $410 to $475.") == 475.0
    assert f("Quotes price between $410 and $475 all-in.") == 475.0
    assert f("Estimated total max $475.") == 475.0
    assert f("Estimated all-in $1,500.50.") == 1500.50
    assert f("No price mentioned here.") is None
    assert f("") is None
    # Skip $0 (surcharge-mention) false positives — a real booking cost is >0.
    assert f("Carrier A has $0 fuel surcharge. Typical quote $475.") == 475.0
    assert f("$0 surcharge under threshold.") is None


def test_plan_action_cost_fallback_fires_when_llm_returns_zero(monkeypatch, context):
    """The action planner sometimes returns estimated_cost_usd=0 (a meaningless
    value for a booking); the fallback must engage just like for None.
    """
    chat = FakeChatProvider(
        reply='{"action_type": "create_booking_draft", "carrier": "A", "lane": "Austin-Dallas", "weight_lb": 15000, "estimated_cost_usd": 0, "requires_approval": false, "rationale": "r"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    agent_text = "Typical 15,000 lb dry-van quote $410-$475 all-in."
    out = plan_action(_booking_state(context, agent_text=agent_text))
    assert out["action_plan"]["estimated_cost_usd"] == 475.0
    assert "cost_extracted_via_fallback" in out.get("degraded", [])


def test_plan_action_cost_fallback_uses_rag_context_when_agent_text_lacks_price(monkeypatch, context):
    """When the reviewer strips the cost line from the visible reply, the
    fallback scans the retrieved RAG context (the source of truth) so the
    booking_draft still captures the right estimated_cost_usd.
    """
    chat = FakeChatProvider(
        reply='{"action_type": "create_booking_draft", "carrier": "A", "lane": "Austin-Dallas", "weight_lb": 15000, "estimated_cost_usd": null, "requires_approval": false, "rationale": "r"}'
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    agent_text = "Book Carrier A dry van. 99.1% on-time. 0.0 surcharge under 20,000 lb."
    rag_context = "[austin_dallas_hot_lane.pdf] Typical 15,000 lb dry-van all-in quote is $410-$475."
    out = plan_action(_booking_state(context, agent_text=agent_text, rag_context=rag_context))
    assert out["action_plan"]["estimated_cost_usd"] == 475.0
    assert "cost_extracted_via_fallback" in out.get("degraded", [])


def _propose_plan(rule="Always prefer Carrier A on TX-AZ.", category="policy", rationale="user pref"):
    return {
        "action_type": "propose_procedure",
        "rule": rule,
        "category": category,
        "rationale": rationale,
    }


def test_execute_action_propose_procedure_commits_on_approval(monkeypatch, context):
    proc_mem = FakeProceduralMemory()
    monkeypatch.setattr(nodes, "get_procedural_memory", lambda: proc_mem)
    monkeypatch.setattr(nodes, "interrupt", lambda _payload: {"approved": True, "approver": "alice"})
    state = {"context": context, "action_plan": _propose_plan()}
    out = execute_action(state)
    proposal = out["procedure_proposal"]
    assert proposal["status"] == "approved"
    assert proposal["approver"] == "alice"
    assert proposal["rule"] == "Always prefer Carrier A on TX-AZ."
    assert proposal["category"] == "policy"
    assert len(proc_mem.committed) == 1
    assert proc_mem.committed[0]["active"] is True
    assert proc_mem.committed[0]["realm_id"] == context.realm_id
    assert proc_mem.rejected == []


def test_execute_action_propose_procedure_rejects(monkeypatch, context):
    proc_mem = FakeProceduralMemory()
    monkeypatch.setattr(nodes, "get_procedural_memory", lambda: proc_mem)
    monkeypatch.setattr(nodes, "interrupt", lambda _payload: {"approved": False, "approver": "bob"})
    state = {"context": context, "action_plan": _propose_plan()}
    out = execute_action(state)
    assert out["procedure_proposal"]["status"] == "rejected"
    assert proc_mem.committed == []
    assert len(proc_mem.rejected) == 1
    assert proc_mem.rejected[0]["approver"] == "bob"


def test_execute_action_propose_procedure_interrupt_payload_shape(monkeypatch, context):
    proc_mem = FakeProceduralMemory()
    captured: dict = {}

    def _stub_interrupt(payload):
        captured.update(payload)
        return {"approved": True, "approver": "ops"}

    monkeypatch.setattr(nodes, "get_procedural_memory", lambda: proc_mem)
    monkeypatch.setattr(nodes, "interrupt", _stub_interrupt)
    execute_action({"context": context, "action_plan": _propose_plan()})
    assert captured["kind"] == "procedure_proposal"
    assert captured["rule"] == "Always prefer Carrier A on TX-AZ."
    assert captured["category"] == "policy"
    assert captured["proposal_id"].startswith("proc_")


def test_execute_action_noop_when_action_type_none(monkeypatch, context):
    monkeypatch.setattr(nodes, "interrupt", lambda _p: (_ for _ in ()).throw(AssertionError("no interrupt")))
    out = execute_action({"context": context, "action_plan": {"action_type": "none"}})
    assert "procedure_proposal" not in out
    assert "booking_draft" not in out


def test_merge_degraded_concatenates_within_turn():
    from agent.nodes import _merge_degraded
    assert _merge_degraded(["retrieve_ltm"], ["retrieve_rag"]) == ["retrieve_ltm", "retrieve_rag"]


def test_merge_degraded_dedupes_same_marker_within_turn():
    from agent.nodes import _merge_degraded
    assert _merge_degraded(["citations_missing"], ["citations_missing"]) == ["citations_missing"]


def test_merge_degraded_reset_sentinel_clears_accumulated_markers():
    from agent.nodes import _DEGRADED_RESET, _merge_degraded
    prior = ["citations_missing", "retrieve_ltm"]
    assert _merge_degraded(prior, [_DEGRADED_RESET]) == []
    assert _merge_degraded(prior, [_DEGRADED_RESET, "citations_missing"]) == ["citations_missing"]


def test_merge_degraded_handles_none_inputs():
    from agent.nodes import _merge_degraded
    assert _merge_degraded(None, None) == []
    assert _merge_degraded(None, ["x"]) == ["x"]
    assert _merge_degraded(["x"], None) == ["x"]


def _plan_state(context, text: str = "ship 5000 lbs TX to AZ", **extra):
    state = {
        "messages": [HumanMessage(content=text)],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": list(["ltm", "episodes", "procedures", "rag", "kg"])},
    }
    state.update(extra)
    return state


def test_think_and_plan_first_pass_mirrors_router(context):
    out = think_and_plan(_plan_state(context))
    plan = out["plan"]
    assert plan["branches"] == ["ltm", "episodes", "procedures", "rag", "kg"]
    assert plan["subquery"] is None
    assert plan["replan_count"] == 0


def test_think_and_plan_replan_narrows_and_refines(context):
    state = _plan_state(
        context,
        plan={"branches": ["ltm", "episodes", "procedures", "rag", "kg"], "subquery": None, "replan_count": 0},
        reflection_eval={"sufficient": False, "followup_subquery": "TX-AZ surcharge over 5000 lbs"},
    )
    out = think_and_plan(state)
    plan = out["plan"]
    assert plan["replan_count"] == 1
    assert plan["subquery"] == "TX-AZ surcharge over 5000 lbs"
    assert set(plan["branches"]).issubset({"rag", "kg", "procedures"})


def test_think_and_plan_does_not_replan_when_budget_exhausted(context):
    state = _plan_state(
        context,
        plan={"branches": ["rag", "kg"], "subquery": "prior", "replan_count": 1},
        reflection_eval={"sufficient": False, "followup_subquery": "second try"},
    )
    out = think_and_plan(state)
    plan = out["plan"]
    assert plan["replan_count"] == 0
    assert plan["subquery"] is None


def test_query_for_prefers_plan_subquery(context):
    from agent.nodes import _query_for
    state = _plan_state(context, text="raw user text", plan={"subquery": "refined supply chain terms"})
    assert _query_for(state) == "refined supply chain terms"
    assert _query_for(_plan_state(context, text="raw user text")) == "raw user text"


def test_reflect_on_evidence_non_grounding_intent_is_sufficient(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="what rules are active?")],
        "context": context,
        "routing": {"intent_label": "list_rules", "branches": ["procedures"]},
    }
    out = reflect_on_evidence(state)
    assert out["reflection_eval"]["sufficient"] is True
    assert chat.calls == []


def test_reflect_on_evidence_grounding_with_hits_is_sufficient(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="recommend a carrier")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": ["rag", "kg"]},
        "rag_hits": [{"source": "guide.md"}],
    }
    out = reflect_on_evidence(state)
    assert out["reflection_eval"]["sufficient"] is True
    assert chat.calls == []


def test_reflect_on_evidence_thin_evidence_calls_llm_for_replan(monkeypatch, context):
    import json
    payload = json.dumps({
        "sufficient": False,
        "missing": ["carrier SLA"],
        "followup_subquery": "carrier SLA for TX-AZ over 5000 lbs",
        "rationale": "no rag hits",
    })
    chat = FakeChatProvider(reply=payload, usage={"input_tokens": 12, "output_tokens": 8})
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="recommend a carrier")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": ["rag", "kg"]},
        "plan": {"replan_count": 0},
        "rag_hits": [],
        "kg_hits": [],
    }
    out = reflect_on_evidence(state)
    eval_out = out["reflection_eval"]
    assert eval_out["sufficient"] is False
    assert eval_out["followup_subquery"].startswith("carrier SLA")
    assert len(chat.calls) == 1


def test_reflect_on_evidence_forwards_when_budget_exhausted(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="recommend a carrier")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": ["rag", "kg"]},
        "plan": {"replan_count": 1},
        "rag_hits": [],
        "kg_hits": [],
    }
    out = reflect_on_evidence(state)
    assert out["reflection_eval"]["sufficient"] is True
    assert "evidence_insufficient" in out["degraded"]
    assert chat.calls == []


def test_reflect_on_evidence_llm_parse_failure_defaults_to_one_rescue(monkeypatch, context):
    chat = FakeChatProvider(reply="not json at all")
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="recommend a carrier")],
        "context": context,
        "routing": {"intent_label": "recommend_shipment", "branches": ["rag", "kg"]},
        "plan": {"replan_count": 0},
        "rag_hits": [],
        "kg_hits": [],
    }
    out = reflect_on_evidence(state)
    eval_out = out["reflection_eval"]
    assert eval_out["sufficient"] is False
    assert eval_out["followup_subquery"] == "recommend a carrier"


def test_retrievers_use_plan_subquery_when_present(monkeypatch, context):
    captured: dict[str, str] = {}

    class CapturingSemantic:
        def search(self, realm_id, user_id, query, limit):
            captured["ltm"] = query
            return []

    monkeypatch.setattr(nodes, "get_semantic_memory", lambda: CapturingSemantic())
    state = {
        "messages": [HumanMessage(content="raw user text")],
        "context": context,
        "plan": {"subquery": "refined carrier SLA lookup"},
    }
    retrieve_ltm(state)
    assert captured["ltm"] == "refined carrier SLA lookup"


def test_graph_wires_think_and_plan_and_reflection_loop(monkeypatch):
    from agent import graph as graph_mod
    monkeypatch.setattr(graph_mod, "get_checkpointer", lambda: None)
    monkeypatch.setattr(graph_mod, "get_store", lambda: None)
    g = graph_mod.build_graph()
    nodes_set = set(g.get_graph().nodes.keys())
    assert {"think_and_plan", "reflect_on_evidence"}.issubset(nodes_set)



# ---------------------- RAG hybrid + planner + reranker ----------------------


def test_query_planner_extracts_lane_and_carrier():
    from core.rag.query_planner import plan_query

    filters = plan_query("Carrier A weight threshold on TX-AZ shipments")
    assert "TX-AZ" in filters.lanes
    assert "Carrier A" in filters.carriers
    assert "lanes=" in filters.rationale


def test_query_planner_detects_doc_type_hints():
    from core.rag.query_planner import plan_query

    filters = plan_query("How do I handle a late delivery exception?")
    assert "exception_playbook" in filters.doc_types


def test_query_planner_empty_query_returns_empty_filters():
    from core.rag.query_planner import plan_query

    filters = plan_query("")
    assert filters.lanes == [] and filters.carriers == [] and filters.doc_types == []


def test_null_reranker_preserves_order_and_trims():
    from core.rag.rerank import NullReranker
    from core.schemas import KnowledgeHit

    hits = [
        KnowledgeHit(doc_type="x", source=f"s{i}", text=f"t{i}", score=1.0 - i * 0.1, metadata={})
        for i in range(4)
    ]
    out = NullReranker().rerank("q", hits, top_k=2)
    assert [h.source for h in out] == ["s0", "s1"]


def test_voyage_reranker_uses_relevance_score(monkeypatch):
    from core.rag.rerank import VoyageReranker
    from core.schemas import KnowledgeHit

    class _Entry:
        def __init__(self, index, score):
            self.index = index
            self.relevance_score = score

    class _Result:
        def __init__(self, entries):
            self.results = entries

    class _FakeClient:
        def rerank(self, *, query, documents, model, top_k):  # noqa: ARG002
            # Returns reversed input order with descending relevance scores
            # so we verify both the re-ordering and the score replacement.
            n = len(documents)
            return _Result([_Entry(n - 1 - rank, 1.0 - rank * 0.1) for rank in range(n)])

    hits = [
        KnowledgeHit(doc_type="x", source=f"s{i}", text=f"t{i}", score=0.5, metadata={})
        for i in range(3)
    ]
    reranker = VoyageReranker(client=_FakeClient())
    out = reranker.rerank("q", hits, top_k=3)
    assert [h.source for h in out] == ["s2", "s1", "s0"]
    assert out[0].score == 1.0
    assert out[1].score == 0.9


def test_rrf_fusion_prefers_documents_present_in_both_lists():
    from core.rag.mongo import _rrf_fuse
    from core.schemas import KnowledgeHit

    def _hit(source, chunk_index):
        return KnowledgeHit(
            doc_type="route_guide", source=source, text=f"text-{source}-{chunk_index}",
            score=0.0, metadata={"chunk_index": chunk_index, "lanes": ["TX-AZ"]},
        )
    # Note: hit identity is (source, chunk_index). Set chunk_index at top-level via extra-allow.
    a = KnowledgeHit(doc_type="x", source="shared", text="t", score=0.0, metadata={}, chunk_index=0)
    b = KnowledgeHit(doc_type="x", source="only_vec", text="t", score=0.0, metadata={}, chunk_index=0)
    c = KnowledgeHit(doc_type="x", source="only_bm25", text="t", score=0.0, metadata={}, chunk_index=0)
    fused = _rrf_fuse([a, b], [a, c], vector_weight=1.0, bm25_weight=1.0)
    assert fused[0].source == "shared"
    assert {h.source for h in fused} == {"shared", "only_vec", "only_bm25"}


def test_hybrid_retriever_runs_both_pipelines_and_dedups(monkeypatch):
    from core.rag.mongo import MongoKnowledgeRetriever
    from tests.fakes import FakeEmbeddings

    class _CapturingCollection:
        def __init__(self):
            self.pipelines: list[list[dict]] = []
            self._vector_hits = [
                {"doc_type": "route_guide", "source": "tx_az_lane.pdf",
                 "chunk_index": 0, "text": "Carrier A TX-AZ primary",
                 "metadata": {"lanes": ["TX-AZ"], "carriers": ["Carrier A"]}, "score": 0.8},
            ]
            self._bm25_hits = [
                {"doc_type": "route_guide", "source": "tx_az_lane.pdf",
                 "chunk_index": 0, "text": "Carrier A TX-AZ primary",
                 "metadata": {"lanes": ["TX-AZ"], "carriers": ["Carrier A"]}, "score": 12.3},
                {"doc_type": "carrier_sla", "source": "carrier_a_2026.pdf",
                 "chunk_index": 1, "text": "fuel surcharge structure TX-AZ Carrier A",
                 "metadata": {"lanes": ["TX-AZ", "TX-TX"], "carriers": ["Carrier A"]},
                 "score": 9.1},
            ]

        def aggregate(self, pipeline):
            self.pipelines.append(pipeline)
            stages = {next(iter(s)) for s in pipeline}
            if "$vectorSearch" in stages:
                return iter(self._vector_hits)
            if "$search" in stages:
                return iter(self._bm25_hits)
            return iter([])

    coll = _CapturingCollection()
    retriever = MongoKnowledgeRetriever(
        collection=coll, embeddings=FakeEmbeddings(),
        index_name="vec_idx", search_index_name="search_idx",
        hybrid_enabled=True, fusion_candidates=10,
    )
    hits = retriever.query(realm_id="r1", text="Carrier A surcharge on TX-AZ", k=5)
    sources = [h.source for h in hits]
    assert "tx_az_lane.pdf" in sources and "carrier_a_2026.pdf" in sources
    assert sources.count("tx_az_lane.pdf") == 1
    # Verified both stages were invoked.
    stage_sets = [{next(iter(s)) for s in p} for p in coll.pipelines]
    assert any("$vectorSearch" in s for s in stage_sets)
    assert any("$search" in s for s in stage_sets)


def test_vector_only_retriever_post_filters_on_lane(monkeypatch):
    from core.rag.mongo import MongoKnowledgeRetriever
    from tests.fakes import FakeEmbeddings

    class _Collection:
        def aggregate(self, _pipeline):
            return iter([
                {"doc_type": "route_guide", "source": "tx_az_lane.pdf", "chunk_index": 0,
                 "text": "TX-AZ guide", "metadata": {"lanes": ["TX-AZ"]}, "score": 0.9},
                {"doc_type": "route_guide", "source": "tx_ca_lane.pdf", "chunk_index": 0,
                 "text": "TX-CA guide", "metadata": {"lanes": ["TX-CA"]}, "score": 0.8},
            ])

    retriever = MongoKnowledgeRetriever(
        collection=_Collection(), embeddings=FakeEmbeddings(), index_name="vec_idx",
    )
    hits = retriever.query(realm_id="r1", text="What about TX-AZ?", k=5)
    assert [h.source for h in hits] == ["tx_az_lane.pdf"]


def test_post_filter_falls_back_when_filter_eliminates_all():
    from core.rag.mongo import MongoKnowledgeRetriever
    from tests.fakes import FakeEmbeddings

    class _Collection:
        def aggregate(self, _pipeline):
            return iter([
                {"doc_type": "route_guide", "source": "tx_ca_lane.pdf", "chunk_index": 0,
                 "text": "TX-CA guide", "metadata": {"lanes": ["TX-CA"]}, "score": 0.8},
            ])

    retriever = MongoKnowledgeRetriever(
        collection=_Collection(), embeddings=FakeEmbeddings(), index_name="vec_idx",
    )
    # Query mentions TX-AZ but only TX-CA chunk exists → fallback returns the candidate set.
    hits = retriever.query(realm_id="r1", text="TX-AZ shipments", k=5)
    assert [h.source for h in hits] == ["tx_ca_lane.pdf"]



# ---------------------------------------------------------------------------
# core.citations — sentence splitter + lexical-overlap matcher
# ---------------------------------------------------------------------------


def test_citations_split_sentences_preserves_offsets():
    from core.citations import split_sentences

    text = "First sentence. Second one is here. Third!"
    spans = split_sentences(text)
    assert [s for _, _, s in spans] == [
        "First sentence.",
        "Second one is here.",
        "Third!",
    ]
    for start, end, sentence in spans:
        assert text[start:end] == sentence


def test_citations_match_picks_highest_overlap_chunk():
    from core.citations import match_citations

    reply = "Carrier A serves the Austin Dallas dry van lane with no fuel surcharge under 20000 lb."
    rag = [
        {"doc_type": "policy", "source": "noise.pdf", "text": "Detention rules and dock holds.", "score": 0.9},
        {
            "doc_type": "route_guide",
            "source": "austin_dallas.pdf",
            "text": "Carrier A dry van Austin Dallas; no fuel surcharge under 20000 lb threshold.",
            "score": 0.6,
        },
    ]
    spans = match_citations(reply, rag, None)
    assert len(spans) == 1
    assert spans[0]["source"] == "austin_dallas.pdf"
    assert spans[0]["kind"] == "rag"
    assert spans[0]["overlap"] >= 3


def test_citations_match_breaks_ties_by_score():
    from core.citations import match_citations

    reply = "Carrier A handles TX-TX same day with dry van."
    rag = [
        {"doc_type": "scorecard", "source": "low.pdf",
         "text": "Carrier A TX-TX dry van same day.", "score": 0.40},
        {"doc_type": "scorecard", "source": "high.pdf",
         "text": "Carrier A TX-TX dry van same day.", "score": 0.95},
    ]
    spans = match_citations(reply, rag, None)
    assert len(spans) == 1
    assert spans[0]["source"] == "high.pdf"


def test_citations_match_returns_empty_when_no_chunks():
    from core.citations import match_citations
    assert match_citations("Anything goes.", [], []) == []
    assert match_citations("Anything goes.", None, None) == []


def test_citations_match_skips_sentences_below_overlap_floor():
    from core.citations import match_citations

    reply = "Hello. Carrier A dry van TX-TX same day Austin Dallas."
    rag = [{"doc_type": "scorecard", "source": "s.pdf",
            "text": "Carrier A dry van TX-TX same day Austin Dallas.", "score": 0.7}]
    spans = match_citations(reply, rag, None)
    # 'Hello.' has no overlap; only the second sentence is cited.
    assert [s["sentence_idx"] for s in spans] == [1]


def test_citations_match_uses_kg_facts():
    from core.citations import match_citations

    reply = "Carrier A serves lane TX-AZ Austin Phoenix priority one."
    kg = [{"fact": "Carrier A serves lane TX-AZ Austin Phoenix priority=1",
           "source_doc": "kg/tx_az.json"}]
    spans = match_citations(reply, None, kg)
    assert len(spans) == 1
    assert spans[0]["kind"] == "kg"
    assert spans[0]["source"] == "kg/tx_az.json"



# ---------------------- P2 #8: retry-from-failed-node helpers ----------------------


def test_parse_failure_marker_recognises_structured_failed():
    from agent.graph import parse_failure_marker

    assert parse_failure_marker("structured_failed:plan_action") == "plan_action"
    assert parse_failure_marker("structured_failed:reflect_on_evidence") == "reflect_on_evidence"
    assert parse_failure_marker("structured_failed:save_memory") == "save_memory"
    assert parse_failure_marker("structured_failed:review_draft") == "review_draft"


def test_parse_failure_marker_recognises_safe_retrieve_markers():
    from agent.graph import parse_failure_marker

    assert parse_failure_marker("retrieve_rag: TimeoutError: connection timed out") == "retrieve_rag"
    assert parse_failure_marker("retrieve_kg: ValueError: bad cursor") == "retrieve_kg"
    assert parse_failure_marker("classify_intent: RuntimeError: router down") == "classify_intent"


def test_parse_failure_marker_recognises_reflection_failure():
    from agent.graph import parse_failure_marker

    assert parse_failure_marker("reflection_failed") == "save_memory"


def test_parse_failure_marker_skips_informational_markers():
    from agent.graph import parse_failure_marker

    for marker in (
        "chat_fallback:openai",
        "structured_retry:plan_action",
        "cost_extracted_via_fallback",
        "citations_missing",
        "evidence_insufficient",
        "draft_review_ok",
        "draft_revised",
        "draft_review_skipped:short_reply",
        "",
        "unknown_marker",
    ):
        assert parse_failure_marker(marker) is None, marker


def test_retryable_failures_dedupes_by_node_and_preserves_order():
    from agent.graph import retryable_failures

    out = retryable_failures([
        "chat_fallback:openai",
        "structured_failed:plan_action",
        "retrieve_rag: TimeoutError: x",
        "structured_failed:plan_action",  # dup → collapsed
        "reflection_failed",
        "citations_missing",
    ])
    assert out == [
        ("structured_failed:plan_action", "plan_action"),
        ("retrieve_rag: TimeoutError: x", "retrieve_rag"),
        ("reflection_failed", "save_memory"),
    ]


def test_retryable_failures_handles_empty_input():
    from agent.graph import retryable_failures

    assert retryable_failures(None) == []
    assert retryable_failures([]) == []


def test_find_retry_checkpoint_walks_history_newest_first():
    """`find_retry_checkpoint` returns the most recent snapshot whose `next` contains the target."""
    from agent.graph import find_retry_checkpoint

    class _Snap:
        def __init__(self, nxt, cid):
            self.next = nxt
            self.config = {"configurable": {"checkpoint_id": cid}}

    class _FakeGraph:
        def __init__(self, history):
            self._history = history
        def get_state_history(self, _config):
            return iter(self._history)

    history = [
        _Snap(("save_memory",), "ck-3"),
        _Snap(("plan_action",), "ck-2-newest-plan"),
        _Snap(("plan_action",), "ck-2-older-plan"),
        _Snap(("retrieve_rag",), "ck-1"),
    ]
    graph = _FakeGraph(history)
    config = {"configurable": {"thread_id": "t1"}}

    cfg = find_retry_checkpoint(graph, config, "plan_action")
    assert cfg == {"configurable": {"checkpoint_id": "ck-2-newest-plan"}}
    cfg = find_retry_checkpoint(graph, config, "retrieve_rag")
    assert cfg == {"configurable": {"checkpoint_id": "ck-1"}}
    assert find_retry_checkpoint(graph, config, "nonexistent_node") is None


def test_find_retry_checkpoint_swallows_history_errors():
    from agent.graph import find_retry_checkpoint

    class _BoomGraph:
        def get_state_history(self, _config):
            raise RuntimeError("checkpointer offline")

    assert find_retry_checkpoint(_BoomGraph(), {"configurable": {}}, "plan_action") is None
