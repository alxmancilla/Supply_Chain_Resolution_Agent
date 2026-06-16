"""Unit tests for the intent router node and conditional routing helper."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent import nodes
from agent.graph import _route_to_retrievers
from agent.nodes import classify_intent
from core.router import ChainedIntentRouter, HeuristicIntentRouter, LLMIntentRouter
from core.schemas import ALL_BRANCHES, RoutingDecision
from tests.fakes import FakeIntentRouter


def _state(context, text: str = "hello"):
    return {"messages": [HumanMessage(content=text)], "context": context}


def test_classify_intent_happy_path(monkeypatch, context):
    decision = {
        "intent_label": "recall_preference",
        "branches": ["ltm", "episodes"],
        "rationale": "asking about prior preference",
    }
    monkeypatch.setattr(nodes, "get_intent_router", lambda: FakeIntentRouter(decision=decision))
    out = classify_intent(_state(context, "what carrier did I prefer?"))
    assert out["routing"]["intent_label"] == "recall_preference"
    assert out["routing"]["branches"] == ["ltm", "episodes"]
    assert "router_ms" in out["latency_ms"]
    assert out["degraded"] == ["__RESET__"]


def test_classify_intent_no_message_falls_back(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_intent_router", lambda: FakeIntentRouter())
    out = classify_intent({"messages": [], "context": context})
    assert out["routing"]["intent_label"] == "fallback"
    assert set(out["routing"]["branches"]) == set(ALL_BRANCHES)


def test_classify_intent_router_failure_degrades(monkeypatch, context):
    monkeypatch.setattr(
        nodes,
        "get_intent_router",
        lambda: FakeIntentRouter(raise_exc=RuntimeError("LLM unreachable")),
    )
    out = classify_intent(_state(context, "ship 15k lbs Austin to Dallas"))
    assert out["routing"]["intent_label"] == "fallback"
    assert set(out["routing"]["branches"]) == set(ALL_BRANCHES)
    assert any("classify_intent" in m for m in out.get("degraded", []))


def test_classify_intent_empty_branches_defaults_to_all(monkeypatch, context):
    decision = {"intent_label": "fallback", "branches": [], "rationale": "unclear"}
    monkeypatch.setattr(nodes, "get_intent_router", lambda: FakeIntentRouter(decision=decision))
    out = classify_intent(_state(context, "??"))
    assert set(out["routing"]["branches"]) == set(ALL_BRANCHES)


def test_route_to_retrievers_maps_subset():
    state = {"routing": {"branches": ["ltm", "episodes"]}}
    assert _route_to_retrievers(state) == ["retrieve_ltm", "retrieve_episodes"]


def test_route_to_retrievers_defaults_to_all_when_missing():
    assert sorted(_route_to_retrievers({})) == sorted(
        ["retrieve_ltm", "retrieve_episodes", "retrieve_procedures", "retrieve_rag", "retrieve_kg"]
    )


def test_route_to_retrievers_ignores_unknown_branches():
    state = {"routing": {"branches": ["ltm", "bogus"]}}
    assert _route_to_retrievers(state) == ["retrieve_ltm"]


def test_heuristic_matches_recall_preference():
    r = HeuristicIntentRouter()
    d = r.route_optional("What carrier did I prefer last time on TX-AZ?")
    assert d is not None
    assert d.intent_label == "recall_preference"
    assert d.branches == ["ltm", "episodes"]


def test_heuristic_matches_recommend_shipment():
    r = HeuristicIntentRouter()
    d = r.route_optional("I need to ship 15,000 lbs Austin to Dallas")
    assert d is not None
    assert d.intent_label == "recommend_shipment"
    assert set(d.branches) == set(ALL_BRANCHES)


def test_heuristic_matches_list_rules():
    r = HeuristicIntentRouter()
    d = r.route_optional("list the operating rules")
    assert d is not None
    assert d.intent_label == "list_rules"
    assert d.branches == ["procedures"]


def test_heuristic_matches_lookup_policy():
    r = HeuristicIntentRouter()
    d = r.route_optional("what does the SLA say about fuel surcharge?")
    assert d is not None
    assert d.intent_label == "lookup_policy"
    assert d.branches == ["rag", "procedures"]


def test_heuristic_matches_structured_lookup_kg():
    r = HeuristicIntentRouter()
    d = r.route_optional("Which carriers serve TX-AZ with no fuel surcharge?")
    assert d is not None
    assert d.intent_label == "structured_lookup"
    assert d.branches == ["kg", "rag"]


def test_heuristic_matches_propose_procedure_going_forward():
    r = HeuristicIntentRouter()
    d = r.route_optional("Going forward, always escalate Carrier C bookings over $5k.")
    assert d is not None
    assert d.intent_label == "propose_procedure"
    assert d.branches == ["procedures"]


def test_heuristic_matches_propose_procedure_from_now_on():
    r = HeuristicIntentRouter()
    d = r.route_optional("from now on, prefer Carrier A on TX-AZ")
    assert d is not None
    assert d.intent_label == "propose_procedure"


def test_route_to_retrievers_maps_kg_only():
    state = {"routing": {"branches": ["kg"]}}
    assert _route_to_retrievers(state) == ["retrieve_kg"]


def test_heuristic_returns_none_on_unknown():
    r = HeuristicIntentRouter()
    assert r.route_optional("hi there") is None


def test_heuristic_route_falls_back_to_all():
    r = HeuristicIntentRouter()
    d = r.route("hi there")
    assert d.intent_label == "fallback"
    assert set(d.branches) == set(ALL_BRANCHES)


class _FakeChatProvider:
    """Mimics the `ChatProvider` protocol — `.invoke(str) -> str`."""

    model_name: str = "fake"

    def __init__(self, content: str): self._content = content
    def invoke(self, _prompt: str) -> str: return self._content
    def invoke_typed(self, _prompt: str, _schema):  # pragma: no cover
        raise NotImplementedError


def test_llm_router_sanitizes_unknown_branches():
    chat = _FakeChatProvider('{"intent_label":"x","branches":["ltm","bogus","rag"],"rationale":"r"}')
    d = LLMIntentRouter(chat=chat).route("anything")
    assert d.branches == ["ltm", "rag"]


def test_llm_router_empty_branches_default_to_all():
    chat = _FakeChatProvider('{"intent_label":"x","branches":[],"rationale":"r"}')
    d = LLMIntentRouter(chat=chat).route("anything")
    assert set(d.branches) == set(ALL_BRANCHES)


def test_chained_router_short_circuits_on_heuristic():
    class _NeverChat:
        model_name = "never"
        def invoke(self, _prompt): raise AssertionError("LLM should not be called")
        def invoke_typed(self, _p, _s): raise AssertionError("LLM should not be called")
    chained = ChainedIntentRouter(
        heuristic=HeuristicIntentRouter(),
        llm_router=LLMIntentRouter(chat=_NeverChat()),
    )
    d = chained.route("What carrier did I prefer last time?")
    assert d.intent_label == "recall_preference"


def test_chained_router_falls_through_to_llm():
    chat = _FakeChatProvider('{"intent_label":"fallback","branches":["ltm"],"rationale":"r"}')
    chained = ChainedIntentRouter(
        heuristic=HeuristicIntentRouter(),
        llm_router=LLMIntentRouter(chat=chat),
    )
    d = chained.route("just a vague question without keywords")
    assert d.intent_label == "fallback"
    assert d.branches == ["ltm"]
