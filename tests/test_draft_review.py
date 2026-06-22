"""Unit tests for the opt-in `review_draft` Writer/Reviewer node."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.nodes import review_draft
from core.settings import Settings
from tests.fakes import FakeChatProvider


def _settings(*, enabled: bool, min_chars: int = 200) -> Settings:
    return Settings(
        mongodb_uri="mongodb://localhost",
        review_draft_enabled=enabled,
        review_draft_min_chars=min_chars,
    )


def _draft_state(context, *, draft: str, user: str = "recommend a carrier", rag_context: str = ""):
    return {
        "messages": [HumanMessage(content=user), AIMessage(content=draft)],
        "context": context,
        "rag_context": rag_context,
    }


def test_review_draft_is_noop_when_flag_off(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=False))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(context, draft="x" * 400, rag_context="Carrier A serves TX-AZ."))
    assert "degraded" not in out
    assert "messages" not in out
    assert chat.calls == []


def test_review_draft_skips_when_no_prior_ai_message(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {"messages": [HumanMessage(content="hi")], "context": context, "rag_context": "x"}
    out = review_draft(state)
    assert out.get("degraded") == ["draft_review_skipped:no_draft"]
    assert chat.calls == []


def test_review_draft_skips_short_reply(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=200))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(context, draft="too short", rag_context="Carrier A serves TX-AZ."))
    assert out.get("degraded") == ["draft_review_skipped:short_reply"]
    assert chat.calls == []


def test_review_draft_skips_when_no_evidence(monkeypatch, context):
    chat = FakeChatProvider(reply="should not be called")
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=10))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(context, draft="x" * 50, rag_context=""))
    assert out.get("degraded") == ["draft_review_skipped:no_evidence"]
    assert chat.calls == []


def test_review_draft_ok_when_reviewer_approves(monkeypatch, context):
    payload = json.dumps({"needs_revision": False, "revised_reply": None, "reasons": ["addresses all sub-questions"]})
    chat = FakeChatProvider(reply=payload, usage={"input_tokens": 30, "output_tokens": 10})
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=10))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(context, draft="x" * 50, rag_context="Carrier A serves TX-AZ."))
    assert "messages" not in out
    assert out.get("degraded") == ["draft_review_ok"]
    assert len(chat.calls) == 1
    assert "usage" in out


def test_review_draft_appends_revised_message_when_flagged(monkeypatch, context):
    revised = "Per carrier_a_2026.pdf, Carrier A serves the TX-AZ lane within 2 days."
    payload = json.dumps({
        "needs_revision": True,
        "revised_reply": revised,
        "reasons": ["draft was missing the source filename"],
    })
    chat = FakeChatProvider(reply=payload)
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=10))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(
        context,
        draft="Carrier A serves TX-AZ within 2 days.",
        rag_context="Carrier A serves TX-AZ; transit 2 days. (carrier_a_2026.pdf)",
    ))
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == revised
    assert "draft_revised" in out.get("degraded", [])
    assert "draft_review_ok" not in out.get("degraded", [])


def test_review_draft_does_not_replace_when_revised_reply_blank(monkeypatch, context):
    payload = json.dumps({"needs_revision": True, "revised_reply": "   ", "reasons": ["unclear"]})
    chat = FakeChatProvider(reply=payload)
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=10))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(context, draft="x" * 50, rag_context="some evidence"))
    assert "messages" not in out
    assert out.get("degraded") == ["draft_review_ok"]


def test_review_draft_records_structured_failure_on_unparseable_output(monkeypatch, context):
    chat = FakeChatProvider(reply="not json at all")
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=10))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    out = review_draft(_draft_state(context, draft="x" * 50, rag_context="some evidence"))
    assert "messages" not in out
    markers = out.get("degraded", [])
    assert "structured_failed:review_draft" in markers
    assert "draft_review_ok" not in markers


def test_graph_includes_review_draft_node(monkeypatch):
    from agent import graph as graph_mod
    monkeypatch.setattr(graph_mod, "get_checkpointer", lambda: None)
    monkeypatch.setattr(graph_mod, "get_store", lambda: None)
    g = graph_mod.build_graph()
    assert "review_draft" in set(g.get_graph().nodes.keys())


def test_route_after_writer_branches_on_flag(monkeypatch):
    from agent import graph as graph_mod
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(enabled=False))
    assert graph_mod._route_after_writer({}) == "validate_citations"
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(enabled=True))
    assert graph_mod._route_after_writer({}) == "review_draft"


def test_review_draft_prompt_preserves_grounded_numbers(monkeypatch, context):
    """Reviewer prompt must include grounded numbers from the evidence and
    instruct the LLM to preserve them; downstream `plan_action` extracts
    `estimated_cost_usd` from the draft and breaks when numbers are dropped.
    """
    payload = json.dumps({"needs_revision": False, "revised_reply": None, "reasons": ["numeric claims grounded"]})
    chat = FakeChatProvider(reply=payload, usage={"input_tokens": 40, "output_tokens": 10})
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(enabled=True, min_chars=10))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    draft = (
        "Recommendation: Carrier A dry van for 15,000 lb Austin-to-Dallas. "
        "Typical all-in quote $410-$475. Sources: route_guides/austin_dallas_hot_lane.pdf"
    )
    evidence = "Carrier A; Austin-Dallas; 15,000 lb; typical all-in $410-$475; transit 8h. (austin_dallas_hot_lane.pdf)"
    out = review_draft(_draft_state(context, draft=draft, rag_context=evidence))
    assert out.get("degraded") == ["draft_review_ok"]
    assert len(chat.calls) == 1
    prompt = chat.calls[0]
    assert "$410-$475" in prompt
    assert "15,000 lb" in prompt
    assert "Preserve every numeric value" in prompt
