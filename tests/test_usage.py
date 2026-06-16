"""Unit tests for token + cost accounting (core/usage.py + state plumbing)."""
from __future__ import annotations

from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.nodes import generate_response, plan_action, save_memory
from core.schemas import BookingProposal, MemoryExtraction
from core.settings import get_settings
from core.usage import (
    compute_cost_usd,
    extract_usage_metadata,
    merge_usage,
    usage_payload,
)
from tests.fakes import (
    FakeChatProvider,
    FakeEpisodicMemory,
    FakeSemanticMemory,
)


def _settings_with_prices(**overrides):
    base = get_settings()
    return replace(base, **overrides)


def test_extract_usage_metadata_returns_input_output_tokens():
    msg = AIMessage(content="hi", usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17})
    assert extract_usage_metadata(msg) == {"input_tokens": 12, "output_tokens": 5}


def test_extract_usage_metadata_returns_none_when_missing():
    assert extract_usage_metadata(AIMessage(content="hi")) is None
    assert extract_usage_metadata(None) is None


def test_compute_cost_usd_applies_per_1k_rates():
    settings = _settings_with_prices(
        chat_input_price_per_1k_usd=0.001,
        chat_output_price_per_1k_usd=0.002,
    )
    assert compute_cost_usd(1000, 500, settings) == 0.001 + 0.001


def test_usage_payload_builds_delta_with_calls():
    settings = _settings_with_prices(
        chat_input_price_per_1k_usd=0.001,
        chat_output_price_per_1k_usd=0.002,
    )
    payload = usage_payload(2000, 1000, settings)
    assert payload == {"tokens_in": 2000.0, "tokens_out": 1000.0, "cost_usd": 0.004, "calls": 1.0}


def test_merge_usage_sums_per_node_deltas():
    a = {"tokens_in": 100.0, "tokens_out": 50.0, "cost_usd": 0.0003, "calls": 1.0}
    b = {"tokens_in": 200.0, "tokens_out": 75.0, "cost_usd": 0.0005, "calls": 1.0}
    merged = merge_usage(a, b)
    assert merged["tokens_in"] == 300.0
    assert merged["tokens_out"] == 125.0
    assert merged["calls"] == 2.0
    assert abs(merged["cost_usd"] - 0.0008) < 1e-9


def test_merge_usage_handles_missing_sides():
    assert merge_usage(None, None) == {}
    assert merge_usage({"calls": 1.0}, None) == {"calls": 1.0}
    assert merge_usage(None, {"calls": 1.0}) == {"calls": 1.0}


class _StreamingLLMWithUsage:
    def __init__(self, deltas, usage):
        self._deltas = deltas
        self._usage = usage

    def stream(self, _prompt):
        for i, d in enumerate(self._deltas):
            meta = self._usage if i == len(self._deltas) - 1 else None
            yield AIMessage(content=d, usage_metadata=meta) if meta else _Chunk(d)


class _Chunk:
    def __init__(self, text):
        self.content = text
        self.usage_metadata = None


def test_generate_response_emits_usage_from_streamed_chunks(monkeypatch, context):
    usage_meta = {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48}
    monkeypatch.setattr(nodes, "get_llm", lambda: _StreamingLLMWithUsage(["Hi ", "there."], usage_meta))
    out = generate_response({"messages": [HumanMessage(content="hello")], "context": context})
    assert out["messages"][0].content == "Hi there."
    assert out["usage"]["tokens_in"] == 40.0
    assert out["usage"]["tokens_out"] == 8.0
    assert out["usage"]["calls"] == 1.0


def test_plan_action_emits_usage_from_chat_provider(monkeypatch, context):
    chat = FakeChatProvider(
        reply='{"action_type": "none"}',
        usage={"input_tokens": 30, "output_tokens": 5},
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    state = {
        "messages": [HumanMessage(content="ship 1000lbs"), AIMessage(content="recommend Carrier A")],
        "context": context,
    }
    out = plan_action(state)
    assert out["action_plan"]["action_type"] == "none"
    assert out["usage"]["tokens_in"] == 30.0
    assert out["usage"]["tokens_out"] == 5.0


def test_save_memory_emits_usage_from_chat_provider(monkeypatch, context):
    chat = FakeChatProvider(
        reply='{"facts": ["User prefers Carrier A."], "episode": null}',
        usage={"input_tokens": 80, "output_tokens": 20},
    )
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chat)
    monkeypatch.setattr(nodes, "get_semantic_memory", lambda: FakeSemanticMemory([]))
    monkeypatch.setattr(nodes, "get_episodic_memory", lambda: FakeEpisodicMemory([]))
    state = {
        "messages": [HumanMessage(content="my carrier was A"), AIMessage(content="noted")],
        "context": context,
    }
    out = save_memory(state)
    assert out["usage"]["tokens_in"] == 80.0
    assert out["usage"]["tokens_out"] == 20.0
    assert out["usage"]["calls"] == 1.0
