"""Unit tests for the provider abstraction layer."""
from __future__ import annotations

import pytest

from core.protocols import ChatProvider, EmbeddingProvider
from core.providers.embeddings._langchain_adapter import (
    LangChainEmbeddingsAdapter,
    to_langchain_embeddings,
)
from tests.fakes import FakeChatProvider, FakeEmbeddings


def test_fake_embeddings_satisfies_protocol():
    emb = FakeEmbeddings()
    assert isinstance(emb, EmbeddingProvider)
    assert emb.dimensions == 1024
    assert len(emb.embed_query("hi")) == 1024
    assert [len(v) for v in emb.embed_documents(["a", "b"])] == [1024, 1024]


def test_fake_chat_provider_satisfies_protocol():
    chat = FakeChatProvider(reply="PONG")
    assert isinstance(chat, ChatProvider)
    assert chat.invoke("ping") == "PONG"
    assert chat.calls == ["ping"]


def test_fake_chat_provider_raises_on_demand():
    chat = FakeChatProvider(raise_exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        chat.invoke("x")


def test_langchain_adapter_round_trips_embedding_provider():
    emb = FakeEmbeddings()
    adapter = to_langchain_embeddings(emb)
    assert isinstance(adapter, LangChainEmbeddingsAdapter)
    assert adapter.model_name == "fake-embeddings"
    assert adapter.dimensions == 1024
    assert adapter.embed_query("q") == [0.0] * 1024
    assert adapter.embed_documents(["a", "b", "c"]) == [[0.0] * 1024] * 3


def test_invoke_typed_validates_against_schema():
    from pydantic import BaseModel

    class Out(BaseModel):
        a: int
        b: str

    chat = FakeChatProvider(reply='{"a": 1, "b": "x"}')
    result = chat.invoke_typed("p", Out)
    assert isinstance(result, Out)
    assert result.a == 1 and result.b == "x"


class _FakeCollection:
    """Stand-in for a pymongo Collection exposing `list_search_indexes`."""

    def __init__(self, name: str, entries: list[dict] | Exception | None = None):
        self.name = name
        self._entries = entries

    def list_search_indexes(self, name: str):  # noqa: ARG002
        if isinstance(self._entries, Exception):
            raise self._entries
        return iter(self._entries or [])


def test_assert_vector_index_dims_passes_on_match():
    from agent.memory import _assert_vector_index_dims

    coll = _FakeCollection(
        "agent_memories",
        [{"latestDefinition": {"fields": [{"type": "vector", "numDimensions": 1024}]}}],
    )
    _assert_vector_index_dims(coll, "agent_memories_vector", 1024)


def test_assert_vector_index_dims_raises_on_mismatch():
    from agent.memory import _assert_vector_index_dims

    coll = _FakeCollection(
        "agent_memories",
        [{"latestDefinition": {"fields": [{"type": "vector", "numDimensions": 3072}]}}],
    )
    with pytest.raises(RuntimeError, match="expects 3072 dims"):
        _assert_vector_index_dims(coll, "agent_memories_vector", 1024)


def test_assert_vector_index_dims_noop_when_missing():
    from agent.memory import _assert_vector_index_dims

    _assert_vector_index_dims(_FakeCollection("agent_memories", []), "agent_memories_vector", 1024)


def test_assert_vector_index_dims_noop_when_unsupported():
    from agent.memory import _assert_vector_index_dims

    coll = _FakeCollection("agent_memories", RuntimeError("listSearchIndexes unsupported"))
    _assert_vector_index_dims(coll, "agent_memories_vector", 1024)


def test_agent_context_auto_generates_unique_correlation_id():
    from core.settings import AgentContext

    a = AgentContext(realm_id="r", user_id="u", agent_id="a")
    b = AgentContext(realm_id="r", user_id="u", agent_id="a")
    assert a.correlation_id and b.correlation_id
    assert a.correlation_id != b.correlation_id


def test_settings_default_model_names_match_baseline():
    import os
    from core.settings import get_settings

    get_settings.cache_clear()
    for key in ("EMBEDDING_MODEL", "CHAT_MODEL"):
        os.environ.pop(key, None)
    os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/test")
    settings = get_settings()
    assert settings.embedding_model == "voyage-4"
    assert settings.chat_model == "gpt-5.5"
    get_settings.cache_clear()


def test_settings_honors_env_overrides_for_model_names(monkeypatch):
    from core.settings import get_settings

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-large-2")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.embedding_model == "voyage-large-2"
        assert settings.chat_model == "gpt-4o-mini"
    finally:
        get_settings.cache_clear()


def test_registry_passes_settings_model_names_to_providers(monkeypatch):
    from core.providers import registry
    from core.settings import get_settings

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-custom")
    monkeypatch.setenv("CHAT_MODEL", "grove-custom")
    get_settings.cache_clear()
    registry.reset_provider_cache()
    try:
        emb = registry.get_embedding_provider()
        chat = registry.get_chat_provider()
        assert emb.model_name == "voyage-custom"
        assert chat.model_name == "grove-custom"
    finally:
        get_settings.cache_clear()
        registry.reset_provider_cache()


# ----------------------- Cross-provider chat fallback -----------------------


class _Synthetic5xx(Exception):
    """HTTP-5xx-style error: detected by `status_code` attribute."""

    def __init__(self, status_code: int = 503) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class RateLimitError(Exception):
    """Class-name match for the openai-python error of the same name."""


class APITimeoutError(Exception):
    """Class-name match for the openai-python error of the same name."""


def test_is_retryable_recognizes_rate_limit_and_timeout_by_name():
    from core.providers.chat.fallback import is_retryable_chat_error

    assert is_retryable_chat_error(RateLimitError("rl"))
    assert is_retryable_chat_error(APITimeoutError("to"))


def test_is_retryable_recognizes_5xx_and_429_by_status_code():
    from core.providers.chat.fallback import is_retryable_chat_error

    assert is_retryable_chat_error(_Synthetic5xx(500))
    assert is_retryable_chat_error(_Synthetic5xx(429))
    assert is_retryable_chat_error(_Synthetic5xx(408))
    assert not is_retryable_chat_error(_Synthetic5xx(400))
    assert not is_retryable_chat_error(_Synthetic5xx(404))


def test_is_retryable_passes_through_builtin_timeout_and_connection():
    from core.providers.chat.fallback import is_retryable_chat_error

    assert is_retryable_chat_error(TimeoutError("t"))
    assert is_retryable_chat_error(ConnectionError("c"))
    assert not is_retryable_chat_error(ValueError("nope"))


def test_fallback_provider_uses_primary_when_healthy():
    from core.providers.chat.fallback import FallbackChatProvider

    primary = FakeChatProvider(reply="P", usage={"input_tokens": 3, "output_tokens": 2})
    secondary = FakeChatProvider(reply="S")
    chain = FallbackChatProvider(("grove", primary), ("openai", secondary))
    assert chain.invoke("ping") == "P"
    assert chain.last_fallback is None
    assert chain.last_usage == {"input_tokens": 3, "output_tokens": 2}
    assert secondary.calls == []


def test_fallback_provider_advances_on_retryable_and_records_name():
    from core.providers.chat.fallback import FallbackChatProvider

    primary = FakeChatProvider(raise_exc=RateLimitError("rate limited"))
    secondary = FakeChatProvider(reply="S", usage={"input_tokens": 1, "output_tokens": 4})
    chain = FallbackChatProvider(("grove", primary), ("openai", secondary))
    assert chain.invoke("ping") == "S"
    assert chain.last_fallback == "openai"
    assert chain.last_usage == {"input_tokens": 1, "output_tokens": 4}


def test_fallback_provider_reraises_non_retryable_immediately():
    from core.providers.chat.fallback import FallbackChatProvider

    primary = FakeChatProvider(raise_exc=ValueError("bad input"))
    secondary = FakeChatProvider(reply="S")
    chain = FallbackChatProvider(("grove", primary), ("openai", secondary))
    with pytest.raises(ValueError, match="bad input"):
        chain.invoke("ping")
    assert secondary.calls == []
    assert chain.last_fallback is None


def test_fallback_provider_raises_runtime_error_when_all_exhausted():
    from core.providers.chat.fallback import FallbackChatProvider

    primary = FakeChatProvider(raise_exc=_Synthetic5xx(503))
    secondary = FakeChatProvider(raise_exc=RateLimitError("rl"))
    chain = FallbackChatProvider(("grove", primary), ("openai", secondary))
    with pytest.raises(RuntimeError, match="all chat providers failed"):
        chain.invoke("ping")


def test_fallback_provider_forwards_invoke_typed():
    from pydantic import BaseModel

    from core.providers.chat.fallback import FallbackChatProvider

    class Out(BaseModel):
        v: int

    primary = FakeChatProvider(raise_exc=APITimeoutError("t"))
    secondary = FakeChatProvider(reply='{"v": 7}')
    chain = FallbackChatProvider(("grove", primary), ("openai", secondary))
    parsed = chain.invoke_typed("p", Out)
    assert isinstance(parsed, Out) and parsed.v == 7
    assert chain.last_fallback == "openai"


def test_registry_returns_single_provider_for_single_name(monkeypatch):
    from core.providers import registry
    from core.settings import get_settings

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
    monkeypatch.delenv("CHAT_PROVIDERS", raising=False)
    get_settings.cache_clear()
    registry.reset_provider_cache()
    try:
        chat = registry.get_chat_provider()
        from core.providers.chat.grove import GroveChatProvider
        assert isinstance(chat, GroveChatProvider)
    finally:
        get_settings.cache_clear()
        registry.reset_provider_cache()


def test_registry_composes_fallback_when_chat_providers_set(monkeypatch):
    from core.providers import registry
    from core.providers.chat.fallback import FallbackChatProvider
    from core.settings import get_settings

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
    monkeypatch.setenv("CHAT_PROVIDERS", "grove,grove")
    get_settings.cache_clear()
    registry.reset_provider_cache()
    try:
        chat = registry.get_chat_provider()
        assert isinstance(chat, FallbackChatProvider)
        assert tuple(chat.provider_chain) == ("grove", "grove")
    finally:
        get_settings.cache_clear()
        registry.reset_provider_cache()


def test_settings_rejects_unknown_chat_providers_entry(monkeypatch):
    from core.settings import get_settings

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
    monkeypatch.setenv("CHAT_PROVIDERS", "grove,does-not-exist")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="unknown providers"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_reflect_on_evidence_emits_chat_fallback_marker(monkeypatch):
    """Acceptance test: a fallback during reflect surfaces in `degraded`."""
    import json
    from langchain_core.messages import HumanMessage

    from agent import nodes
    from core.providers.chat.fallback import FallbackChatProvider
    from core.settings import AgentContext

    payload = json.dumps({
        "sufficient": False,
        "missing": ["carrier SLA"],
        "followup_subquery": "carrier SLA for TX-AZ",
        "rationale": "no rag hits",
    })
    primary = FakeChatProvider(raise_exc=RateLimitError("rate limited"))
    secondary = FakeChatProvider(reply=payload)
    chain = FallbackChatProvider(("grove", primary), ("openai", secondary))
    monkeypatch.setattr(nodes, "get_chat_provider", lambda: chain)
    state = {
        "messages": [HumanMessage(content="recommend a carrier")],
        "context": AgentContext(realm_id="r", user_id="u", agent_id="a"),
        "routing": {"intent_label": "recommend_shipment", "branches": ["rag", "kg"]},
        "plan": {"replan_count": 0},
        "rag_hits": [],
        "kg_hits": [],
    }
    out = nodes.reflect_on_evidence(state)
    assert out["reflection_eval"]["sufficient"] is False
    assert "chat_fallback:openai" in out["degraded"]


def test_timed_decorator_records_latency_and_opens_span_safely():
    from core.latency import timed
    from core.settings import AgentContext

    @timed("probe_ms")
    def node(state):
        return {"ok": True}

    ctx = AgentContext(realm_id="r", user_id="u", agent_id="a")
    out = node({"context": ctx, "routing": {"intent_label": "test"}})
    assert out["ok"] is True
    assert "probe_ms" in out["latency_ms"]
    assert out["latency_ms"]["probe_ms"] >= 0.0
