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
