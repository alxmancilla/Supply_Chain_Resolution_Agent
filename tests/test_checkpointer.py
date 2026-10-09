"""Checkpoint serializer wiring remains explicit and strict-mode ready."""
from __future__ import annotations


def test_get_checkpointer_allows_only_agent_context_custom_state(monkeypatch):
    from agent import memory

    captured = {}

    class _FakeSaver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(memory, "MongoDBSaver", _FakeSaver)
    monkeypatch.setattr(memory, "get_mongo_client", lambda: object())
    memory.get_checkpointer.cache_clear()
    try:
        memory.get_checkpointer()
    finally:
        memory.get_checkpointer.cache_clear()

    allowed = captured["serde"]._allowed_msgpack_modules
    assert allowed == {("core.settings", "AgentContext")}