"""Shared pytest fixtures. Tests run without env vars or Atlas access."""
from __future__ import annotations

import pytest

from core.settings import AgentContext, get_settings


@pytest.fixture(autouse=True)
def _isolate_cached_settings(monkeypatch):
    """Keep unit tests independent of execution order and local `.env` files."""
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def context() -> AgentContext:
    return AgentContext(realm_id="realm-test", user_id="user-test", agent_id="agent-test")
