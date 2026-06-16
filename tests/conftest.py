"""Shared pytest fixtures. Tests run without env vars or Atlas access."""
from __future__ import annotations

import pytest

from core.settings import AgentContext


@pytest.fixture
def context() -> AgentContext:
    return AgentContext(realm_id="realm-test", user_id="user-test", agent_id="agent-test")
