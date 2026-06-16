"""Centralized configuration and typed agent context.

`Settings` reads env vars exactly once. `AgentContext` is the typed
working-memory cell threaded through the graph state — replaces the
scattered `realm_id` / `user_id` / `AGENT_ID` lookups previously done
in every node and seed script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from uuid import uuid4

EMBEDDING_PROVIDERS = ("voyage",)
CHAT_PROVIDERS = ("grove",)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or .env file."
        )
    return value


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).lower()
    if value not in allowed:
        raise RuntimeError(
            f"{name}={value!r} not in {allowed}. Set one of those values "
            f"or unset {name} to use the default '{default}'."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Process-wide configuration. Loaded once from env."""

    mongodb_uri: str
    realm_id: str = "customer-tenant-001"
    user_id: str = "user-demo"
    agent_id: str = "supply-chain-resolution-agent"
    embedding_provider: str = "voyage"
    chat_provider: str = "grove"
    embedding_model: str = "voyage-4"
    chat_model: str = "gpt-5.5"
    semantic_dedup_threshold: float = 0.92
    episodic_dedup_threshold: float = 0.92
    chat_input_price_per_1k_usd: float = 0.0
    chat_output_price_per_1k_usd: float = 0.0
    reflect_every_n_turns: int = 0
    reflect_threshold: float = 0.88


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build a `Settings` from environment variables (cached).

    Provider-specific API keys are pulled lazily by the providers themselves
    (`core/providers/registry.py`) so a process that only uses one provider
    never has to set the other's key.
    """
    return Settings(
        mongodb_uri=_require_env("MONGODB_URI"),
        realm_id=os.environ.get("REALM_ID", "customer-tenant-001"),
        user_id=os.environ.get("USER_ID", "user-demo"),
        agent_id=os.environ.get("AGENT_ID", "supply-chain-resolution-agent"),
        embedding_provider=_env_choice("EMBEDDING_PROVIDER", "voyage", EMBEDDING_PROVIDERS),
        chat_provider=_env_choice("CHAT_PROVIDER", "grove", CHAT_PROVIDERS),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "voyage-4"),
        chat_model=os.environ.get("CHAT_MODEL", "gpt-5.5"),
        semantic_dedup_threshold=float(os.environ.get("SEMANTIC_DEDUP_THRESHOLD", "0.92")),
        episodic_dedup_threshold=float(os.environ.get("EPISODIC_DEDUP_THRESHOLD", "0.92")),
        chat_input_price_per_1k_usd=float(os.environ.get("CHAT_INPUT_PRICE_PER_1K_USD", "0.0")),
        chat_output_price_per_1k_usd=float(os.environ.get("CHAT_OUTPUT_PRICE_PER_1K_USD", "0.0")),
        reflect_every_n_turns=int(os.environ.get("REFLECT_EVERY_N_TURNS", "0")),
        reflect_threshold=float(os.environ.get("REFLECT_THRESHOLD", "0.88")),
    )


@dataclass(frozen=True)
class AgentContext:
    """Per-invocation identity passed through the graph state.

    Frozen dataclass so it round-trips cleanly through the MongoDB
    checkpointer's serializer. Keep it small and string-only. The
    `correlation_id` is auto-generated when omitted and propagated to
    every OTel span so traces from one user turn can be correlated.
    """

    realm_id: str
    user_id: str
    agent_id: str
    correlation_id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AgentContext":
        s = settings or get_settings()
        return cls(realm_id=s.realm_id, user_id=s.user_id, agent_id=s.agent_id)


__all__ = ["Settings", "AgentContext", "get_settings"]
