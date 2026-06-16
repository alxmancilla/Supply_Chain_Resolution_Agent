"""Single entry point for getting the configured embedding + chat providers.

`Settings` decides which concrete class is wired in via `EMBEDDING_PROVIDER`
and `CHAT_PROVIDER` env vars. Adding a new provider means adding a branch
here plus a class file — no changes to the dozen modules that call
`get_embedding_provider()`.
"""
from __future__ import annotations

from functools import lru_cache

from core.protocols import ChatProvider, EmbeddingProvider
from core.settings import get_settings


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Resolve `Settings.embedding_provider` to a concrete EmbeddingProvider."""
    settings = get_settings()
    name = settings.embedding_provider
    if name == "voyage":
        from .embeddings.voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(model_name=settings.embedding_model)
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER: {name!r}")


@lru_cache(maxsize=1)
def get_chat_provider() -> ChatProvider:
    """Resolve `Settings.chat_provider` to a concrete ChatProvider."""
    settings = get_settings()
    name = settings.chat_provider
    if name == "grove":
        from .chat.grove import GroveChatProvider

        return GroveChatProvider(model_name=settings.chat_model)
    raise RuntimeError(f"Unknown CHAT_PROVIDER: {name!r}")


def reset_provider_cache() -> None:
    """Test helper: clear the cached singletons so a new env can take effect."""
    get_embedding_provider.cache_clear()
    get_chat_provider.cache_clear()


__all__ = ["get_embedding_provider", "get_chat_provider", "reset_provider_cache"]
