"""Single entry point for getting the configured embedding + chat providers.

`Settings` decides which concrete class is wired in via `EMBEDDING_PROVIDER`
and `CHAT_PROVIDER` env vars. Adding a new provider means adding a branch
here plus a class file — no changes to the dozen modules that call
`get_embedding_provider()`.

For chat, `CHAT_PROVIDERS` (comma-separated) overrides `CHAT_PROVIDER`
and wraps the chain in a `FallbackChatProvider` so transient failures
on the primary fail over to the next entry.
"""
from __future__ import annotations

from functools import lru_cache

from core.protocols import ChatProvider, EmbeddingProvider
from core.settings import Settings, get_settings


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Resolve `Settings.embedding_provider` to a concrete EmbeddingProvider."""
    settings = get_settings()
    name = settings.embedding_provider
    if name == "voyage":
        from .embeddings.voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(model_name=settings.embedding_model)
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER: {name!r}")


def _build_chat_provider(name: str, settings: Settings) -> ChatProvider:
    if name == "grove":
        from .chat.grove import GroveChatProvider

        return GroveChatProvider(model_name=settings.chat_model)
    raise RuntimeError(f"Unknown chat provider: {name!r}")


@lru_cache(maxsize=1)
def get_chat_provider() -> ChatProvider:
    """Resolve the configured chat chain to a concrete ChatProvider.

    Single name (default): returns the underlying provider directly.
    Multiple names (via `CHAT_PROVIDERS`): returns a `FallbackChatProvider`
    that tries each entry in order on retryable failures.
    """
    settings = get_settings()
    chain = settings.chat_providers or (settings.chat_provider,)
    built = [(name, _build_chat_provider(name, settings)) for name in chain]
    if len(built) == 1:
        return built[0][1]
    from .chat.fallback import FallbackChatProvider

    return FallbackChatProvider(built[0], *built[1:])


def reset_provider_cache() -> None:
    """Test helper: clear the cached singletons so a new env can take effect."""
    get_embedding_provider.cache_clear()
    get_chat_provider.cache_clear()


__all__ = ["get_embedding_provider", "get_chat_provider", "reset_provider_cache"]
