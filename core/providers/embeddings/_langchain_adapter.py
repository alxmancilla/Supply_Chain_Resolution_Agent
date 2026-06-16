"""Adapter wrapping an EmbeddingProvider in langchain-core's `Embeddings` interface.

`MongoDBStore` and the seed scripts depend on the langchain `Embeddings`
abstract base class. The adapter lets us keep that dependency limited to
the wiring layer while feature code only sees `EmbeddingProvider`.
"""
from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from core.protocols import EmbeddingProvider


class LangChainEmbeddingsAdapter(Embeddings):
    """Wraps a `core.protocols.EmbeddingProvider` as a langchain `Embeddings`."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._provider.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:  # pragma: no cover
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return self.embed_documents(texts)


def to_langchain_embeddings(provider: EmbeddingProvider) -> Any:
    """Convenience: wrap any EmbeddingProvider as a langchain Embeddings."""
    return LangChainEmbeddingsAdapter(provider)


__all__ = ["LangChainEmbeddingsAdapter", "to_langchain_embeddings"]
