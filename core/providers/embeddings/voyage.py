"""Voyage AI implementation of `core.protocols.EmbeddingProvider`.

Uses the official `voyageai` client directly. The `input_type` hint is
set per call (`"query"` vs `"document"`) so Voyage can apply the
retrieval-tuned variant for each side of the search.
"""
from __future__ import annotations

import os
from typing import Sequence


class VoyageEmbeddingProvider:
    """Implements `core.protocols.EmbeddingProvider` via `voyage-4` (1024 dim)."""

    model_name: str = "voyage-4"
    dimensions: int = 1024

    def __init__(self, *, model_name: str | None = None, dimensions: int | None = None) -> None:
        if model_name is not None:
            self.model_name = model_name
        if dimensions is not None:
            self.dimensions = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "VOYAGE_API_KEY is required when EMBEDDING_PROVIDER=voyage."
                )
            import voyageai

            self._client = voyageai.Client(api_key=api_key)
        return self._client

    def embed_query(self, text: str) -> list[float]:
        result = self._get_client().embed([text], model=self.model_name, input_type="query")
        return list(result.embeddings[0])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._get_client().embed(list(texts), model=self.model_name, input_type="document")
        return [list(v) for v in result.embeddings]


__all__ = ["VoyageEmbeddingProvider"]
