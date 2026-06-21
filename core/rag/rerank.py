"""Reranker implementations of `core.protocols.Reranker`.

`NullReranker` is the no-op identity reranker used when reranking is
disabled or in unit tests. `VoyageReranker` wraps the Voyage AI rerank
endpoint and is the production default once `RAG_RERANK_ENABLED=1`.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from core.schemas import KnowledgeHit


class NullReranker:
    """Identity reranker — preserves input order, trims to `top_k`."""

    def rerank(self, query: str, hits: Sequence[KnowledgeHit], *, top_k: int) -> Sequence[KnowledgeHit]:
        return list(hits)[:top_k]


class VoyageReranker:
    """Cross-encoder reranker backed by `voyageai.Client.rerank`.

    The Voyage rerank score replaces the upstream fusion score so callers
    can sort by `KnowledgeHit.score` consistently downstream.
    """

    def __init__(self, *, model: str = "rerank-2-lite", client: Any | None = None) -> None:
        self._model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "VOYAGE_API_KEY is required when RAG_RERANK_ENABLED=1."
                )
            import voyageai

            self._client = voyageai.Client(api_key=api_key)
        return self._client

    def rerank(self, query: str, hits: Sequence[KnowledgeHit], *, top_k: int) -> Sequence[KnowledgeHit]:
        if not hits:
            return []
        documents = [h.text for h in hits]
        result = self._get_client().rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=min(top_k, len(documents)),
        )
        reranked: list[KnowledgeHit] = []
        for entry in result.results:
            original = hits[entry.index]
            data = original.model_dump()
            data["score"] = float(entry.relevance_score)
            reranked.append(KnowledgeHit(**data))
        return reranked


__all__ = ["NullReranker", "VoyageReranker"]
