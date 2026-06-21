"""MongoDB-backed knowledge retriever for the RAG layer.

Supports two retrieval modes behind a single `KnowledgeRetriever` surface:

* Vector-only (default): `$vectorSearch` over the corpus embedding field.
* Hybrid: vector + BM25 (`$search`), fused with weighted reciprocal-rank
  fusion (RRF) and optionally reranked by a cross-encoder.

In both modes the heuristic `query_planner` extracts lanes / carriers /
doc_types from the query and applies them as a post-filter (with graceful
fallback to unfiltered hits on zero recall).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from core.protocols import Reranker
from core.rag.query_planner import plan_query
from core.rag.rerank import NullReranker
from core.schemas import KnowledgeHit, RagQueryFilters

_RRF_K = 60  # RRF dampening constant; matches the canonical Cormack/Buettcher value.


class MongoKnowledgeRetriever:
    """Hybrid `$vectorSearch` + `$search` retriever with optional rerank."""

    def __init__(
        self,
        *,
        collection: Any,
        embeddings: Any,
        index_name: str,
        num_candidates: int = 100,
        search_index_name: str | None = None,
        hybrid_enabled: bool = False,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        fusion_candidates: int = 20,
        reranker: Reranker | None = None,
        filter_planner: Any = plan_query,
    ) -> None:
        self._collection = collection
        self._embeddings = embeddings
        self._index_name = index_name
        self._num_candidates = num_candidates
        self._search_index_name = search_index_name
        self._hybrid_enabled = hybrid_enabled and bool(search_index_name)
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._fusion_candidates = fusion_candidates
        self._reranker: Reranker = reranker or NullReranker()
        self._filter_planner = filter_planner

    def query(self, realm_id: str, text: str, k: int) -> Sequence[KnowledgeHit]:
        filters = self._filter_planner(text) if self._filter_planner else RagQueryFilters()
        candidate_k = max(k, self._fusion_candidates) if self._hybrid_enabled or self._reranker is not None else k
        vector_hits = self._vector_search(realm_id, text, candidate_k, filters)
        if self._hybrid_enabled:
            bm25_hits = self._bm25_search(realm_id, text, candidate_k)
            fused = _rrf_fuse(vector_hits, bm25_hits, self._vector_weight, self._bm25_weight)
        else:
            fused = vector_hits
        narrowed = _apply_post_filter(fused, filters) or fused
        return list(self._reranker.rerank(text, narrowed, top_k=k))

    def _vector_search(self, realm_id: str, text: str, k: int, filters: RagQueryFilters) -> list[KnowledgeHit]:
        pre_filter: dict[str, Any] = {"realm_id": realm_id}
        if filters.doc_types:
            pre_filter["doc_type"] = {"$in": list(filters.doc_types)}
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": "embedding",
                    "queryVector": self._embeddings.embed_query(text),
                    "numCandidates": self._num_candidates,
                    "limit": k,
                    "filter": pre_filter,
                }
            },
            {"$project": _PROJECT_STAGE},
        ]
        return [KnowledgeHit(**doc) for doc in self._collection.aggregate(pipeline)]

    def _bm25_search(self, realm_id: str, text: str, k: int) -> list[KnowledgeHit]:
        pipeline = [
            {
                "$search": {
                    "index": self._search_index_name,
                    "compound": {
                        "must": [{"text": {"query": text, "path": "text"}}],
                        "filter": [{"equals": {"path": "realm_id", "value": realm_id}}],
                    },
                }
            },
            {"$limit": k},
            {"$project": {**_PROJECT_STAGE, "score": {"$meta": "searchScore"}}},
        ]
        return [KnowledgeHit(**doc) for doc in self._collection.aggregate(pipeline)]


_PROJECT_STAGE = {
    "_id": 0,
    "doc_type": 1,
    "source": 1,
    "chunk_index": 1,
    "text": 1,
    "metadata": 1,
    "score": {"$meta": "vectorSearchScore"},
}


def _hit_key(hit: KnowledgeHit) -> tuple[str, Any]:
    extra = hit.model_dump()
    return (hit.source, extra.get("chunk_index", hit.text[:64]))


def _rrf_fuse(
    vector_hits: Sequence[KnowledgeHit],
    bm25_hits: Sequence[KnowledgeHit],
    vector_weight: float,
    bm25_weight: float,
) -> list[KnowledgeHit]:
    """Weighted reciprocal-rank fusion across the two candidate lists."""
    scores: dict[tuple[str, Any], float] = {}
    seen: dict[tuple[str, Any], KnowledgeHit] = {}
    for rank, hit in enumerate(vector_hits):
        key = _hit_key(hit)
        scores[key] = scores.get(key, 0.0) + vector_weight / (_RRF_K + rank + 1)
        seen.setdefault(key, hit)
    for rank, hit in enumerate(bm25_hits):
        key = _hit_key(hit)
        scores[key] = scores.get(key, 0.0) + bm25_weight / (_RRF_K + rank + 1)
        seen.setdefault(key, hit)
    fused: list[KnowledgeHit] = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        base = seen[key].model_dump()
        base["score"] = float(score)
        fused.append(KnowledgeHit(**base))
    return fused


def _apply_post_filter(hits: Sequence[KnowledgeHit], filters: RagQueryFilters) -> list[KnowledgeHit]:
    """AND-filter on lanes/carriers using `metadata.lanes` / `metadata.carriers`.

    Returns an empty list on zero match so the caller can fall back to the
    unfiltered candidate set rather than starving the response.
    """
    if not filters.lanes and not filters.carriers:
        return list(hits)
    kept: list[KnowledgeHit] = []
    for hit in hits:
        meta = hit.metadata or {}
        lanes = {*_listify(meta.get("lanes")), *_listify(meta.get("lane"))}
        carriers = {*_listify(meta.get("carriers")), *_listify(meta.get("carrier"))}
        if filters.lanes and not (set(filters.lanes) & lanes):
            continue
        if filters.carriers and not (set(filters.carriers) & carriers):
            continue
        kept.append(hit)
    return kept


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


@lru_cache(maxsize=1)
def get_knowledge_retriever() -> MongoKnowledgeRetriever:
    """Process-wide default retriever wired to the shared Atlas client."""
    from agent.memory import (
        KNOWLEDGE_VECTOR_INDEX,
        _assert_vector_index_dims,
        _embedding_dims,
        get_embeddings,
        get_knowledge_collection,
    )
    from core.settings import get_settings

    settings = get_settings()
    collection = get_knowledge_collection()
    _assert_vector_index_dims(collection, KNOWLEDGE_VECTOR_INDEX, _embedding_dims())
    reranker: Reranker | None = None
    if settings.rag_rerank_enabled:
        from core.rag.rerank import VoyageReranker

        reranker = VoyageReranker(model=settings.rag_rerank_model)
    return MongoKnowledgeRetriever(
        collection=collection,
        embeddings=get_embeddings(),
        index_name=KNOWLEDGE_VECTOR_INDEX,
        search_index_name=settings.rag_search_index_name,
        hybrid_enabled=settings.rag_hybrid_enabled,
        vector_weight=settings.rag_vector_weight,
        bm25_weight=settings.rag_bm25_weight,
        fusion_candidates=settings.rag_fusion_candidates,
        reranker=reranker,
    )


__all__ = ["MongoKnowledgeRetriever", "get_knowledge_retriever"]
