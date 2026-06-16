"""MongoDB-backed knowledge retriever for the RAG layer.

Wraps a `$vectorSearch` aggregation against the corpus collection and
returns typed `KnowledgeHit`s. Implements `core.protocols.KnowledgeRetriever`
so graph nodes depend only on the protocol, not on pymongo.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from core.schemas import KnowledgeHit


class MongoKnowledgeRetriever:
    """`$vectorSearch` retriever over the shared knowledge corpus collection."""

    def __init__(
        self,
        *,
        collection: Any,
        embeddings: Any,
        index_name: str,
        num_candidates: int = 100,
    ) -> None:
        self._collection = collection
        self._embeddings = embeddings
        self._index_name = index_name
        self._num_candidates = num_candidates

    def query(self, realm_id: str, text: str, k: int) -> Sequence[KnowledgeHit]:
        query_vector = self._embeddings.embed_query(text)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": self._num_candidates,
                    "limit": k,
                    "filter": {"realm_id": realm_id},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "doc_type": 1,
                    "source": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return [KnowledgeHit(**doc) for doc in self._collection.aggregate(pipeline)]


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

    collection = get_knowledge_collection()
    _assert_vector_index_dims(collection, KNOWLEDGE_VECTOR_INDEX, _embedding_dims())
    return MongoKnowledgeRetriever(
        collection=collection,
        embeddings=get_embeddings(),
        index_name=KNOWLEDGE_VECTOR_INDEX,
    )


__all__ = ["MongoKnowledgeRetriever", "get_knowledge_retriever"]
