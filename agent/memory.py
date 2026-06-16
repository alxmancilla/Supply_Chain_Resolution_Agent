"""Shared MongoDB connections for the Supply Chain Resolution Agent.

One Atlas cluster serves five concurrent workloads:
  * `checkpoints`       — short-term memory via `MongoDBSaver`
  * `agent_memories`    — semantic LTM via `MongoDBStore` (vector search)
  * `agent_episodes`    — episodic LTM via `MongoDBStore` (vector search)
  * `agent_procedures`  — procedural LTM via raw pymongo (rule list)
  * `knowledge_corpus`  — RAG corpus via raw `$vectorSearch` aggregation
"""
from __future__ import annotations

import os
from functools import lru_cache

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from pymongo import MongoClient

from core.providers.embeddings._langchain_adapter import to_langchain_embeddings
from core.providers.registry import get_embedding_provider

DB_NAME = "meridian_genai"
KNOWLEDGE_COLLECTION = "knowledge_corpus"
MEMORIES_COLLECTION = "agent_memories"
EPISODES_COLLECTION = "agent_episodes"
PROCEDURES_COLLECTION = "agent_procedures"
CHECKPOINTS_COLLECTION = "checkpoints"
REGISTRY_COLLECTION = "agent_registry"
KG_CARRIERS_COLLECTION = "kg_carriers"
KG_LANES_COLLECTION = "kg_lanes"
KG_SLAS_COLLECTION = "kg_slas"
KG_SERVES_COLLECTION = "kg_serves"
BOOKING_DRAFTS_COLLECTION = "booking_drafts"
PROCEDURE_PROPOSALS_COLLECTION = "procedure_proposals"

KNOWLEDGE_VECTOR_INDEX = "knowledge_corpus_vector"
MEMORIES_VECTOR_INDEX = "agent_memories_vector"
EPISODES_VECTOR_INDEX = "agent_episodes_vector"

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or .env file."
        )
    return value


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    """Single shared `MongoClient` for all workloads (proving the one-cluster story)."""
    uri = _require_env("MONGODB_URI")
    return MongoClient(uri)


@lru_cache(maxsize=1)
def get_embeddings():
    """Langchain-style `Embeddings` wrapper around the configured EmbeddingProvider.

    Kept for backward-compat with `MongoDBStore` and seed scripts that take a
    langchain `Embeddings`. New code should depend on
    `core.providers.registry.get_embedding_provider()` directly.
    """
    return to_langchain_embeddings(get_embedding_provider())


EMBEDDING_DIMS = 1024


def _embedding_dims() -> int:
    """Live dimensions from the active provider (preferred for index config)."""
    return get_embedding_provider().dimensions


def _assert_vector_index_dims(collection, index_name: str, expected_dims: int) -> None:
    """Fail fast if an existing Atlas vector index disagrees with the active provider.

    A silent dim mismatch (e.g. swapping `voyage-4` (1024) for `text-embedding-3-large`
    (3072) against a pre-populated collection) returns zero-recall queries; this
    catches it at boot rather than at retrieval time. No-op when the index does
    not yet exist (it will be created with the correct dims) or when the cluster
    does not support `listSearchIndexes`.
    """
    try:
        cursor = collection.list_search_indexes(name=index_name)
        entries = list(cursor)
    except Exception:
        return
    if not entries:
        return
    definition = entries[0].get("latestDefinition") or {}
    for field in definition.get("fields", []):
        if field.get("type") != "vector":
            continue
        actual = field.get("numDimensions")
        if actual is not None and actual != expected_dims:
            raise RuntimeError(
                f"Vector index '{index_name}' on collection "
                f"'{collection.name}' expects {actual} dims but the active "
                f"embedding provider ({get_embedding_provider().model_name}) "
                f"produces {expected_dims}. Drop the index or switch providers "
                f"to match before continuing."
            )


@lru_cache(maxsize=1)
def get_checkpointer() -> MongoDBSaver:
    """Short-term memory: `MongoDBSaver` wired into the compiled LangGraph."""
    return MongoDBSaver(
        client=get_mongo_client(),
        db_name=DB_NAME,
        checkpoint_collection_name=CHECKPOINTS_COLLECTION,
    )


@lru_cache(maxsize=1)
def get_store() -> MongoDBStore:
    """Long-term memory: `MongoDBStore` with Voyage-backed vector search.

    The store auto-creates an Atlas Vector Search index named
    `agent_memories_vector` on the `value.content` field the first time it
    runs against the collection.
    """
    dims = _embedding_dims()
    index_config = create_vector_index_config(
        embed=get_embeddings(),
        dims=dims,
        fields=["content"],
        filters=["namespace"],
        name=MEMORIES_VECTOR_INDEX,
        relevance_score_fn="cosine",
    )
    client = get_mongo_client()
    db = client[DB_NAME]
    if MEMORIES_COLLECTION not in db.list_collection_names():
        db.create_collection(MEMORIES_COLLECTION)
    _assert_vector_index_dims(db[MEMORIES_COLLECTION], MEMORIES_VECTOR_INDEX, dims)
    return MongoDBStore(
        collection=db[MEMORIES_COLLECTION],
        index_config=index_config,
        auto_index_timeout=70,
    )


@lru_cache(maxsize=1)
def get_episodes_store() -> MongoDBStore:
    """Episodic LTM: structured past events with vector-searchable summaries."""
    dims = _embedding_dims()
    index_config = create_vector_index_config(
        embed=get_embeddings(),
        dims=dims,
        fields=["summary"],
        filters=["namespace"],
        name=EPISODES_VECTOR_INDEX,
        relevance_score_fn="cosine",
    )
    client = get_mongo_client()
    db = client[DB_NAME]
    if EPISODES_COLLECTION not in db.list_collection_names():
        db.create_collection(EPISODES_COLLECTION)
    _assert_vector_index_dims(db[EPISODES_COLLECTION], EPISODES_VECTOR_INDEX, dims)
    return MongoDBStore(
        collection=db[EPISODES_COLLECTION],
        index_config=index_config,
        auto_index_timeout=70,
    )


def get_knowledge_collection():
    """Raw collection handle for RAG `$vectorSearch` queries."""
    return get_mongo_client()[DB_NAME][KNOWLEDGE_COLLECTION]


def get_procedures_collection():
    """Procedural LTM: learned/curated rules injected into the system prompt."""
    return get_mongo_client()[DB_NAME][PROCEDURES_COLLECTION]


def get_registry_collection():
    """Operational agent registry collection."""
    return get_mongo_client()[DB_NAME][REGISTRY_COLLECTION]


def get_kg_collections():
    """Return the four KG collections (carriers, lanes, slas, serves)."""
    db = get_mongo_client()[DB_NAME]
    return (
        db[KG_CARRIERS_COLLECTION],
        db[KG_LANES_COLLECTION],
        db[KG_SLAS_COLLECTION],
        db[KG_SERVES_COLLECTION],
    )


def get_booking_drafts_collection():
    """Drafts of proposed bookings (pending_approval → approved → executed)."""
    return get_mongo_client()[DB_NAME][BOOKING_DRAFTS_COLLECTION]


def get_procedure_proposals_collection():
    """Staging area for agent-proposed procedural rules (pending_approval → approved | rejected)."""
    return get_mongo_client()[DB_NAME][PROCEDURE_PROPOSALS_COLLECTION]


def memory_namespace(realm_id: str, user_id: str) -> tuple[str, str]:
    """LTM namespace tuple enforcing tenant + user isolation."""
    return (realm_id, user_id)
