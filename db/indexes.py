"""Create Atlas Vector Search indexes for the demo.

Run once after seeding:
    python -m db.indexes

The `agent_memories_vector` index is created automatically by `MongoDBStore`
when `agent/memory.get_store()` is first called, so this script only has to
manage `knowledge_corpus_vector` explicitly.
"""
from __future__ import annotations

import sys
import time

from dotenv import load_dotenv

load_dotenv()

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from agent.memory import (
    DB_NAME,
    EMBEDDING_DIMS,
    EPISODES_COLLECTION,
    EPISODES_VECTOR_INDEX,
    KG_CARRIERS_COLLECTION,
    KG_LANES_COLLECTION,
    KG_SERVES_COLLECTION,
    KG_SLAS_COLLECTION,
    KNOWLEDGE_COLLECTION,
    KNOWLEDGE_VECTOR_INDEX,
    MEMORIES_COLLECTION,
    MEMORIES_VECTOR_INDEX,
    get_episodes_store,
    get_mongo_client,
    get_store,
)

KNOWLEDGE_INDEX_DEFINITION = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": EMBEDDING_DIMS,
            "similarity": "cosine",
        },
        {"type": "filter", "path": "realm_id"},
        {"type": "filter", "path": "doc_type"},
        {"type": "filter", "path": "metadata.lanes"},
        {"type": "filter", "path": "metadata.carriers"},
    ]
}

KNOWLEDGE_SEARCH_INDEX_NAME = "knowledge_corpus_search"
KNOWLEDGE_SEARCH_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "text": {"type": "string", "analyzer": "lucene.standard"},
            "realm_id": {"type": "token"},
            "doc_type": {"type": "token"},
            "metadata": {
                "type": "document",
                "fields": {
                    "lanes": {"type": "token"},
                    "carriers": {"type": "token"},
                },
            },
        },
    }
}


def _index_exists(collection, name: str) -> bool:
    return any(ix.get("name") == name for ix in collection.list_search_indexes())


def ensure_knowledge_index() -> None:
    client = get_mongo_client()
    db = client[DB_NAME]
    if KNOWLEDGE_COLLECTION not in db.list_collection_names():
        db.create_collection(KNOWLEDGE_COLLECTION)
    collection = db[KNOWLEDGE_COLLECTION]

    if _index_exists(collection, KNOWLEDGE_VECTOR_INDEX):
        print(f"[knowledge_corpus] '{KNOWLEDGE_VECTOR_INDEX}' already exists — skipping.")
        return

    model = SearchIndexModel(
        definition=KNOWLEDGE_INDEX_DEFINITION,
        name=KNOWLEDGE_VECTOR_INDEX,
        type="vectorSearch",
    )
    try:
        collection.create_search_index(model=model)
    except OperationFailure as exc:
        print(f"[knowledge_corpus] create_search_index failed: {exc}", file=sys.stderr)
        raise

    print(f"[knowledge_corpus] '{KNOWLEDGE_VECTOR_INDEX}' submitted. Waiting for it to become queryable...")
    deadline = time.time() + 600
    while time.time() < deadline:
        for ix in collection.list_search_indexes():
            if ix.get("name") == KNOWLEDGE_VECTOR_INDEX and ix.get("queryable"):
                print(f"[knowledge_corpus] '{KNOWLEDGE_VECTOR_INDEX}' is queryable.")
                return
        time.sleep(5)
    print(f"[knowledge_corpus] WARNING: '{KNOWLEDGE_VECTOR_INDEX}' did not become queryable within 10 minutes.")


def ensure_knowledge_search_index() -> None:
    """Create the BM25 Atlas Search index used by the hybrid retriever."""
    client = get_mongo_client()
    db = client[DB_NAME]
    if KNOWLEDGE_COLLECTION not in db.list_collection_names():
        db.create_collection(KNOWLEDGE_COLLECTION)
    collection = db[KNOWLEDGE_COLLECTION]

    if _index_exists(collection, KNOWLEDGE_SEARCH_INDEX_NAME):
        print(f"[knowledge_corpus] '{KNOWLEDGE_SEARCH_INDEX_NAME}' already exists — skipping.")
        return

    model = SearchIndexModel(
        definition=KNOWLEDGE_SEARCH_INDEX_DEFINITION,
        name=KNOWLEDGE_SEARCH_INDEX_NAME,
        type="search",
    )
    try:
        collection.create_search_index(model=model)
    except OperationFailure as exc:
        print(f"[knowledge_corpus] create_search_index (BM25) failed: {exc}", file=sys.stderr)
        raise

    print(f"[knowledge_corpus] '{KNOWLEDGE_SEARCH_INDEX_NAME}' submitted. Waiting for it to become queryable...")
    deadline = time.time() + 600
    while time.time() < deadline:
        for ix in collection.list_search_indexes():
            if ix.get("name") == KNOWLEDGE_SEARCH_INDEX_NAME and ix.get("queryable"):
                print(f"[knowledge_corpus] '{KNOWLEDGE_SEARCH_INDEX_NAME}' is queryable.")
                return
        time.sleep(5)
    print(f"[knowledge_corpus] WARNING: '{KNOWLEDGE_SEARCH_INDEX_NAME}' did not become queryable within 10 minutes.")


def ensure_memories_index() -> None:
    """Trigger MongoDBStore's built-in vector index creation."""
    print(f"[agent_memories] Initializing MongoDBStore (auto-creates '{MEMORIES_VECTOR_INDEX}')...")
    get_store()
    collection = get_mongo_client()[DB_NAME][MEMORIES_COLLECTION]
    for ix in collection.list_search_indexes():
        if ix.get("name") == MEMORIES_VECTOR_INDEX:
            status = "queryable" if ix.get("queryable") else ix.get("status", "pending")
            print(f"[agent_memories] '{MEMORIES_VECTOR_INDEX}' status: {status}")
            return
    print(f"[agent_memories] '{MEMORIES_VECTOR_INDEX}' was not detected — check MongoDBStore logs.")


def ensure_episodes_index() -> None:
    """Trigger MongoDBStore's built-in vector index creation for episodic LTM."""
    print(f"[agent_episodes] Initializing MongoDBStore (auto-creates '{EPISODES_VECTOR_INDEX}')...")
    get_episodes_store()
    collection = get_mongo_client()[DB_NAME][EPISODES_COLLECTION]
    for ix in collection.list_search_indexes():
        if ix.get("name") == EPISODES_VECTOR_INDEX:
            status = "queryable" if ix.get("queryable") else ix.get("status", "pending")
            print(f"[agent_episodes] '{EPISODES_VECTOR_INDEX}' status: {status}")
            return
    print(f"[agent_episodes] '{EPISODES_VECTOR_INDEX}' was not detected — check MongoDBStore logs.")


def ensure_kg_indexes() -> None:
    """Provision the four KG collections + the b-tree indexes that back
    the `$graphLookup` + `$lookup` joins in core/kg/mongo.py."""
    db = get_mongo_client()[DB_NAME]
    plan = {
        KG_CARRIERS_COLLECTION: [("realm_id", 1), ("carrier_id", 1)],
        KG_LANES_COLLECTION: [("realm_id", 1), ("lane_id", 1)],
        KG_SLAS_COLLECTION: [("realm_id", 1), ("carrier_id", 1), ("lane_id", 1)],
        KG_SERVES_COLLECTION: [("realm_id", 1), ("lane_id", 1), ("carrier_id", 1)],
    }
    for coll_name, keys in plan.items():
        if coll_name not in db.list_collection_names():
            db.create_collection(coll_name)
        coll = db[coll_name]
        idx_name = "_".join(f"{k}{d}" for k, d in keys)
        existing = {ix["name"] for ix in coll.list_indexes()}
        if idx_name in existing:
            print(f"[{coll_name}] '{idx_name}' already exists — skipping.")
            continue
        coll.create_index(keys, name=idx_name)
        print(f"[{coll_name}] created index '{idx_name}'.")


def main() -> None:
    ensure_knowledge_index()
    ensure_knowledge_search_index()
    ensure_memories_index()
    ensure_episodes_index()
    ensure_kg_indexes()


if __name__ == "__main__":
    main()
