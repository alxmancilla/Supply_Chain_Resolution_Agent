"""Seed the `knowledge_corpus` collection with chunked, embedded documents.

Usage:
    python -m data.seed_corpus
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from agent.memory import (
    DB_NAME,
    KNOWLEDGE_COLLECTION,
    get_embeddings,
    get_mongo_client,
    get_registry_collection,
)
from core.rag.query_planner import plan_query
from core.settings import get_settings
from data.corpus_content import DOCUMENTS

_SETTINGS = get_settings()
REALM_ID = _SETTINGS.realm_id
AGENT_ID = _SETTINGS.agent_id
EMBED_BATCH = 32


def _enrich_metadata(base: dict, paragraph: str) -> dict:
    """Merge regex-extracted lanes/carriers into the doc's static metadata.

    Lets the hybrid retriever apply per-chunk post-filters on
    `metadata.lanes` / `metadata.carriers` without re-scanning text at
    query time. The doc-level `lane` / `carrier` keys are preserved.
    """
    enriched = dict(base)
    filters = plan_query(paragraph)
    seed_lanes: set[str] = set()
    if isinstance(enriched.get("lane"), str):
        seed_lanes.add(enriched["lane"])
    seed_lanes.update(filters.lanes)
    seed_carriers: set[str] = set()
    if isinstance(enriched.get("carrier"), str):
        seed_carriers.add(enriched["carrier"])
    seed_carriers.update(filters.carriers)
    if seed_lanes:
        enriched["lanes"] = sorted(seed_lanes)
    if seed_carriers:
        enriched["carriers"] = sorted(seed_carriers)
    return enriched


def _chunks() -> list[dict]:
    """Flatten the structured docs into per-paragraph chunks ready to insert."""
    flat: list[dict] = []
    for doc in DOCUMENTS:
        for idx, paragraph in enumerate(doc["paragraphs"]):
            flat.append(
                {
                    "realm_id": REALM_ID,
                    "doc_type": doc["doc_type"],
                    "source": doc["source"],
                    "chunk_index": idx,
                    "text": paragraph,
                    "metadata": _enrich_metadata(doc["metadata"], paragraph),
                }
            )
    return flat


def _embed_all(texts: list[str]) -> list[list[float]]:
    embeddings = get_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        print(f"  embedding chunks {start + 1}-{start + len(batch)} / {len(texts)}")
        vectors.extend(embeddings.embed_documents(batch))
    return vectors


def _register_agent() -> None:
    registry = get_registry_collection()
    registry.update_one(
        {"agent_id": AGENT_ID, "realm_id": REALM_ID},
        {
            "$set": {
                "agent_id": AGENT_ID,
                "realm_id": REALM_ID,
                "version": "1.0.0",
                "description": "Meridian Supply Chain Resolution Agent demo (RAG + LTM on Atlas).",
                "status": "active",
                "registered_at": datetime.now(tz=timezone.utc),
            }
        },
        upsert=True,
    )
    print(f"[agent_registry] registered {AGENT_ID} for realm '{REALM_ID}'.")


def main() -> None:
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[KNOWLEDGE_COLLECTION]

    chunks = _chunks()
    print(f"Prepared {len(chunks)} chunks across {len(DOCUMENTS)} source documents.")

    existing = collection.count_documents({"realm_id": REALM_ID})
    if existing:
        print(f"[knowledge_corpus] removing {existing} existing chunks for realm '{REALM_ID}' before re-seeding.")
        collection.delete_many({"realm_id": REALM_ID})

    started = time.perf_counter()
    vectors = _embed_all([c["text"] for c in chunks])
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    print(f"Embedded {len(chunks)} chunks in {time.perf_counter() - started:.1f}s.")

    result = collection.insert_many(chunks)
    print(f"[knowledge_corpus] inserted {len(result.inserted_ids)} chunks for realm '{REALM_ID}'.")

    _register_agent()
    print("Seed complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
