"""Periodic consolidation pass over a (semantic | episodic) LTM namespace.

`LLMMemoryReflector` clusters live facts by cosine similarity on the stored
embeddings, asks the configured `ChatProvider` to merge each cluster into a
single canonical fact, writes the canonical row, and tombstones the
originals so they fall out of subsequent retrievals (the memory layer's
`search` already filters `tombstoned: True`).

The reflector is opt-in maintenance — invoked from `tools/reflect.py`, not
on the per-turn hot path. It depends on a `MemoryAdmin` mini-protocol so
the same logic exercises both the live Atlas-backed admin and a no-infra
fake in unit tests.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from agent.prompts import MEMORY_CONSOLIDATION_PROMPT
from core.protocols import ChatProvider, EmbeddingProvider


class StoredFact(BaseModel):
    """Raw fact pulled by `MemoryAdmin.list_live` for clustering."""
    key: str
    content: str
    embedding: list[float] = Field(default_factory=list)


class ReflectionReport(BaseModel):
    """Summary returned by `LLMMemoryReflector.reflect`."""
    clusters_found: int = 0
    canonical_written: list[str] = Field(default_factory=list)
    tombstoned_keys: list[str] = Field(default_factory=list)
    skipped_singletons: int = 0


class _CanonicalFact(BaseModel):
    canonical: str


@runtime_checkable
class MemoryAdmin(Protocol):
    """Maintenance surface for a single LTM namespace."""

    def list_live(self, realm_id: str, user_id: str) -> Sequence[StoredFact]: ...
    def tombstone(self, realm_id: str, user_id: str, keys: Sequence[str]) -> None: ...
    def write_canonical(
        self, realm_id: str, user_id: str, content: str, source_keys: Sequence[str]
    ) -> str: ...


class MongoMemoryAdmin:
    """`MemoryAdmin` implementation against a langgraph `MongoDBStore` collection.

    Reads the embedding already persisted on every doc (no re-embedding for
    clustering). `write_canonical` does call the embedding provider so the
    new canonical row is itself vector-searchable.
    """

    def __init__(self, *, collection: Any, content_field: str, embeddings: EmbeddingProvider) -> None:
        self._coll = collection
        self._field = content_field
        self._embeddings = embeddings

    def list_live(self, realm_id: str, user_id: str) -> list[StoredFact]:
        cursor = self._coll.find({
            "namespace": [realm_id, user_id],
            "value.tombstoned": {"$ne": True},
        })
        out: list[StoredFact] = []
        for d in cursor:
            value = d.get("value", {}) or {}
            content = value.get(self._field, "")
            if not content:
                continue
            out.append(StoredFact(key=d["key"], content=content, embedding=d.get("embedding", []) or []))
        return out

    def tombstone(self, realm_id: str, user_id: str, keys: Sequence[str]) -> None:
        if not keys:
            return
        now = datetime.now(timezone.utc)
        self._coll.update_many(
            {"namespace": [realm_id, user_id], "key": {"$in": list(keys)}},
            {"$set": {"value.tombstoned": True, "tombstoned_at": now}},
        )

    def write_canonical(
        self, realm_id: str, user_id: str, content: str, source_keys: Sequence[str]
    ) -> str:
        key = f"canon_{hashlib.sha1(content.encode('utf-8')).hexdigest()[:16]}"
        now = datetime.now(timezone.utc)
        value = {
            self._field: content,
            "canonical": True,
            "consolidated_from": list(source_keys),
            "seen_count": max(len(source_keys), 1),
        }
        doc = {
            "namespace": [realm_id, user_id],
            "namespace_str": f"{realm_id}/{user_id}",
            "namespace_prefix": [realm_id, f"{realm_id}/{user_id}"],
            "key": key,
            "value": value,
            "embedding": self._embeddings.embed_query(content),
            "updated_at": now,
        }
        self._coll.update_one(
            {"namespace": [realm_id, user_id], "key": key},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return key


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _greedy_cluster(facts: Sequence[StoredFact], threshold: float) -> list[list[StoredFact]]:
    """Single-link greedy clustering by cosine similarity on the embeddings."""
    pool = list(facts)
    clusters: list[list[StoredFact]] = []
    while pool:
        seed = pool.pop(0)
        members = [seed]
        leftover: list[StoredFact] = []
        for cand in pool:
            if _cosine(seed.embedding, cand.embedding) >= threshold:
                members.append(cand)
            else:
                leftover.append(cand)
        clusters.append(members)
        pool = leftover
    return clusters


class LLMMemoryReflector:
    """Implements `core.protocols.MemoryReflector` via clustering + LLM consolidation."""

    def __init__(
        self,
        *,
        admin: MemoryAdmin,
        chat: ChatProvider,
        similarity_threshold: float = 0.88,
    ) -> None:
        self._admin = admin
        self._chat = chat
        self._threshold = similarity_threshold

    def reflect(self, realm_id: str, user_id: str) -> ReflectionReport:
        facts = list(self._admin.list_live(realm_id, user_id))
        clusters = _greedy_cluster(facts, self._threshold)
        report = ReflectionReport()
        for cluster in clusters:
            if len(cluster) < 2:
                report.skipped_singletons += 1
                continue
            report.clusters_found += 1
            facts_text = "\n".join(f"- {c.content}" for c in cluster)
            merged = self._chat.invoke_typed(
                MEMORY_CONSOLIDATION_PROMPT.format(facts=facts_text), _CanonicalFact
            )
            source_keys = [c.key for c in cluster]
            canonical_key = self._admin.write_canonical(
                realm_id, user_id, merged.canonical, source_keys
            )
            report.canonical_written.append(canonical_key)
            self._admin.tombstone(realm_id, user_id, source_keys)
            report.tombstoned_keys.extend(source_keys)
        return report


__all__ = [
    "StoredFact",
    "ReflectionReport",
    "MemoryAdmin",
    "MongoMemoryAdmin",
    "LLMMemoryReflector",
]
