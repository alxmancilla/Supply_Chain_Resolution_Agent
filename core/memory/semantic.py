"""MongoDB-backed semantic LTM (durable per-user facts, vector-retrieved)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from core.schemas import SemanticFact


class MongoSemanticMemory:
    """Implements `core.protocols.SemanticMemory` over a `MongoDBStore`.

    `put` is dedup-aware: it probes the top-1 nearest neighbour and, if the
    cosine similarity meets `dedup_threshold`, bumps `seen_count` on the
    existing row instead of inserting a near-duplicate. `search` filters out
    rows flagged `tombstoned: True` (set by the reflection pass).
    """

    def __init__(self, *, store: Any, dedup_threshold: float = 0.92) -> None:
        self._store = store
        self._dedup_threshold = dedup_threshold

    @staticmethod
    def _namespace(realm_id: str, user_id: str) -> tuple[str, str]:
        return (realm_id, user_id)

    def search(self, realm_id: str, user_id: str, query: str, limit: int) -> Sequence[SemanticFact]:
        results = self._store.search(self._namespace(realm_id, user_id), query=query, limit=limit * 2)
        facts: list[SemanticFact] = []
        for item in results:
            value = item.value if isinstance(item.value, dict) else {}
            if value.get("tombstoned"):
                continue
            content = value.get("content", "") if isinstance(item.value, dict) else str(item.value)
            facts.append(
                SemanticFact(
                    key=item.key,
                    content=content,
                    score=item.score,
                    updated_at=item.updated_at,
                )
            )
            if len(facts) >= limit:
                break
        return facts

    def put(self, realm_id: str, user_id: str, key: str, content: str) -> None:
        namespace = self._namespace(realm_id, user_id)
        for item in self._store.search(namespace, query=content, limit=5):
            value = item.value if isinstance(item.value, dict) else {}
            if value.get("tombstoned"):
                continue
            if (item.score or 0.0) < self._dedup_threshold:
                break
            seen_count = int(value.get("seen_count", 1)) + 1
            merged = {**value, "content": value.get("content", content), "seen_count": seen_count}
            self._store.put(namespace, key=item.key, value=merged)
            return
        self._store.put(namespace, key=key, value={"content": content, "seen_count": 1})


@lru_cache(maxsize=1)
def get_semantic_memory() -> MongoSemanticMemory:
    """Process-wide default semantic memory wired to the shared Atlas store."""
    from agent.memory import get_store
    from core.settings import get_settings

    return MongoSemanticMemory(
        store=get_store(),
        dedup_threshold=get_settings().semantic_dedup_threshold,
    )


__all__ = ["MongoSemanticMemory", "get_semantic_memory"]
