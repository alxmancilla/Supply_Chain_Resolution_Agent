"""MongoDB-backed episodic LTM (structured past interactions, vector-retrieved)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

from core.schemas import Episode


class MongoEpisodicMemory:
    """Implements `core.protocols.EpisodicMemory` over a `MongoDBStore`.

    `put` is dedup-aware: it probes the top-1 nearest neighbour on the
    candidate `summary`. If the cosine similarity meets `dedup_threshold`
    the existing row's `seen_count` is bumped (and `occurred_at` refreshed)
    instead of inserting a near-duplicate. `search` filters out rows
    flagged `tombstoned: True`.
    """

    def __init__(self, *, store: Any, dedup_threshold: float = 0.92) -> None:
        self._store = store
        self._dedup_threshold = dedup_threshold

    @staticmethod
    def _namespace(realm_id: str, user_id: str) -> tuple[str, str]:
        return (realm_id, user_id)

    def search(self, realm_id: str, user_id: str, query: str, limit: int) -> Sequence[Episode]:
        results = self._store.search(self._namespace(realm_id, user_id), query=query, limit=limit * 2)
        episodes: list[Episode] = []
        for item in results:
            value = item.value if isinstance(item.value, dict) else {}
            if value.get("tombstoned"):
                continue
            episodes.append(
                Episode(
                    key=item.key,
                    summary=value.get("summary", ""),
                    lane=value.get("lane"),
                    recommendation=value.get("recommendation"),
                    outcome=value.get("outcome"),
                    occurred_at=value.get("occurred_at"),
                    score=item.score,
                )
            )
            if len(episodes) >= limit:
                break
        return episodes

    def put(self, realm_id: str, user_id: str, key: str, episode: Mapping[str, str]) -> None:
        namespace = self._namespace(realm_id, user_id)
        new_value = dict(episode)
        summary = new_value.get("summary", "")
        if summary:
            for item in self._store.search(namespace, query=summary, limit=5):
                value = item.value if isinstance(item.value, dict) else {}
                if value.get("tombstoned"):
                    continue
                if (item.score or 0.0) < self._dedup_threshold:
                    break
                seen_count = int(value.get("seen_count", 1)) + 1
                merged = {**value, "seen_count": seen_count}
                if new_value.get("occurred_at"):
                    merged["occurred_at"] = new_value["occurred_at"]
                self._store.put(namespace, key=item.key, value=merged)
                return
        new_value.setdefault("seen_count", 1)
        self._store.put(namespace, key=key, value=new_value)


@lru_cache(maxsize=1)
def get_episodic_memory() -> MongoEpisodicMemory:
    """Process-wide default episodic memory wired to the shared Atlas store."""
    from agent.memory import get_episodes_store
    from core.settings import get_settings

    return MongoEpisodicMemory(
        store=get_episodes_store(),
        dedup_threshold=get_settings().episodic_dedup_threshold,
    )


__all__ = ["MongoEpisodicMemory", "get_episodic_memory"]
