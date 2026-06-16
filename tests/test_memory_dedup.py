"""Pre-write dedup + tombstone-aware reads for semantic & episodic memory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.memory.episodic import MongoEpisodicMemory
from core.memory.semantic import MongoSemanticMemory


@dataclass
class _Item:
    """Stand-in for langgraph's SearchItem (only fields the memory layer reads)."""
    key: str
    value: dict[str, Any]
    score: float | None = None
    updated_at: Any = None


class FakeStore:
    """Tiny in-memory MongoDBStore double.

    `search` returns items sorted by a caller-supplied score map keyed on
    `(query, key)` — lets each test pre-stage the similarity it wants without
    pulling in a real embedder.
    """

    def __init__(self, *, score_map: dict[tuple[str, str], float] | None = None):
        self._items: dict[tuple, dict[str, _Item]] = {}
        self._score_map = score_map or {}

    def put(self, namespace, *, key, value):
        self._items.setdefault(namespace, {})[key] = _Item(key=key, value=dict(value))

    def search(self, namespace, *, query, limit):
        bucket = self._items.get(namespace, {})
        scored: list[_Item] = []
        for k, item in bucket.items():
            score = self._score_map.get((query, k), 0.0)
            scored.append(_Item(key=item.key, value=dict(item.value), score=score, updated_at=None))
        scored.sort(key=lambda i: i.score or 0.0, reverse=True)
        return scored[:limit]


def test_semantic_put_inserts_when_no_match():
    store = FakeStore()
    mem = MongoSemanticMemory(store=store, dedup_threshold=0.9)
    mem.put("r", "u", key="k1", content="prefers carrier A")
    items = store.search(("r", "u"), query="x", limit=10)
    assert len(items) == 1
    assert items[0].value == {"content": "prefers carrier A", "seen_count": 1}


def test_semantic_put_bumps_seen_count_when_above_threshold():
    store = FakeStore(score_map={("prefers carrier A again", "k1"): 0.95})
    mem = MongoSemanticMemory(store=store, dedup_threshold=0.9)
    store.put(("r", "u"), key="k1", value={"content": "prefers carrier A", "seen_count": 1})
    mem.put("r", "u", key="k2", content="prefers carrier A again")
    items = store.search(("r", "u"), query="prefers carrier A again", limit=10)
    assert len(items) == 1
    assert items[0].key == "k1"
    assert items[0].value["seen_count"] == 2
    assert items[0].value["content"] == "prefers carrier A"


def test_semantic_put_inserts_when_below_threshold():
    store = FakeStore(score_map={("totally different", "k1"): 0.42})
    mem = MongoSemanticMemory(store=store, dedup_threshold=0.9)
    store.put(("r", "u"), key="k1", value={"content": "prefers carrier A", "seen_count": 1})
    mem.put("r", "u", key="k2", content="totally different")
    items = sorted(
        store.search(("r", "u"), query="x", limit=10), key=lambda i: i.key
    )
    assert [i.key for i in items] == ["k1", "k2"]


def test_semantic_search_filters_tombstoned():
    store = FakeStore(score_map={("q", "live"): 0.8, ("q", "dead"): 0.9})
    store.put(("r", "u"), key="live", value={"content": "live fact"})
    store.put(("r", "u"), key="dead", value={"content": "dead fact", "tombstoned": True})
    mem = MongoSemanticMemory(store=store)
    facts = mem.search("r", "u", query="q", limit=5)
    assert [f.key for f in facts] == ["live"]


def test_semantic_dedup_ignores_tombstoned_neighbour():
    store = FakeStore(score_map={("recall_x", "old"): 0.99})
    store.put(("r", "u"), key="old", value={"content": "x", "tombstoned": True})
    mem = MongoSemanticMemory(store=store, dedup_threshold=0.9)
    mem.put("r", "u", key="new", content="recall_x")
    keys = {item.key for item in store.search(("r", "u"), query="z", limit=10)}
    assert keys == {"old", "new"}


def test_semantic_dedup_bumps_canonical_after_skipping_tombstoned_exact_match():
    """Repro for the smoke-test zombie-revival: identical key tombstoned + canonical above threshold."""
    store = FakeStore(score_map={
        ("prefers Carrier A", "mem_zombie"): 1.0,
        ("prefers Carrier A", "canon_live"): 0.93,
    })
    store.put(("r", "u"), key="mem_zombie", value={"content": "prefers Carrier A", "tombstoned": True})
    store.put(("r", "u"), key="canon_live", value={"content": "Prefers Carrier A on TX-AZ lanes.",
                                                    "canonical": True, "seen_count": 9})
    mem = MongoSemanticMemory(store=store, dedup_threshold=0.92)
    mem.put("r", "u", key="mem_zombie", content="prefers Carrier A")
    items = {i.key: i.value for i in store.search(("r", "u"), query="z", limit=10)}
    assert items["canon_live"]["seen_count"] == 10
    assert items["mem_zombie"]["tombstoned"] is True  # not revived


def test_episodic_put_bumps_seen_count_on_similar_summary():
    store = FakeStore(score_map={("shipped to Dallas", "ep1"): 0.96})
    mem = MongoEpisodicMemory(store=store, dedup_threshold=0.9)
    store.put(("r", "u"), key="ep1",
              value={"summary": "shipped to Dallas", "occurred_at": "2026-01-01", "seen_count": 1})
    mem.put("r", "u", key="ep2",
            episode={"summary": "shipped to Dallas", "occurred_at": "2026-06-15"})
    items = store.search(("r", "u"), query="x", limit=10)
    assert len(items) == 1
    assert items[0].key == "ep1"
    assert items[0].value["seen_count"] == 2
    assert items[0].value["occurred_at"] == "2026-06-15"


def test_episodic_search_filters_tombstoned():
    store = FakeStore(score_map={("q", "live"): 0.7, ("q", "dead"): 0.95})
    store.put(("r", "u"), key="live", value={"summary": "live"})
    store.put(("r", "u"), key="dead", value={"summary": "dead", "tombstoned": True})
    mem = MongoEpisodicMemory(store=store)
    eps = mem.search("r", "u", query="q", limit=5)
    assert [e.key for e in eps] == ["live"]


def test_reset_memory_cache_clears_all_three_singletons(monkeypatch):
    from core import memory as memory_pkg

    def _stub_semantic():
        from agent.memory import get_store
        return MongoSemanticMemory(store=get_store(), dedup_threshold=0.9)

    def _stub_episodic():
        from agent.memory import get_episodes_store
        return MongoEpisodicMemory(store=get_episodes_store(), dedup_threshold=0.9)

    monkeypatch.setattr("agent.memory.get_store", lambda: FakeStore())
    monkeypatch.setattr("agent.memory.get_episodes_store", lambda: FakeStore())
    monkeypatch.setattr("agent.memory.get_procedures_collection", lambda: object())
    monkeypatch.setattr("agent.memory.get_procedure_proposals_collection", lambda: object())

    memory_pkg.get_semantic_memory.cache_clear()
    memory_pkg.get_episodic_memory.cache_clear()
    memory_pkg.get_procedural_memory.cache_clear()

    first_sem = memory_pkg.get_semantic_memory()
    first_epi = memory_pkg.get_episodic_memory()
    first_proc = memory_pkg.get_procedural_memory()
    assert memory_pkg.get_semantic_memory() is first_sem
    assert memory_pkg.get_episodic_memory() is first_epi
    assert memory_pkg.get_procedural_memory() is first_proc

    memory_pkg.reset_memory_cache()
    assert memory_pkg.get_semantic_memory() is not first_sem
    assert memory_pkg.get_episodic_memory() is not first_epi
    assert memory_pkg.get_procedural_memory() is not first_proc
