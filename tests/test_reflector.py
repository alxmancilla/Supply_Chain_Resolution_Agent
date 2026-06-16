"""Unit tests for LLMMemoryReflector clustering + consolidation."""
from __future__ import annotations

from typing import Sequence

from core.memory.reflector import (
    LLMMemoryReflector,
    MongoMemoryAdmin,
    ReflectionReport,
    StoredFact,
    _cosine,
    _greedy_cluster,
)
from tests.fakes import FakeChatProvider, FakeEmbeddings


class FakeMemoryAdmin:
    """Implements `MemoryAdmin` against a dict of pre-staged facts."""

    def __init__(self, facts: list[StoredFact]):
        self._facts = list(facts)
        self.tombstoned: list[str] = []
        self.canonicals: list[tuple[str, list[str]]] = []

    def list_live(self, realm_id: str, user_id: str) -> Sequence[StoredFact]:
        return list(self._facts)

    def tombstone(self, realm_id: str, user_id: str, keys: Sequence[str]) -> None:
        self.tombstoned.extend(keys)

    def write_canonical(
        self, realm_id: str, user_id: str, content: str, source_keys: Sequence[str]
    ) -> str:
        key = f"canon_{len(self.canonicals)}"
        self.canonicals.append((content, list(source_keys)))
        return key


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_greedy_cluster_groups_similar_drops_dissimilar():
    facts = [
        StoredFact(key="a", content="x", embedding=[1.0, 0.0]),
        StoredFact(key="b", content="y", embedding=[0.99, 0.14]),
        StoredFact(key="c", content="z", embedding=[0.0, 1.0]),
    ]
    clusters = _greedy_cluster(facts, threshold=0.9)
    assert len(clusters) == 2
    assert {f.key for f in clusters[0]} == {"a", "b"}
    assert {f.key for f in clusters[1]} == {"c"}


def test_reflector_consolidates_cluster_and_tombstones_originals():
    facts = [
        StoredFact(key="f1", content="prefers Carrier A on TX-AZ", embedding=[1.0, 0.0]),
        StoredFact(key="f2", content="user picks Carrier A for TX-AZ lanes", embedding=[0.99, 0.05]),
        StoredFact(key="f3", content="weight threshold 20000 lb", embedding=[0.0, 1.0]),
    ]
    admin = FakeMemoryAdmin(facts)
    chat = FakeChatProvider(reply='{"canonical": "Prefers Carrier A on TX-AZ lanes."}')
    reflector = LLMMemoryReflector(admin=admin, chat=chat, similarity_threshold=0.9)

    report = reflector.reflect("r", "u")

    assert isinstance(report, ReflectionReport)
    assert report.clusters_found == 1
    assert report.skipped_singletons == 1
    assert sorted(report.tombstoned_keys) == ["f1", "f2"]
    assert report.canonical_written == ["canon_0"]
    assert admin.canonicals == [("Prefers Carrier A on TX-AZ lanes.", ["f1", "f2"])]
    assert sorted(admin.tombstoned) == ["f1", "f2"]


def test_reflector_noop_when_all_singletons():
    facts = [
        StoredFact(key="a", content="x", embedding=[1.0, 0.0]),
        StoredFact(key="b", content="y", embedding=[0.0, 1.0]),
    ]
    admin = FakeMemoryAdmin(facts)
    chat = FakeChatProvider(reply='{"canonical": "unused"}')
    reflector = LLMMemoryReflector(admin=admin, chat=chat, similarity_threshold=0.9)

    report = reflector.reflect("r", "u")

    assert report.clusters_found == 0
    assert report.canonical_written == []
    assert admin.tombstoned == []
    assert admin.canonicals == []
    assert chat.calls == []  # LLM not invoked on singletons


def test_mongo_memory_admin_round_trip_with_fake_collection():
    class FakeColl:
        def __init__(self):
            self.docs: list[dict] = []
        def find(self, filt):
            ns = filt["namespace"]
            for d in self.docs:
                if d["namespace"] != ns:
                    continue
                if d.get("value", {}).get("tombstoned"):
                    continue
                yield d
        def update_many(self, filt, update):
            ns = filt["namespace"]; keys = filt["key"]["$in"]
            for d in self.docs:
                if d["namespace"] == ns and d["key"] in keys:
                    d.setdefault("value", {}).update({"tombstoned": True})
        def update_one(self, filt, update, upsert=False):
            for d in self.docs:
                if d["namespace"] == filt["namespace"] and d["key"] == filt["key"]:
                    d.update(update["$set"])
                    return
            if upsert:
                doc = {**update["$set"]}
                doc.update(update.get("$setOnInsert", {}))
                self.docs.append(doc)

    coll = FakeColl()
    coll.docs.append({
        "namespace": ["r", "u"], "key": "f1",
        "value": {"content": "fact one"}, "embedding": [1.0, 0.0],
    })
    admin = MongoMemoryAdmin(collection=coll, content_field="content", embeddings=FakeEmbeddings())

    live = admin.list_live("r", "u")
    assert len(live) == 1 and live[0].key == "f1"

    key = admin.write_canonical("r", "u", "canonical fact", ["f1"])
    admin.tombstone("r", "u", ["f1"])

    live_after = admin.list_live("r", "u")
    assert [f.key for f in live_after] == [key]
    assert key.startswith("canon_")
