"""In-memory fakes for the Atlas-backed dependencies used by graph nodes.

These mimic only the minimal surface of `MongoDBStore`, raw pymongo
collections, and the `EmbeddingProvider` / `ChatProvider` protocols that
the nodes actually call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class FakeStoreItem:
    """Stand-in for `langgraph.store.base.Item`."""
    key: str
    value: dict[str, Any]
    score: float | None = None
    updated_at: str | None = None


@dataclass
class FakeStore:
    """Mimics `MongoDBStore` — only `.search()` and `.put()` used by nodes."""
    items: list[FakeStoreItem] = field(default_factory=list)
    puts: list[tuple] = field(default_factory=list)

    def search(self, namespace, query: str, limit: int):
        return self.items[:limit]

    def put(self, namespace, key, value):
        self.puts.append((namespace, key, value))


class _FakeFindCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *_args, **_kwargs):
        return iter(self._docs)


class FakeProceduresCollection:
    """Mimics the raw `agent_procedures` collection: only `.find().sort()`."""
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs

    def find(self, filt: dict, _proj=None):
        matching = [
            d for d in self.docs
            if d.get("realm_id") == filt.get("realm_id")
            and d.get("agent_id") == filt.get("agent_id")
            and d.get("active", True) == filt.get("active", True)
        ]
        return _FakeFindCursor(matching)


class FakeKnowledgeCollection:
    """Mimics the raw `knowledge_corpus` collection: only `.aggregate()`."""
    def __init__(self, hits: list[dict[str, Any]]):
        self.hits = hits

    def aggregate(self, _pipeline):
        return iter(self.hits)


class FakeEmbeddings:
    """Implements `core.protocols.EmbeddingProvider` with constant zero vectors."""

    model_name: str = "fake-embeddings"
    dimensions: int = 1024

    def embed_query(self, _text: str) -> list[float]:
        return [0.0] * self.dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]


class FakeChatProvider:
    """Implements `core.protocols.ChatProvider` from a fixed reply (or raises)."""

    model_name: str = "fake-chat"

    def __init__(
        self,
        *,
        reply: str = "",
        replies: list[str] | None = None,
        raise_exc: Exception | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self._reply = reply
        self._replies = list(replies) if replies is not None else None
        self._raise = raise_exc
        self._usage = usage
        self.calls: list[str] = []
        self.last_usage: dict[str, int] | None = None

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        self.last_usage = dict(self._usage) if self._usage is not None else None
        if self._replies is not None:
            return self._replies.pop(0) if self._replies else ""
        return self._reply

    def invoke_typed(self, prompt: str, schema):
        import json

        raw = self.invoke(prompt)
        return schema.model_validate(json.loads(raw))


class FakeKnowledgeRetriever:
    """Implements `core.protocols.KnowledgeRetriever` over a fixed hit list."""
    def __init__(self, hits: list[dict[str, Any]]):
        from core.schemas import KnowledgeHit
        self._hits = [KnowledgeHit(**h) for h in hits]

    def query(self, realm_id: str, text: str, k: int):
        return self._hits[:k]


class FakeSemanticMemory:
    """Implements `core.protocols.SemanticMemory` from a fixed fact list."""
    def __init__(self, facts: list[dict[str, Any]] | None = None):
        from core.schemas import SemanticFact
        self._facts = [SemanticFact(**f) for f in (facts or [])]
        self.puts: list[tuple] = []

    def search(self, realm_id: str, user_id: str, query: str, limit: int):
        return self._facts[:limit]

    def put(self, realm_id: str, user_id: str, key: str, content: str) -> None:
        self.puts.append((realm_id, user_id, key, content))


class FakeEpisodicMemory:
    """Implements `core.protocols.EpisodicMemory` from a fixed episode list."""
    def __init__(self, episodes: list[dict[str, Any]] | None = None):
        from core.schemas import Episode
        self._episodes = [Episode(**e) for e in (episodes or [])]
        self.puts: list[tuple] = []

    def search(self, realm_id: str, user_id: str, query: str, limit: int):
        return self._episodes[:limit]

    def put(self, realm_id: str, user_id: str, key: str, episode):
        self.puts.append((realm_id, user_id, key, dict(episode)))


class FakeProceduralMemory:
    """Implements `core.protocols.ProceduralMemory` filtering by realm + agent.

    Also mimics the `propose / commit / reject` staging surface used by
    `execute_action` for the `propose_procedure` action type.
    """
    def __init__(self, rules: list[dict[str, Any]] | None = None):
        self._rules = list(rules or [])
        self._proposals: list[dict[str, Any]] = []
        self.proposed: list[dict[str, Any]] = []
        self.committed: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []

    def list_active(self, realm_id: str, agent_id: str):
        from core.schemas import ProceduralRule
        return [
            ProceduralRule(rule_id=r["rule_id"], rule=r["rule"], category=r.get("category", "general"))
            for r in self._rules
            if r.get("realm_id") == realm_id and r.get("agent_id") == agent_id and r.get("active", True)
        ]

    def propose(self, realm_id, agent_id, *, rule, category="general", rationale="",
                proposed_by=None, correlation_id=None):
        from core.memory.procedural import _stable_rule_id
        rule_id = _stable_rule_id(rule)
        doc = {
            "proposal_id": rule_id, "rule_id": rule_id, "realm_id": realm_id,
            "agent_id": agent_id, "rule": rule.strip(), "category": category,
            "rationale": rationale, "status": "pending_approval",
            "proposed_by": proposed_by, "correlation_id": correlation_id,
        }
        self._proposals = [
            p for p in self._proposals
            if not (p["realm_id"] == realm_id and p["agent_id"] == agent_id
                    and p["proposal_id"] == rule_id)
        ]
        self._proposals.append(doc)
        self.proposed.append(doc)
        return doc

    def commit(self, realm_id, agent_id, proposal_id, *, approver=None):
        proposal = next(
            (p for p in self._proposals
             if p["realm_id"] == realm_id and p["agent_id"] == agent_id
             and p["proposal_id"] == proposal_id),
            None,
        )
        if proposal is None:
            raise LookupError(proposal_id)
        rule_doc = {
            "rule_id": proposal["rule_id"], "rule": proposal["rule"],
            "category": proposal.get("category", "general"),
            "realm_id": realm_id, "agent_id": agent_id, "active": True,
            "source_proposal_id": proposal_id, "approver": approver,
        }
        self._rules.append(rule_doc)
        proposal["status"] = "approved"
        proposal["approver"] = approver
        self.committed.append(rule_doc)
        return rule_doc

    def reject(self, realm_id, agent_id, proposal_id, *, approver=None):
        for p in self._proposals:
            if (p["realm_id"] == realm_id and p["agent_id"] == agent_id
                    and p["proposal_id"] == proposal_id):
                p["status"] = "rejected"
                p["approver"] = approver
                self.rejected.append(p)
                return


class FakeIntentRouter:
    """Implements `core.protocols.IntentRouter` from a fixed decision (or raises)."""
    def __init__(self, *, decision: dict[str, Any] | None = None, raise_exc: Exception | None = None):
        self._decision = decision
        self._raise = raise_exc
        self.calls: list[str] = []

    def route(self, user_message: str):
        self.calls.append(user_message)
        if self._raise is not None:
            raise self._raise
        from core.schemas import ALL_BRANCHES, RoutingDecision
        return RoutingDecision(**(self._decision or {
            "intent_label": "fallback",
            "branches": list(ALL_BRANCHES),
            "rationale": "default",
        }))


class FakeEntityExtractor:
    """Implements `core.protocols.EntityExtractor` from a fixed EntitySpec."""
    def __init__(self, *, lanes=None, carriers=None, weight_lb=None, constraints=None):
        self._lanes = list(lanes or [])
        self._carriers = list(carriers or [])
        self._weight_lb = weight_lb
        self._constraints = dict(constraints or {})
        self.calls: list[str] = []

    def extract(self, user_message: str):
        self.calls.append(user_message)
        from core.schemas import EntitySpec
        return EntitySpec(
            lanes=list(self._lanes),
            carriers=list(self._carriers),
            weight_lb=self._weight_lb,
            constraints=dict(self._constraints),
        )


class FakeKnowledgeGraph:
    """Implements `core.protocols.KnowledgeGraph` from a fixed Subgraph (or raises)."""
    def __init__(self, *, subgraph: dict[str, Any] | None = None, raise_exc: Exception | None = None):
        self._subgraph = subgraph
        self._raise = raise_exc
        self.calls: list[tuple] = []

    def query(self, realm_id: str, spec, *, limit: int):
        self.calls.append((realm_id, spec, limit))
        if self._raise is not None:
            raise self._raise
        from core.schemas import Subgraph
        return Subgraph(**(self._subgraph or {}))
