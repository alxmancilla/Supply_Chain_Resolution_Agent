"""Abstract contracts for the memory, RAG, and model-provider layers.

Concrete backends (MongoDB today, anything tomorrow) implement these
protocols. Graph nodes depend only on the protocols, never on the
backend modules — this is what makes the layers swappable and
unit-testable with in-memory fakes.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

from .schemas import (
    Episode,
    EntitySpec,
    KnowledgeHit,
    ProceduralRule,
    RoutingDecision,
    SemanticFact,
    Subgraph,
)


@runtime_checkable
class SemanticMemory(Protocol):
    """Durable user-scoped facts/preferences, vector-retrieved."""

    def search(self, realm_id: str, user_id: str, query: str, limit: int) -> Sequence[SemanticFact]:
        ...

    def put(self, realm_id: str, user_id: str, key: str, content: str) -> None:
        ...


@runtime_checkable
class EpisodicMemory(Protocol):
    """User-scoped past interactions, vector-retrieved on summary."""

    def search(self, realm_id: str, user_id: str, query: str, limit: int) -> Sequence[Episode]:
        ...

    def put(self, realm_id: str, user_id: str, key: str, episode: Mapping[str, str]) -> None:
        ...


@runtime_checkable
class ProceduralMemory(Protocol):
    """Tenant + agent scoped operating rules, returned as a flat list."""

    def list_active(self, realm_id: str, agent_id: str) -> Sequence[ProceduralRule]:
        ...


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """Tenant-scoped RAG retrieval over the knowledge corpus."""

    def query(self, realm_id: str, text: str, k: int) -> Sequence[KnowledgeHit]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder-style reranker applied to a candidate hit list."""

    def rerank(self, query: str, hits: Sequence[KnowledgeHit], *, top_k: int) -> Sequence[KnowledgeHit]:
        ...


@runtime_checkable
class IntentRouter(Protocol):
    """Decides which retrieval branches to activate for a given user turn."""

    def route(self, user_message: str) -> RoutingDecision:
        ...


@runtime_checkable
class EntityExtractor(Protocol):
    """Pulls supply-chain entities (lanes, carriers, constraints) from text."""

    def extract(self, user_message: str) -> EntitySpec:
        ...


@runtime_checkable
class KnowledgeGraph(Protocol):
    """Tenant-scoped structured retrieval over the supply chain knowledge graph."""

    def query(self, realm_id: str, spec: EntitySpec, *, limit: int) -> Subgraph:
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Pluggable text embedder behind a stable interface.

    Concrete providers (Voyage today, OpenAI / Cohere / local model tomorrow)
    implement this protocol. `dimensions` is checked at boot against the
    vector index spec so a provider swap fails fast instead of silently
    producing zero-recall queries.
    """

    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@runtime_checkable
class ChatProvider(Protocol):
    """Pluggable chat-completion backend behind a stable interface.

    `invoke` is plain-text in / plain-text out. `invoke_typed` is the
    structured-output path used by the router and the memory extraction
    prompts — each provider implements it natively (OpenAI structured
    outputs, Anthropic tool-use, etc.) so callers never depend on a
    specific provider's mechanism.
    """

    model_name: str

    def invoke(self, prompt: str) -> str:
        ...

    def invoke_typed(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        ...


@runtime_checkable
class MemoryReflector(Protocol):
    """Periodic consolidation pass over a (semantic | episodic) memory namespace."""

    def reflect(self, realm_id: str, user_id: str) -> Any:
        ...


__all__ = [
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    "KnowledgeRetriever",
    "IntentRouter",
    "EntityExtractor",
    "KnowledgeGraph",
    "EmbeddingProvider",
    "ChatProvider",
    "MemoryReflector",
]
