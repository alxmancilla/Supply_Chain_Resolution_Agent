"""Pydantic value shapes used across the memory and RAG layers.

These models are the on-the-wire contract for what nodes hand back to the
graph. They intentionally mirror the dict shapes already present in
agent/nodes.py so the refactor is behavior-preserving.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


BranchName = Literal["ltm", "episodes", "procedures", "rag", "kg"]
ALL_BRANCHES: tuple[BranchName, ...] = ("ltm", "episodes", "procedures", "rag", "kg")


class SemanticFact(BaseModel):
    """A durable, declarative fact retrieved from semantic LTM."""
    model_config = ConfigDict(extra="allow")

    key: str
    content: str
    score: Optional[float] = None
    updated_at: Any = None


class Episode(BaseModel):
    """A structured past interaction retrieved from episodic LTM."""
    model_config = ConfigDict(extra="allow")

    key: str
    summary: str
    lane: Optional[str] = None
    recommendation: Optional[str] = None
    outcome: Optional[str] = None
    occurred_at: Optional[str] = None
    score: Optional[float] = None


class ProceduralRule(BaseModel):
    """A tenant-curated operating rule retrieved from procedural LTM."""
    model_config = ConfigDict(extra="allow")

    rule_id: str
    rule: str
    category: str = "general"


class KnowledgeHit(BaseModel):
    """A RAG chunk retrieved from the knowledge corpus."""
    model_config = ConfigDict(extra="allow")

    doc_type: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """An intent-router decision narrowing the per-turn retrieval fan-out."""
    model_config = ConfigDict(extra="allow")

    intent_label: str
    branches: list[BranchName]
    rationale: str = ""


class EntitySpec(BaseModel):
    """Extracted entities used as seeds for a knowledge-graph query."""
    model_config = ConfigDict(extra="allow")

    lanes: list[str] = Field(default_factory=list)
    carriers: list[str] = Field(default_factory=list)
    weight_lb: Optional[float] = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """A typed node in the supply chain knowledge graph."""
    model_config = ConfigDict(extra="allow")

    kind: str
    id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A typed edge between two GraphNodes."""
    model_config = ConfigDict(extra="allow")

    kind: str
    from_id: str
    to_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class Subgraph(BaseModel):
    """The materialized subgraph returned by a KnowledgeGraph query."""
    model_config = ConfigDict(extra="allow")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ExtractedEpisode(BaseModel):
    """Episode shape returned by the combined memory extractor."""
    model_config = ConfigDict(extra="allow")

    summary: str
    lane: Optional[str] = None
    recommendation: Optional[str] = None
    outcome: Optional[str] = None


class MemoryExtraction(BaseModel):
    """Combined semantic + episodic output of a single extraction call."""
    model_config = ConfigDict(extra="allow")

    facts: list[str] = Field(default_factory=list)
    episode: Optional[ExtractedEpisode] = None


ActionType = Literal["create_booking_draft", "propose_procedure", "none"]
ProcedureCategory = Literal["escalation", "formatting", "units", "policy", "general"]


class BookingProposal(BaseModel):
    """Structured booking proposal extracted from the agent reply."""
    model_config = ConfigDict(extra="allow")

    action_type: ActionType = "none"
    carrier: Optional[str] = None
    lane: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    weight_lb: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    requires_approval: bool = False
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_defaults(cls, data: Any) -> Any:
        """LLMs often emit `null` for action_type=none turns; fall back to defaults."""
        if isinstance(data, dict):
            if data.get("requires_approval") is None:
                data["requires_approval"] = False
            if data.get("rationale") is None:
                data["rationale"] = ""
            if data.get("action_type") is None:
                data["action_type"] = "none"
        return data


class ProcedureProposal(BaseModel):
    """Structured procedural-rule proposal extracted from a propose_procedure turn."""
    model_config = ConfigDict(extra="allow")

    action_type: Literal["propose_procedure", "none"] = "none"
    rule: str = ""
    category: ProcedureCategory = "general"
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("action_type") is None:
                data["action_type"] = "none"
            if data.get("rule") is None:
                data["rule"] = ""
            if data.get("category") is None:
                data["category"] = "general"
            if data.get("rationale") is None:
                data["rationale"] = ""
        return data


class ResearchPlan(BaseModel):
    """The per-turn retrieval plan emitted by `think_and_plan`.

    A pass-through copy of the router's branches on the first pass; on a
    re-plan (triggered by the Reflection Agent) the LLM may narrow the
    branch set and substitute a refined `subquery` used by the retrievers.
    """
    model_config = ConfigDict(extra="allow")

    branches: list[BranchName] = Field(default_factory=lambda: list(ALL_BRANCHES))
    subquery: Optional[str] = None
    rationale: str = ""
    replan_count: int = 0


class EvidenceReflection(BaseModel):
    """The data-sufficiency verdict emitted by `reflect_on_evidence`."""
    model_config = ConfigDict(extra="allow")

    sufficient: bool = True
    missing: list[str] = Field(default_factory=list)
    followup_subquery: Optional[str] = None
    rationale: str = ""


__all__ = [
    "SemanticFact",
    "Episode",
    "ProceduralRule",
    "KnowledgeHit",
    "RoutingDecision",
    "BranchName",
    "ALL_BRANCHES",
    "EntitySpec",
    "GraphNode",
    "GraphEdge",
    "Subgraph",
    "ExtractedEpisode",
    "MemoryExtraction",
    "ActionType",
    "ProcedureCategory",
    "BookingProposal",
    "ProcedureProposal",
    "ResearchPlan",
    "EvidenceReflection",
]
