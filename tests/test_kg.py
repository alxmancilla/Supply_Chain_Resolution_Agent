"""Unit tests for the KG layer: entity extraction + retrieve_kg node."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent import nodes
from agent.nodes import retrieve_kg
from core.kg.extractor import RegexEntityExtractor
from core.kg.mongo import MongoKnowledgeGraph
from core.schemas import EntitySpec
from tests.fakes import FakeEntityExtractor, FakeKnowledgeGraph


def _state(context, text: str = "hello"):
    return {"messages": [HumanMessage(content=text)], "context": context}


def test_regex_extractor_lane_code():
    spec = RegexEntityExtractor().extract("Which carriers serve TX-AZ?")
    assert "TX-AZ" in spec.lanes


def test_regex_extractor_city_pair_to_state_lane():
    spec = RegexEntityExtractor().extract("I need to ship Austin to Dallas")
    assert "TX-TX" in spec.lanes


def test_regex_extractor_weight_and_surcharge_constraints():
    spec = RegexEntityExtractor().extract(
        "Show carriers over 18,000 lbs with no fuel surcharge on TX-AZ"
    )
    assert spec.weight_lb == 18000.0
    assert spec.constraints.get("surcharge_max") == 0.0
    assert spec.constraints.get("weight_threshold_lb_min") == 18000.0
    assert "TX-AZ" in spec.lanes


def test_regex_extractor_carrier_letter():
    spec = RegexEntityExtractor().extract("How does Carrier B do on TX-AZ?")
    assert "carrier_b" in spec.carriers


def test_retrieve_kg_happy_path(monkeypatch, context):
    subgraph = {
        "nodes": [
            {"kind": "lane", "id": "TX-AZ", "properties": {}},
            {"kind": "carrier", "id": "carrier_a", "properties": {"name": "Carrier A"}},
        ],
        "edges": [
            {"kind": "serves", "from_id": "carrier_a", "to_id": "TX-AZ", "properties": {"priority": 1}},
        ],
        "facts": ["- Carrier A serves lane TX-AZ; surcharge=0"],
        "sources": ["route_guides/tx_az_lane.pdf"],
    }
    monkeypatch.setattr(
        nodes, "get_entity_extractor", lambda: FakeEntityExtractor(lanes=["TX-AZ"])
    )
    monkeypatch.setattr(
        nodes, "get_knowledge_graph", lambda: FakeKnowledgeGraph(subgraph=subgraph)
    )
    out = retrieve_kg(_state(context, "Which carriers serve TX-AZ?"))
    assert len(out["kg_hits"]) == 1
    assert out["kg_hits"][0]["fact"].startswith("- Carrier A serves lane TX-AZ")
    assert "Carrier A serves lane TX-AZ" in out["kg_context"]
    assert "kg_ms" in out["latency_ms"]


def test_retrieve_kg_no_query(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_entity_extractor", lambda: FakeEntityExtractor())
    monkeypatch.setattr(nodes, "get_knowledge_graph", lambda: FakeKnowledgeGraph())
    out = retrieve_kg({"messages": [], "context": context})
    assert out["kg_hits"] == []
    assert out["kg_context"] == "(no query)"


def test_retrieve_kg_empty_entity_spec_short_circuits(monkeypatch, context):
    monkeypatch.setattr(nodes, "get_entity_extractor", lambda: FakeEntityExtractor())
    graph = FakeKnowledgeGraph(subgraph={"facts": ["should not appear"]})
    monkeypatch.setattr(nodes, "get_knowledge_graph", lambda: graph)
    out = retrieve_kg(_state(context, "hello there"))
    assert out["kg_hits"] == []
    assert "no entities resolved" in out["kg_context"]
    assert graph.calls == []


def test_retrieve_kg_empty_subgraph(monkeypatch, context):
    monkeypatch.setattr(
        nodes, "get_entity_extractor", lambda: FakeEntityExtractor(lanes=["TX-AZ"])
    )
    monkeypatch.setattr(nodes, "get_knowledge_graph", lambda: FakeKnowledgeGraph())
    out = retrieve_kg(_state(context, "TX-AZ"))
    assert out["kg_hits"] == []
    assert "no matching facts" in out["kg_context"]


def test_retrieve_kg_degrades_on_backend_failure(monkeypatch, context):
    monkeypatch.setattr(
        nodes, "get_entity_extractor", lambda: FakeEntityExtractor(lanes=["TX-AZ"])
    )
    monkeypatch.setattr(
        nodes,
        "get_knowledge_graph",
        lambda: FakeKnowledgeGraph(raise_exc=RuntimeError("graph unreachable")),
    )
    out = retrieve_kg(_state(context, "TX-AZ"))
    assert out["kg_hits"] == []
    assert "retrieval degraded" in out["kg_context"]
    assert any("retrieve_kg" in m for m in out.get("degraded", []))
    assert "kg_ms" in out["latency_ms"]


class _FakeLanesAggregate:
    """Stand-in for the kg_lanes collection: records the pipeline + returns fixed rows."""
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.pipelines: list[list[dict]] = []

    def aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        return iter(self.rows)


def test_multi_hop_pipeline_uses_maxdepth_1_and_chains_serves_lookup():
    """Pipeline must traverse $graphLookup maxDepth=1 then re-join kg_serves by carrier."""
    lanes = _FakeLanesAggregate(rows=[])
    kg = MongoKnowledgeGraph(lanes=lanes, carriers=None, slas=None, serves=None)
    kg.query("realm-x", EntitySpec(lanes=["TX-AZ"]), limit=5)

    assert len(lanes.pipelines) == 1
    stages = lanes.pipelines[0]
    graph_lookup = next(s["$graphLookup"] for s in stages if "$graphLookup" in s)
    assert graph_lookup["maxDepth"] == 1
    assert graph_lookup["from"] == "kg_serves"

    serves_lookups = [
        s["$lookup"] for s in stages
        if "$lookup" in s and s["$lookup"].get("from") == "kg_serves"
    ]
    assert len(serves_lookups) == 1, "second hop must re-join kg_serves by carrier_id"
    inner = serves_lookups[0]["pipeline"]
    add_fields = next(stage for stage in inner if "$addFields" in stage)
    assert "hop" in add_fields["$addFields"]


def test_multi_hop_subgraph_distinguishes_direct_and_expanded_facts():
    """hop=1 rows render as 'serves'; hop=2 rows render as 'also serves (hop 2)'."""
    rows = [
        {
            "lane_id": "TX-AZ", "lane_origin": "TX", "lane_dest": "AZ",
            "carrier_id": "carrier_a", "carrier_name": "Carrier A", "tier": "preferred",
            "priority": 1, "hop": 1,
            "surcharge_rate": 0.0, "weight_threshold_lb": 20000, "transit_hours": 18,
            "lane_source_doc": "route_guides/tx_az.pdf",
            "carrier_source_doc": "carrier_agreements/a.pdf",
            "sla_source_doc": "slas/a_tx_az.pdf",
        },
        {
            "lane_id": "TX-CA", "lane_origin": "TX", "lane_dest": "CA",
            "carrier_id": "carrier_a", "carrier_name": "Carrier A", "tier": "preferred",
            "priority": 2, "hop": 2,
            "surcharge_rate": 0.045, "weight_threshold_lb": 15000, "transit_hours": 30,
            "lane_source_doc": "route_guides/tx_ca.pdf",
            "carrier_source_doc": "carrier_agreements/a.pdf",
            "sla_source_doc": "slas/a_tx_ca.pdf",
        },
    ]
    lanes = _FakeLanesAggregate(rows=rows)
    kg = MongoKnowledgeGraph(lanes=lanes, carriers=None, slas=None, serves=None)

    sub = kg.query("realm-x", EntitySpec(lanes=["TX-AZ"]), limit=10)
    facts = sub.facts
    assert any("Carrier A serves lane TX-AZ" in f for f in facts)
    assert any("Carrier A also serves (hop 2) lane TX-CA" in f for f in facts)
    edges_by_lane = {e.to_id: e.properties.get("hop") for e in sub.edges}
    assert edges_by_lane == {"TX-AZ": 1, "TX-CA": 2}


def test_multi_hop_constraints_filter_hop1_and_drop_orphan_hop2():
    """Hop-2 rows are dropped when their carrier's hop-1 row fails constraints."""
    rows = [
        {  # hop-1 with high surcharge — will be filtered out
            "lane_id": "TX-AZ", "carrier_id": "carrier_b", "carrier_name": "Carrier B",
            "priority": 1, "hop": 1, "surcharge_rate": 0.05, "weight_threshold_lb": 25000,
        },
        {  # hop-2 from same carrier — should be dropped (orphaned)
            "lane_id": "TX-CA", "carrier_id": "carrier_b", "carrier_name": "Carrier B",
            "priority": 2, "hop": 2, "surcharge_rate": 0.0, "weight_threshold_lb": 30000,
        },
        {  # hop-1 that satisfies constraints
            "lane_id": "TX-AZ", "carrier_id": "carrier_a", "carrier_name": "Carrier A",
            "priority": 1, "hop": 1, "surcharge_rate": 0.0, "weight_threshold_lb": 25000,
        },
        {  # hop-2 from surviving carrier — should be kept
            "lane_id": "TX-NV", "carrier_id": "carrier_a", "carrier_name": "Carrier A",
            "priority": 3, "hop": 2, "surcharge_rate": 0.045, "weight_threshold_lb": 10000,
        },
    ]
    lanes = _FakeLanesAggregate(rows=rows)
    kg = MongoKnowledgeGraph(lanes=lanes, carriers=None, slas=None, serves=None)

    sub = kg.query(
        "realm-x",
        EntitySpec(lanes=["TX-AZ"], constraints={"surcharge_max": 0.0, "weight_threshold_lb_min": 20000}),
        limit=10,
    )
    carriers = {n.id for n in sub.nodes if n.kind == "carrier"}
    assert carriers == {"carrier_a"}
    lanes_seen = {n.id for n in sub.nodes if n.kind == "lane"}
    assert lanes_seen == {"TX-AZ", "TX-NV"}
