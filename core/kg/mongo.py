"""MongoDB-backed structured retrieval over the supply chain knowledge graph.

Pipeline shape:
  kg_lanes (seed)
    -> $graphLookup over kg_serves (edges) — maxDepth=1, direct edges for the
       seed lane
    -> $lookup kg_serves AGAIN by carrier_id (true second hop: "other lanes
       those carriers also serve") tagged with hop=1 (direct) / hop=2 (via
       carrier expansion)
    -> $lookup kg_carriers, kg_slas, kg_lanes for the hop-2 lane metadata
    -> constraint filtering on hop=1 only (surcharge_max,
       weight_threshold_lb_min)
    -> $project into deterministic rows; sorted hop ASC, priority ASC

The same Atlas cluster already serves RAG and LTM — that is the demo's
value prop.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from core.schemas import EntitySpec, GraphEdge, GraphNode, Subgraph


class MongoKnowledgeGraph:
    """Implements `core.protocols.KnowledgeGraph` over four KG collections."""

    def __init__(
        self,
        *,
        lanes: Any,
        carriers: Any,
        slas: Any,
        serves: Any,
        serves_collection_name: str = "kg_serves",
        carriers_collection_name: str = "kg_carriers",
        slas_collection_name: str = "kg_slas",
    ) -> None:
        self._lanes = lanes
        self._carriers = carriers
        self._slas = slas
        self._serves = serves
        self._serves_name = serves_collection_name
        self._carriers_name = carriers_collection_name
        self._slas_name = slas_collection_name

    def query(self, realm_id: str, spec: EntitySpec, *, limit: int = 10) -> Subgraph:
        if not spec.lanes and not spec.carriers:
            return Subgraph()
        rows = self._fetch_rows(realm_id, spec, limit=limit * 4)
        rows = _apply_constraints(rows, spec.constraints)[:limit]
        return _to_subgraph(rows)

    def _fetch_rows(self, realm_id: str, spec: EntitySpec, *, limit: int) -> list[dict[str, Any]]:
        match: dict[str, Any] = {"realm_id": realm_id}
        if spec.lanes:
            match["lane_id"] = {"$in": list(spec.lanes)}
        pipeline = [
            {"$match": match},
            {"$graphLookup": {
                "from": self._serves_name,
                "startWith": "$lane_id",
                "connectFromField": "lane_id",
                "connectToField": "lane_id",
                "as": "serves_edges",
                "maxDepth": 1,
                "restrictSearchWithMatch": {"realm_id": realm_id},
            }},
            {"$unwind": "$serves_edges"},
            {"$lookup": {
                "from": self._serves_name,
                "let": {"cid": "$serves_edges.carrier_id", "seed_lane": "$lane_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [
                        {"$eq": ["$carrier_id", "$$cid"]},
                        {"$eq": ["$realm_id", realm_id]},
                    ]}}},
                    {"$addFields": {"hop": {"$cond": [
                        {"$eq": ["$lane_id", "$$seed_lane"]}, 1, 2,
                    ]}}},
                ],
                "as": "carrier_serves_all",
            }},
            {"$unwind": "$carrier_serves_all"},
            {"$lookup": {
                "from": self._carriers_name,
                "let": {"cid": "$carrier_serves_all.carrier_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [
                        {"$eq": ["$carrier_id", "$$cid"]},
                        {"$eq": ["$realm_id", realm_id]},
                    ]}}}
                ],
                "as": "carrier",
            }},
            {"$unwind": "$carrier"},
            {"$lookup": {
                "from": self._slas_name,
                "let": {"cid": "$carrier_serves_all.carrier_id", "lid": "$carrier_serves_all.lane_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [
                        {"$eq": ["$carrier_id", "$$cid"]},
                        {"$eq": ["$lane_id", "$$lid"]},
                        {"$eq": ["$realm_id", realm_id]},
                    ]}}}
                ],
                "as": "sla",
            }},
            {"$unwind": {"path": "$sla", "preserveNullAndEmptyArrays": True}},
            {"$lookup": {
                "from": "kg_lanes",
                "let": {"lid": "$carrier_serves_all.lane_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [
                        {"$eq": ["$lane_id", "$$lid"]},
                        {"$eq": ["$realm_id", realm_id]},
                    ]}}}
                ],
                "as": "served_lane",
            }},
            {"$unwind": "$served_lane"},
            {"$project": {
                "_id": 0,
                "seed_lane_id": "$lane_id",
                "lane_id": "$carrier_serves_all.lane_id",
                "lane_origin": "$served_lane.origin_state",
                "lane_dest": "$served_lane.dest_state",
                "carrier_id": "$carrier.carrier_id",
                "carrier_name": "$carrier.name",
                "tier": "$carrier.tier",
                "priority": "$carrier_serves_all.priority",
                "hop": "$carrier_serves_all.hop",
                "surcharge_rate": "$sla.surcharge_rate",
                "weight_threshold_lb": "$sla.weight_threshold_lb",
                "transit_hours": "$sla.transit_hours",
                "lane_source_doc": "$served_lane.source_doc",
                "carrier_source_doc": "$carrier.source_doc",
                "sla_source_doc": "$sla.source_doc",
            }},
            {"$sort": {"hop": 1, "priority": 1, "carrier_id": 1, "lane_id": 1}},
            {"$limit": limit},
        ]
        return list(self._lanes.aggregate(pipeline))


def _apply_constraints(rows: list[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter hop-1 rows by user constraints; keep hop-2 rows only for surviving carriers."""
    surcharge_max = constraints.get("surcharge_max")
    weight_min = constraints.get("weight_threshold_lb_min")
    surviving_carriers: set[Any] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        if (r.get("hop") or 1) != 1:
            continue
        if surcharge_max is not None and (r.get("surcharge_rate") or 0) > surcharge_max:
            continue
        if weight_min is not None and (r.get("weight_threshold_lb") or 0) < weight_min:
            continue
        out.append(r)
        surviving_carriers.add(r.get("carrier_id"))
    for r in rows:
        if (r.get("hop") or 1) == 1:
            continue
        if r.get("carrier_id") in surviving_carriers:
            out.append(r)
    return out


def _to_subgraph(rows: list[dict[str, Any]]) -> Subgraph:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    facts: list[str] = []
    sources: list[str] = []
    seen_nodes: set[tuple[str, str]] = set()

    def _add_node(kind: str, ident: str, props: dict[str, Any]) -> None:
        key = (kind, ident)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        nodes.append(GraphNode(kind=kind, id=ident, properties=props))

    for r in rows:
        lane = r.get("lane_id") or "?"
        carrier = r.get("carrier_id") or "?"
        hop = r.get("hop") or 1
        _add_node("lane", lane, {"origin": r.get("lane_origin"), "dest": r.get("lane_dest")})
        _add_node("carrier", carrier, {"name": r.get("carrier_name"), "tier": r.get("tier")})
        edges.append(GraphEdge(
            kind="serves",
            from_id=carrier,
            to_id=lane,
            properties={"priority": r.get("priority"), "hop": hop},
        ))
        src_parts = [s for s in (r.get("lane_source_doc"), r.get("carrier_source_doc"), r.get("sla_source_doc")) if s]
        for s in src_parts:
            if s not in sources:
                sources.append(s)
        surcharge = r.get("surcharge_rate")
        wt = r.get("weight_threshold_lb")
        transit = r.get("transit_hours")
        verb = "serves" if hop == 1 else "also serves (hop 2)"
        fact = (
            f"- {r.get('carrier_name') or carrier} {verb} lane {lane} "
            f"({r.get('lane_origin','?')}->{r.get('lane_dest','?')}); "
            f"priority={r.get('priority','?')}, "
            f"surcharge_rate={surcharge if surcharge is not None else 'n/a'}, "
            f"weight_threshold_lb={wt if wt is not None else 'n/a'}, "
            f"transit_hours={transit if transit is not None else 'n/a'} "
            f"[sources: {', '.join(src_parts) or 'n/a'}]"
        )
        facts.append(fact)
    return Subgraph(nodes=nodes, edges=edges, facts=facts, sources=sources)


@lru_cache(maxsize=1)
def get_knowledge_graph() -> MongoKnowledgeGraph:
    """Process-wide default KG wired to the shared Atlas client."""
    from agent.memory import get_kg_collections

    carriers, lanes, slas, serves = get_kg_collections()
    return MongoKnowledgeGraph(carriers=carriers, lanes=lanes, slas=slas, serves=serves)


__all__ = ["MongoKnowledgeGraph", "get_knowledge_graph"]
