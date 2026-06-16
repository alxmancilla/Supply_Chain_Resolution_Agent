"""Seed the supply chain knowledge graph collections.

Populates four collections (kg_carriers, kg_lanes, kg_slas, kg_serves)
with a small deterministic dataset aligned with the existing RAG corpus
(route guides + carrier agreements) and episodic seeds. Every row carries
`source_doc` so the agent can cite the underlying knowledge artifact.

Usage:
    python -m data.seed_kg
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from agent.memory import get_kg_collections
from core.settings import get_settings

_SETTINGS = get_settings()
REALM_ID = _SETTINGS.realm_id

CARRIERS = [
    {"carrier_id": "carrier_a", "name": "Carrier A", "tier": "primary",
     "hq_state": "TX", "equipment_types": ["dry_van", "reefer"],
     "source_doc": "carrier_agreements/carrier_a_2026.pdf"},
    {"carrier_id": "carrier_b", "name": "Carrier B", "tier": "secondary",
     "hq_state": "AZ", "equipment_types": ["dry_van"],
     "source_doc": "carrier_agreements/carrier_b_2026.pdf"},
    {"carrier_id": "carrier_c", "name": "Carrier C", "tier": "spot",
     "hq_state": "NM", "equipment_types": ["dry_van", "flatbed"],
     "source_doc": "carrier_agreements/carrier_c_2026.pdf"},
]

LANES = [
    {"lane_id": "TX-TX", "origin_state": "TX", "dest_state": "TX",
     "distance_mi": 250, "source_doc": "route_guides/tx_tx_lane.pdf"},
    {"lane_id": "TX-AZ", "origin_state": "TX", "dest_state": "AZ",
     "distance_mi": 830, "source_doc": "route_guides/tx_az_lane.pdf"},
    {"lane_id": "TX-NM", "origin_state": "TX", "dest_state": "NM",
     "distance_mi": 600, "source_doc": "route_guides/tx_nm_lane.pdf"},
]

# Edges: which carriers serve which lanes
SERVES = [
    {"carrier_id": "carrier_a", "lane_id": "TX-TX", "priority": 1, "since": "2024-01-01"},
    {"carrier_id": "carrier_a", "lane_id": "TX-AZ", "priority": 1, "since": "2024-01-01"},
    {"carrier_id": "carrier_b", "lane_id": "TX-AZ", "priority": 2, "since": "2024-03-15"},
    {"carrier_id": "carrier_b", "lane_id": "TX-NM", "priority": 1, "since": "2024-03-15"},
    {"carrier_id": "carrier_c", "lane_id": "TX-NM", "priority": 2, "since": "2025-06-01"},
    {"carrier_id": "carrier_c", "lane_id": "TX-AZ", "priority": 3, "since": "2025-06-01"},
]

# SLAs: per-(carrier, lane) terms — the structured facts that vector RAG
# can only stitch together via LLM reasoning.
SLAS = [
    {"sla_id": "sla_a_txtx", "carrier_id": "carrier_a", "lane_id": "TX-TX",
     "surcharge_rate": 0.0, "weight_threshold_lb": 20000, "transit_hours": 8,
     "source_doc": "carrier_agreements/carrier_a_2026.pdf"},
    {"sla_id": "sla_a_txaz", "carrier_id": "carrier_a", "lane_id": "TX-AZ",
     "surcharge_rate": 0.0, "weight_threshold_lb": 22000, "transit_hours": 24,
     "source_doc": "carrier_agreements/carrier_a_2026.pdf"},
    {"sla_id": "sla_b_txaz", "carrier_id": "carrier_b", "lane_id": "TX-AZ",
     "surcharge_rate": 0.085, "weight_threshold_lb": 15000, "transit_hours": 30,
     "source_doc": "carrier_agreements/carrier_b_2026.pdf"},
    {"sla_id": "sla_b_txnm", "carrier_id": "carrier_b", "lane_id": "TX-NM",
     "surcharge_rate": 0.0, "weight_threshold_lb": 18000, "transit_hours": 18,
     "source_doc": "carrier_agreements/carrier_b_2026.pdf"},
    {"sla_id": "sla_c_txnm", "carrier_id": "carrier_c", "lane_id": "TX-NM",
     "surcharge_rate": 0.05, "weight_threshold_lb": 12000, "transit_hours": 22,
     "source_doc": "carrier_agreements/carrier_c_2026.pdf"},
    {"sla_id": "sla_c_txaz", "carrier_id": "carrier_c", "lane_id": "TX-AZ",
     "surcharge_rate": 0.12, "weight_threshold_lb": 10000, "transit_hours": 36,
     "source_doc": "carrier_agreements/carrier_c_2026.pdf"},
]


def _upsert(coll, key_fields: tuple[str, ...], docs: list[dict], now: str, label: str) -> None:
    for d in docs:
        doc = {**d, "realm_id": REALM_ID, "updated_at": now}
        filt = {"realm_id": REALM_ID, **{k: d[k] for k in key_fields}}
        coll.update_one(filt, {"$set": doc, "$setOnInsert": {"created_at": now}}, upsert=True)
    print(f"  [{label}] upserted {len(docs)} docs")


def main() -> None:
    carriers, lanes, slas, serves = get_kg_collections()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Seeding KG for realm '{REALM_ID}':")
    _upsert(carriers, ("carrier_id",), CARRIERS, now, "kg_carriers")
    _upsert(lanes, ("lane_id",), LANES, now, "kg_lanes")
    _upsert(slas, ("sla_id",), SLAS, now, "kg_slas")
    _upsert(serves, ("carrier_id", "lane_id"), SERVES, now, "kg_serves")
    print("KG seed complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
