"""Pre-seed procedural memory rules into `agent_procedures`.

These rules are injected verbatim into the system prompt on every turn,
so they directly steer the agent's behavior for this tenant.

Usage:
    python -m data.seed_procedures
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from agent.memory import get_procedures_collection
from core.settings import get_settings

_SETTINGS = get_settings()
REALM_ID = _SETTINGS.realm_id
AGENT_ID = _SETTINGS.agent_id

PROCEDURES: list[dict[str, str]] = [
    {
        "rule_id": "proc_001",
        "category": "units",
        "rule": "Always express shipment weights in both pounds and kilograms (1 lb = 0.4536 kg) when reporting a recommendation.",
    },
    {
        "rule_id": "proc_002",
        "category": "escalation",
        "rule": "If a recommendation involves Carrier B on the TX-AZ lane, append a one-line warning that Carrier B has documented variable surcharge risk on this lane.",
    },
    {
        "rule_id": "proc_003",
        "category": "formatting",
        "rule": "End every response with a 'Sources:' line listing the route guide, SLA, or policy filenames you cited.",
    },
]


def main() -> None:
    coll = get_procedures_collection()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Seeding {len(PROCEDURES)} procedural rules for realm '{REALM_ID}', agent '{AGENT_ID}'.")
    for rule in PROCEDURES:
        doc = {
            **rule,
            "realm_id": REALM_ID,
            "agent_id": AGENT_ID,
            "active": True,
            "updated_at": now,
        }
        coll.update_one(
            {"realm_id": REALM_ID, "agent_id": AGENT_ID, "rule_id": rule["rule_id"]},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        print(f"  upsert {rule['rule_id']} ({rule['category']}): {rule['rule'][:70]}{'...' if len(rule['rule']) > 70 else ''}")
    print("Procedure seed complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
