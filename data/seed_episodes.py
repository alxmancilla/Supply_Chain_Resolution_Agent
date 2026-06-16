"""Pre-seed Session 1 episodic memories into `agent_episodes`.

Usage:
    python -m data.seed_episodes
"""
from __future__ import annotations

import hashlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.memory import get_episodes_store, memory_namespace
from core.settings import get_settings

_SETTINGS = get_settings()
REALM_ID = _SETTINGS.realm_id
USER_ID = _SETTINGS.user_id

SESSION_ONE_EPISODES: list[dict[str, str]] = [
    {
        "summary": "User shipped 18,000 lbs from El Paso TX to Phoenix AZ; recommended Carrier A under the TX-AZ route guide.",
        "lane": "TX-AZ",
        "recommendation": "Carrier A dedicated dry-van",
        "outcome": "carrier booked, no surcharge",
        "occurred_at": "2026-04-08T14:22:00+00:00",
    },
    {
        "summary": "User requested expedited Dallas TX to Tucson AZ at 22,000 lbs; surfaced [REQUIRES HUMAN APPROVAL] because total exceeded $10,000.",
        "lane": "TX-AZ",
        "recommendation": "Carrier A expedited + approval workflow",
        "outcome": "requires human approval",
        "occurred_at": "2026-05-21T09:10:00+00:00",
    },
    {
        "summary": "User asked about a 9,500 lb Houston TX to San Antonio TX shipment; recommended Carrier A on the TX-TX lane with no surcharge.",
        "lane": "TX-TX",
        "recommendation": "Carrier A dedicated dry-van",
        "outcome": "carrier recommended, awaiting booking",
        "occurred_at": "2026-05-30T16:45:00+00:00",
    },
]


def main() -> None:
    store = get_episodes_store()
    namespace = memory_namespace(REALM_ID, USER_ID)
    print(f"Seeding {len(SESSION_ONE_EPISODES)} Session 1 episodes into namespace {namespace}.")
    for episode in SESSION_ONE_EPISODES:
        digest = hashlib.sha1(episode["summary"].encode("utf-8")).hexdigest()[:16]
        key = f"ep_seed_{digest}"
        store.put(namespace, key=key, value=episode)
        print(f"  put {key}: {episode['summary'][:70]}{'...' if len(episode['summary']) > 70 else ''}")
    print("Episode seed complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
