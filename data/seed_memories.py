"""Pre-seed Session 1 long-term memories into `agent_memories`.

Usage:
    python -m data.seed_memories
"""
from __future__ import annotations

import hashlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.memory import get_store, memory_namespace
from core.settings import get_settings

_SETTINGS = get_settings()
REALM_ID = _SETTINGS.realm_id
USER_ID = _SETTINGS.user_id

SESSION_ONE_MEMORIES: list[str] = [
    "User prefers Carrier A for TX-AZ lanes. Carrier B was evaluated but carries variable surcharge risk — a prior shipment saw 28% cost overrun.",
    "User approval threshold is $10,000 on the TX-AZ lane.",
]


def main() -> None:
    store = get_store()
    namespace = memory_namespace(REALM_ID, USER_ID)
    print(f"Seeding {len(SESSION_ONE_MEMORIES)} Session 1 memories into namespace {namespace}.")
    for memory in SESSION_ONE_MEMORIES:
        digest = hashlib.sha1(memory.encode("utf-8")).hexdigest()[:16]
        key = f"mem_seed_{digest}"
        store.put(namespace, key=key, value={"content": memory})
        print(f"  put {key}: {memory[:60]}{'...' if len(memory) > 60 else ''}")
    print("Memory seed complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
