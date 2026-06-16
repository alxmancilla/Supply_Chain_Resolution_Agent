"""Delete LTM rows for a given (realm_id, user_id) namespace.

By default targets all user-scoped LTM collections (semantic + episodic).
Procedural memory is tenant-scoped, not user-scoped, so it is NOT touched
by --user; use --type procedural --realm <realm> to clear those.

Usage:
    .venv/bin/python -m tools.cleanup_memories --user user-vaibhav
    .venv/bin/python -m tools.cleanup_memories --user user-demo --type semantic --yes
    .venv/bin/python -m tools.cleanup_memories --type procedural --realm customer-tenant-001 --yes
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from agent.memory import (
    DB_NAME,
    EPISODES_COLLECTION,
    MEMORIES_COLLECTION,
    PROCEDURES_COLLECTION,
    get_mongo_client,
)

USER_SCOPED = {
    "semantic": MEMORIES_COLLECTION,
    "episodic": EPISODES_COLLECTION,
}


def _delete_user_scoped(coll_name: str, namespace: list[str], yes: bool) -> None:
    coll = get_mongo_client()[DB_NAME][coll_name]
    matched = coll.count_documents({"namespace": namespace})
    print(f"Matched {matched} doc(s) in '{coll_name}' for namespace {namespace}.")
    if matched == 0:
        return
    if not yes:
        confirm = input(f"Type 'delete' to remove from {coll_name}: ").strip().lower()
        if confirm != "delete":
            print("  Aborted.")
            return
    result = coll.delete_many({"namespace": namespace})
    print(f"  Deleted {result.deleted_count} doc(s) from '{coll_name}'.")


def _delete_procedures(realm: str, yes: bool) -> None:
    coll = get_mongo_client()[DB_NAME][PROCEDURES_COLLECTION]
    matched = coll.count_documents({"realm_id": realm})
    print(f"Matched {matched} doc(s) in '{PROCEDURES_COLLECTION}' for realm '{realm}'.")
    if matched == 0:
        return
    if not yes:
        confirm = input(f"Type 'delete' to remove from {PROCEDURES_COLLECTION}: ").strip().lower()
        if confirm != "delete":
            print("  Aborted.")
            return
    result = coll.delete_many({"realm_id": realm})
    print(f"  Deleted {result.deleted_count} doc(s) from '{PROCEDURES_COLLECTION}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realm",
        default=os.environ.get("REALM_ID", "customer-tenant-001"),
        help="realm_id to target (default: REALM_ID or customer-tenant-001)",
    )
    parser.add_argument("--user", help="user_id to delete (required unless --type procedural)")
    parser.add_argument(
        "--type",
        choices=["semantic", "episodic", "procedural", "all"],
        default="all",
        help="which LTM collection(s) to clean (default: all)",
    )
    parser.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    args = parser.parse_args()

    if args.type in {"semantic", "episodic", "all"}:
        if not args.user:
            parser.error("--user is required for semantic/episodic/all cleanup")
        namespace = [args.realm, args.user]
        targets = (
            [USER_SCOPED[args.type]] if args.type in USER_SCOPED else list(USER_SCOPED.values())
        )
        for coll_name in targets:
            _delete_user_scoped(coll_name, namespace, args.yes)

    if args.type in {"procedural", "all"}:
        _delete_procedures(args.realm, args.yes)


if __name__ == "__main__":
    main()
