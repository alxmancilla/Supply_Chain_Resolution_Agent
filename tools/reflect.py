"""Run the `LLMMemoryReflector` against semantic and/or episodic LTM.

Consolidates near-duplicate facts (above `--threshold` cosine similarity)
into a single canonical row and tombstones the originals. Tombstoned rows
are filtered out of subsequent retrievals by the memory layer.

Usage:
    .venv/bin/python -m tools.reflect --user user-demo
    .venv/bin/python -m tools.reflect --user user-demo --type semantic
    .venv/bin/python -m tools.reflect --user user-demo --threshold 0.92 --dry-run
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.memory import (
    DB_NAME,
    EPISODES_COLLECTION,
    MEMORIES_COLLECTION,
    get_mongo_client,
)
from core.memory.reflector import LLMMemoryReflector, MongoMemoryAdmin, ReflectionReport
from core.providers.registry import get_chat_provider, get_embedding_provider
from core.settings import get_settings

TARGETS = {
    "semantic": (MEMORIES_COLLECTION, "content"),
    "episodic": (EPISODES_COLLECTION, "summary"),
}


def _run_one(*, kind: str, realm_id: str, user_id: str, threshold: float, dry_run: bool) -> ReflectionReport:
    coll_name, field = TARGETS[kind]
    coll = get_mongo_client()[DB_NAME][coll_name]
    embeddings = get_embedding_provider()
    chat = get_chat_provider()
    admin = MongoMemoryAdmin(collection=coll, content_field=field, embeddings=embeddings)

    if dry_run:
        facts = admin.list_live(realm_id, user_id)
        from core.memory.reflector import _greedy_cluster
        clusters = _greedy_cluster(facts, threshold)
        merge_clusters = [c for c in clusters if len(c) >= 2]
        print(f"[{kind}] {len(facts)} live facts, "
              f"{len(merge_clusters)} merge-clusters, "
              f"{sum(len(c) for c in merge_clusters)} rows would be tombstoned.")
        for idx, cluster in enumerate(merge_clusters, 1):
            print(f"  cluster {idx}:")
            for f in cluster:
                preview = (f.content[:80] + "...") if len(f.content) > 80 else f.content
                print(f"    - {f.key}: {preview}")
        return ReflectionReport(clusters_found=len(merge_clusters))

    reflector = LLMMemoryReflector(admin=admin, chat=chat, similarity_threshold=threshold)
    report = reflector.reflect(realm_id, user_id)
    print(f"[{kind}] clusters_consolidated={report.clusters_found} "
          f"canonical_written={len(report.canonical_written)} "
          f"tombstoned={len(report.tombstoned_keys)} "
          f"singletons_skipped={report.skipped_singletons}")
    return report


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Reflect over LTM and consolidate near-duplicates.")
    parser.add_argument("--realm", default=s.realm_id, help="Tenant realm id.")
    parser.add_argument("--user", default=s.user_id, help="User id whose namespace to reflect.")
    parser.add_argument("--type", choices=("semantic", "episodic", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=0.88,
                        help="Cosine similarity threshold for clustering (default 0.88 — looser than dedup).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print clusters that would be consolidated, no writes.")
    args = parser.parse_args(argv)

    kinds = ("semantic", "episodic") if args.type == "both" else (args.type,)
    for kind in kinds:
        _run_one(kind=kind, realm_id=args.realm, user_id=args.user,
                 threshold=args.threshold, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
