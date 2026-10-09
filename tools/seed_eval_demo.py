"""Upsert one clearly labeled synthetic eval run for dashboard rehearsals.

This does not evaluate agent traffic and never writes checkpoints. Use the
real `tools.eval_live_traffic` command for the actual MongoDB automation demo.

Usage:
    python -m tools.seed_eval_demo
    python -m tools.seed_eval_demo --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from agent.memory import DB_NAME, EVAL_RUNS_COLLECTION, get_mongo_client
from core.settings import get_settings
from evals.judges import JUDGE_NAMES
from tools.eval_live_traffic import aggregate


def _turn(
    thread_id: str, question: str, answer: str, scores: tuple[float, float, float], reason: str,
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "checkpoint_ts": "synthetic-demo",
        "question": question,
        "answer_preview": answer,
        "context_chunks": 3,
        "scores": {
            name: {"score": score, "reason": reason if score == min(scores) else "demo fixture"}
            for name, score in zip(JUDGE_NAMES, scores)
        },
    }


def build_demo_run(*, realm_id: str, agent_id: str) -> dict[str, Any]:
    """Build an idempotent fixture with one intentionally diagnosable weak turn."""
    per_turn = [
        _turn(
            "demo-eval-grounded", "Which carrier serves TX-AZ without fuel surcharge?",
            "Carrier A serves TX-AZ and has no fuel surcharge under the documented threshold.",
            (0.98, 0.97, 0.95), "grounded demo response",
        ),
        _turn(
            "demo-eval-policy", "What is the same-day Austin-Dallas hours-of-service limit?",
            "The policy requires the driver to remain within the documented hours-of-service limit.",
            (0.93, 0.96, 0.91), "grounded policy response",
        ),
        _turn(
            "demo-eval-drilldown", "What carrier should I choose?",
            "Carrier B is guaranteed to be fastest.",
            (0.42, 0.88, 0.62), "intentional drill-down example: unsupported guarantee",
        ),
    ]
    stable_id = hashlib.sha1(f"{realm_id}|{agent_id}|mongodb-eval-demo".encode()).hexdigest()[:16]
    return {
        "run_id": f"demo-{stable_id}",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "realm_id": realm_id,
        "agent_id": agent_id,
        "window_hours": 24,
        "n_turns": len(per_turn),
        "scores": aggregate(per_turn),
        "per_turn": per_turn,
        "demo_fixture": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    settings = get_settings()
    parser.add_argument("--realm-id", default=settings.realm_id,
                        help="Realm to scope the synthetic fixture to (default: REALM_ID).")
    parser.add_argument("--agent-id", default=settings.agent_id,
                        help="Agent to scope the synthetic fixture to (default: AGENT_ID).")
    parser.add_argument("--dry-run", action="store_true", help="Print the fixture without upserting it.")
    args = parser.parse_args(argv)
    payload = build_demo_run(realm_id=args.realm_id, agent_id=args.agent_id)
    if not args.dry_run:
        get_mongo_client()[DB_NAME][EVAL_RUNS_COLLECTION].update_one(
            {"run_id": payload["run_id"]}, {"$set": payload}, upsert=True,
        )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())