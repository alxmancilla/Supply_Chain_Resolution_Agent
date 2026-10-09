"""Read-only, tenant-scoped view model for live-traffic evaluation runs.

The Streamlit app uses these pure helpers to render `eval_runs`; keeping the
MongoDB query and aggregation here makes the dashboard independently testable
and prevents presentation code from accidentally broadening tenant scope.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from evals.judges import JUDGE_NAMES


def fetch_eval_runs(
    collection: Any,
    *,
    realm_id: str,
    agent_id: str,
    days: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch newest evaluation runs for exactly one realm / agent pair.

    New run timestamps are ISO-8601 UTC strings, whose lexical ordering matches
    chronological ordering. Explicit ownership filters intentionally exclude
    legacy records with no scope metadata rather than risk cross-tenant display.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    cursor = collection.find(
        {
            "realm_id": realm_id,
            "agent_id": agent_id,
            "run_at": {"$gte": cutoff},
        },
        {"_id": 0},
    ).sort("run_at", -1).limit(limit)
    return [dict(run) for run in cursor]


def summarize_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return run counts and n-weighted per-judge means for dashboard cards."""
    materialized = list(runs)
    judges: dict[str, dict[str, float | int]] = {}
    for name in JUDGE_NAMES:
        weighted_total = 0.0
        n_total = 0
        for run in materialized:
            score = (run.get("scores") or {}).get(name) or {}
            mean = score.get("mean")
            n = score.get("n")
            if isinstance(mean, (int, float)) and isinstance(n, int) and n > 0:
                weighted_total += float(mean) * n
                n_total += n
        judges[name] = {
            "mean": round(weighted_total / n_total, 4) if n_total else 0.0,
            "n": n_total,
        }
    return {
        "n_runs": len(materialized),
        "n_turn_samples": sum(int(run.get("n_turns") or 0) for run in materialized),
        "judges": judges,
    }


def trend_rows(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten one row per run for a compact score-trend table."""
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: str(r.get("run_at") or "")):
        scores = run.get("scores") or {}
        row: dict[str, Any] = {
            "run_at": run.get("run_at", ""),
            "run_id": run.get("run_id", ""),
            "turns": int(run.get("n_turns") or 0),
        }
        for name in JUDGE_NAMES:
            value = (scores.get(name) or {}).get("mean")
            row[name] = round(float(value), 3) if isinstance(value, (int, float)) else None
        rows.append(row)
    return rows


def low_score_turns(runs: Iterable[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    """Flatten per-turn records, ranking by the lowest available judge score."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_at = run.get("run_at", "")
        run_id = run.get("run_id", "")
        for turn in run.get("per_turn") or []:
            scores = turn.get("scores") or {}
            values = [
                float((scores.get(name) or {}).get("score"))
                for name in JUDGE_NAMES
                if isinstance((scores.get(name) or {}).get("score"), (int, float))
            ]
            if not values:
                continue
            weakest_name = min(
                (name for name in JUDGE_NAMES if isinstance((scores.get(name) or {}).get("score"), (int, float))),
                key=lambda name: float(scores[name]["score"]),
            )
            rows.append({
                "run_at": run_at,
                "run_id": run_id,
                "thread_id": turn.get("thread_id", ""),
                "checkpoint_ts": turn.get("checkpoint_ts", ""),
                "question": turn.get("question", ""),
                "answer_preview": turn.get("answer_preview", ""),
                "context_chunks": int(turn.get("context_chunks") or 0),
                "lowest_score": round(min(values), 3),
                "weakest_judge": weakest_name,
                "reason": (scores.get(weakest_name) or {}).get("reason", ""),
                "scores": scores,
            })
    return sorted(rows, key=lambda row: row["lowest_score"])[:limit]


__all__ = ["fetch_eval_runs", "summarize_runs", "trend_rows", "low_score_turns"]