"""Live-traffic eval: judge recent agent turns and write scores to `eval_runs`.

Reads checkpoints written by `MongoDBSaver` over a rolling window (default
24h), takes the terminal checkpoint per `thread_id`, extracts
`(user_question, agent_reply, retrieved_context)` from `channel_values`,
runs three LLM judges (`faithfulness`, `answer_relevancy`,
`context_relevancy`) over each turn, and writes a single document to the
`eval_runs` collection with per-judge means + per-turn detail. Prints the
same payload to stdout as JSON.

Usage:
    .venv/bin/python -m tools.eval_live_traffic
    .venv/bin/python -m tools.eval_live_traffic --window-hours 24
    .venv/bin/python -m tools.eval_live_traffic --limit 50 --dry-run
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from agent.memory import (
    DB_NAME,
    EVAL_RUNS_COLLECTION,
    get_checkpointer,
    get_mongo_client,
)
from core.protocols import ChatProvider
from core.providers.registry import get_chat_provider
from evals.judges import JUDGE_NAMES, JudgeScore, run_all

_CONTEXT_CHANNELS = (
    "rag_context", "kg_context", "ltm_context",
    "episodic_context", "procedural_context",
)


def _last_of(messages: Iterable[Any], cls: type) -> Any | None:
    for msg in reversed(list(messages or [])):
        if isinstance(msg, cls):
            return msg
    return None


def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


def extract_turn(channel_values: dict[str, Any]) -> dict[str, Any] | None:
    """Pull (question, answer, context_chunks) from a checkpoint's channel_values.

    Returns None if either side of the turn is missing — the judges have
    nothing to score on a partial state.
    """
    messages = channel_values.get("messages") or []
    user = _last_of(messages, HumanMessage)
    agent = _last_of(messages, AIMessage)
    if not user or not agent:
        return None
    question = _msg_text(user).strip()
    answer = _msg_text(agent).strip()
    if not (question and answer):
        return None
    context_chunks = [
        str(channel_values.get(ch) or "").strip()
        for ch in _CONTEXT_CHANNELS
        if str(channel_values.get(ch) or "").strip()
    ]
    return {"question": question, "answer": answer, "context": context_chunks}


def iter_recent_turns(saver, *, window_hours: int, limit: int | None) -> Iterable[dict[str, Any]]:
    """Yield one extracted turn per thread_id, newest first, within the window.

    `MongoDBSaver.list(config=None)` returns checkpoints newest-first; we
    pick the first one we see per `thread_id` (which is the terminal
    checkpoint for that turn).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    seen_threads: set[str] = set()
    yielded = 0
    for tup in saver.list(None):
        ts_raw = tup.checkpoint.get("ts")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")) if ts_raw else None
        except (AttributeError, ValueError):
            ts = None
        if ts is not None and ts < cutoff:
            continue
        thread_id = (tup.config.get("configurable") or {}).get("thread_id")
        if not thread_id or thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)
        turn = extract_turn(tup.checkpoint.get("channel_values") or {})
        if turn is None:
            continue
        turn["thread_id"] = thread_id
        turn["checkpoint_ts"] = ts_raw
        yield turn
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _score_one(chat: ChatProvider, turn: dict[str, Any]) -> dict[str, JudgeScore]:
    return run_all(
        chat,
        question=turn["question"],
        answer=turn["answer"],
        context=turn["context"],
    )


def aggregate(per_turn: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Mean score per judge across the per-turn records (n counts non-empty scores)."""
    out: dict[str, dict[str, float | int]] = {}
    for name in JUDGE_NAMES:
        values = [t["scores"][name]["score"] for t in per_turn if name in t.get("scores", {})]
        out[name] = {
            "mean": round(statistics.fmean(values), 4) if values else 0.0,
            "n": len(values),
        }
    return out


def build_run_payload(per_turn: list[dict[str, Any]], *, window_hours: int) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "n_turns": len(per_turn),
        "scores": aggregate(per_turn),
        "per_turn": per_turn,
    }


def score_turns(chat: ChatProvider, turns: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    per_turn: list[dict[str, Any]] = []
    for turn in turns:
        scores = _score_one(chat, turn)
        per_turn.append({
            "thread_id": turn.get("thread_id"),
            "checkpoint_ts": turn.get("checkpoint_ts"),
            "question": turn["question"],
            "answer_preview": turn["answer"][:240],
            "context_chunks": len(turn["context"]),
            "scores": {name: s.model_dump() for name, s in scores.items()},
        })
    return per_turn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window-hours", type=int, default=24,
                        help="How far back to look in `checkpoints` (default 24).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of turns to score (default: all in window).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the JSON payload but do not insert into eval_runs.")
    parser.add_argument("--out", default=None,
                        help="Optional path to also write the JSON payload to disk.")
    args = parser.parse_args(argv)

    saver = get_checkpointer()
    chat = get_chat_provider()
    turns = list(iter_recent_turns(saver, window_hours=args.window_hours, limit=args.limit))
    per_turn = score_turns(chat, turns)
    payload = build_run_payload(per_turn, window_hours=args.window_hours)

    if not args.dry_run:
        get_mongo_client()[DB_NAME][EVAL_RUNS_COLLECTION].insert_one(dict(payload))
    print(json.dumps(payload, indent=2, default=str))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
