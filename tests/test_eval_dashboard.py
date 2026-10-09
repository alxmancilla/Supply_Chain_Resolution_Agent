"""Unit tests for the read-only, tenant-scoped evaluation dashboard model."""
from __future__ import annotations

from typing import Any

from evals.live_dashboard import fetch_eval_runs, low_score_turns, summarize_runs, trend_rows
from tools.seed_eval_demo import build_demo_run


def _run(
    run_id: str,
    *,
    run_at: str = "2026-10-09T12:00:00+00:00",
    realm_id: str = "r1",
    agent_id: str = "a1",
    faithfulness: float = 0.8,
    answer_relevancy: float = 0.9,
    context_relevancy: float = 0.7,
    n: int = 2,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_at": run_at,
        "realm_id": realm_id,
        "agent_id": agent_id,
        "n_turns": n,
        "scores": {
            "faithfulness": {"mean": faithfulness, "n": n},
            "answer_relevancy": {"mean": answer_relevancy, "n": n},
            "context_relevancy": {"mean": context_relevancy, "n": n},
        },
        "per_turn": [],
    }


class _Cursor(list):
    def sort(self, field: str, direction: int):
        assert field == "run_at" and direction == -1
        return _Cursor(sorted(self, key=lambda row: row["run_at"], reverse=True))

    def limit(self, count: int):
        return _Cursor(self[:count])


class _Collection:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.query: dict[str, Any] | None = None
        self.projection: dict[str, int] | None = None

    def find(self, query: dict[str, Any], projection: dict[str, int]):
        self.query = query
        self.projection = projection
        rows = [
            row for row in self.rows
            if row.get("realm_id") == query["realm_id"]
            and row.get("agent_id") == query["agent_id"]
            and row.get("run_at", "") >= query["run_at"]["$gte"]
        ]
        return _Cursor(rows)


def test_fetch_eval_runs_is_owner_scoped_and_newest_first():
    coll = _Collection([
        _run("old", run_at="2026-10-08T12:00:00+00:00"),
        _run("new", run_at="2026-10-09T12:00:00+00:00"),
        _run("other-realm", realm_id="r2"),
        _run("other-agent", agent_id="a2"),
    ])
    # A very large window keeps fixture timestamps independent of wall-clock time.
    rows = fetch_eval_runs(coll, realm_id="r1", agent_id="a1", days=10_000, limit=1)
    assert [row["run_id"] for row in rows] == ["new"]
    assert coll.query is not None
    assert coll.query["realm_id"] == "r1"
    assert coll.query["agent_id"] == "a1"
    assert coll.projection == {"_id": 0}


def test_summarize_runs_weights_means_by_judge_sample_count():
    small = _run("small", faithfulness=1.0, n=1)
    large = _run("large", faithfulness=0.5, n=3)
    summary = summarize_runs([small, large])
    assert summary["n_runs"] == 2
    assert summary["n_turn_samples"] == 4
    assert summary["judges"]["faithfulness"] == {"mean": 0.625, "n": 4}


def test_trend_rows_orders_oldest_to_newest_for_charting():
    rows = trend_rows([
        _run("new", run_at="2026-10-09T12:00:00+00:00"),
        _run("old", run_at="2026-10-08T12:00:00+00:00"),
    ])
    assert [row["run_id"] for row in rows] == ["old", "new"]
    assert rows[0]["faithfulness"] == 0.8


def test_low_score_turns_ranks_by_weakest_judge_and_preserves_rationale():
    run = _run("r")
    run["per_turn"] = [
        {
            "thread_id": "strong", "question": "q1", "answer_preview": "a1", "context_chunks": 2,
            "scores": {
                "faithfulness": {"score": 0.9, "reason": "supported"},
                "answer_relevancy": {"score": 0.8, "reason": "relevant"},
                "context_relevancy": {"score": 0.8, "reason": "on topic"},
            },
        },
        {
            "thread_id": "weak", "question": "q2", "answer_preview": "a2", "context_chunks": 1,
            "scores": {
                "faithfulness": {"score": 0.2, "reason": "unsupported claim"},
                "answer_relevancy": {"score": 0.9, "reason": "relevant"},
                "context_relevancy": {"score": 0.7, "reason": "partially on topic"},
            },
        },
    ]
    rows = low_score_turns([run])
    assert [row["thread_id"] for row in rows] == ["weak", "strong"]
    assert rows[0]["weakest_judge"] == "faithfulness"
    assert rows[0]["reason"] == "unsupported claim"


def test_demo_fixture_is_scoped_idempotently_and_contains_a_drill_down_case():
    fixture = build_demo_run(realm_id="r1", agent_id="a1")
    assert fixture["run_id"].startswith("demo-")
    assert fixture["realm_id"] == "r1"
    assert fixture["agent_id"] == "a1"
    assert fixture["demo_fixture"] is True
    weak = low_score_turns([fixture], limit=1)
    assert weak[0]["thread_id"] == "demo-eval-drilldown"
    assert weak[0]["lowest_score"] == 0.42