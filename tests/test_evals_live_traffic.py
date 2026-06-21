"""Unit tests for live-traffic eval: judges, checkpoint extraction, aggregation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from evals.judges import (
    JUDGE_NAMES,
    JudgeScore,
    judge_answer_relevancy,
    judge_context_relevancy,
    judge_faithfulness,
    run_all,
)
from tests.fakes import FakeChatProvider
from tools.eval_live_traffic import (
    aggregate,
    build_run_payload,
    extract_turn,
    iter_recent_turns,
    score_turns,
)


# ----------------------------- JudgeScore basics ----------------------------


def test_judge_score_clips_above_one_and_below_zero():
    assert JudgeScore(score=2.5, reason="hi").score == 1.0
    assert JudgeScore(score=-0.3, reason="hi").score == 0.0


def test_judge_score_rejects_non_floatable():
    with pytest.raises(Exception):
        JudgeScore(score="not-a-number", reason="x")  # type: ignore[arg-type]


# ----------------------------- Individual judges ----------------------------


def test_judge_faithfulness_returns_parsed_score():
    chat = FakeChatProvider(reply='{"score": 0.8, "reason": "all claims supported"}')
    result = judge_faithfulness(
        chat, question="q", answer="a", context=["c1", "c2"],
    )
    assert result.score == 0.8
    assert "supported" in result.reason
    # The prompt should contain the joined context separator.
    assert "---" in chat.calls[0]
    assert "AGENT REPLY" in chat.calls[0]


def test_judge_answer_relevancy_uses_question_answer_only():
    chat = FakeChatProvider(reply='{"score": 0.6, "reason": "partial"}')
    result = judge_answer_relevancy(chat, question="why?", answer="because")
    assert result.score == 0.6
    # No "RETRIEVED CONTEXT" header in the answer-relevancy prompt.
    assert "RETRIEVED CONTEXT" not in chat.calls[0]


def test_judge_context_relevancy_no_chunks_uses_placeholder():
    chat = FakeChatProvider(reply='{"score": 0.0, "reason": "no context"}')
    result = judge_context_relevancy(chat, question="q", context=[])
    assert result.score == 0.0
    assert "(no context retrieved)" in chat.calls[0]


def test_judge_falls_back_to_zero_when_retries_exhausted():
    chat = FakeChatProvider(reply="not-json-at-all")
    result = judge_faithfulness(
        chat, question="q", answer="a", context=["c"], max_attempts=2,
    )
    assert result.score == 0.0
    assert result.reason.startswith("judge_parse_failed")


def test_run_all_returns_all_three_judges():
    chat = FakeChatProvider(replies=[
        '{"score": 1.0, "reason": "f"}',
        '{"score": 0.9, "reason": "a"}',
        '{"score": 0.7, "reason": "c"}',
    ])
    scores = run_all(chat, question="q", answer="a", context=["c"])
    assert set(scores.keys()) == set(JUDGE_NAMES)
    assert scores["faithfulness"].score == 1.0
    assert scores["answer_relevancy"].score == 0.9
    assert scores["context_relevancy"].score == 0.7


# ----------------------------- extract_turn --------------------------------


def test_extract_turn_happy_path_collects_context_chunks():
    cv = {
        "messages": [
            HumanMessage(content="ship 1000 lb to Phoenix"),
            AIMessage(content="Carrier A on TX-AZ"),
        ],
        "rag_context": "TX-AZ lane chunk",
        "kg_context": "Carrier A serves TX-AZ",
        "ltm_context": "",
        "episodic_context": "  ",
        "procedural_context": "always show kg",
    }
    out = extract_turn(cv)
    assert out is not None
    assert out["question"] == "ship 1000 lb to Phoenix"
    assert out["answer"] == "Carrier A on TX-AZ"
    # Three non-blank contexts retained, blank ones dropped.
    assert out["context"] == [
        "TX-AZ lane chunk", "Carrier A serves TX-AZ", "always show kg",
    ]


def test_extract_turn_returns_none_when_no_human_message():
    cv = {"messages": [AIMessage(content="orphan reply")]}
    assert extract_turn(cv) is None


def test_extract_turn_returns_none_when_no_agent_reply():
    cv = {"messages": [HumanMessage(content="hi")]}
    assert extract_turn(cv) is None


def test_extract_turn_returns_none_on_blank_content():
    cv = {"messages": [HumanMessage(content="  "), AIMessage(content="x")]}
    assert extract_turn(cv) is None


# ----------------------------- iter_recent_turns ---------------------------


class _FakeTuple:
    def __init__(self, *, thread_id: str, ts: str, channel_values: dict[str, Any]):
        self.config = {"configurable": {"thread_id": thread_id}}
        self.checkpoint = {"ts": ts, "channel_values": channel_values}


class _FakeSaver:
    def __init__(self, tuples: list[_FakeTuple]):
        self._tuples = tuples

    def list(self, _config):
        # MongoDBSaver.list returns newest-first; mimic that ordering.
        return iter(self._tuples)


def _good_state(text: str = "hello") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=text), AIMessage(content="ok")],
        "rag_context": "some context",
    }


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def test_iter_recent_turns_dedups_by_thread_id_keeping_newest():
    now = datetime.now(timezone.utc)
    saver = _FakeSaver([
        # Newest first per thread t1, then an older one for t1 (skipped).
        _FakeTuple(thread_id="t1", ts=_iso(now), channel_values=_good_state("newest")),
        _FakeTuple(thread_id="t1", ts=_iso(now - timedelta(minutes=5)),
                   channel_values=_good_state("older")),
        _FakeTuple(thread_id="t2", ts=_iso(now - timedelta(hours=1)),
                   channel_values=_good_state("other")),
    ])
    turns = list(iter_recent_turns(saver, window_hours=24, limit=None))
    assert [t["thread_id"] for t in turns] == ["t1", "t2"]
    assert turns[0]["question"] == "newest"


def test_iter_recent_turns_filters_outside_window():
    now = datetime.now(timezone.utc)
    saver = _FakeSaver([
        _FakeTuple(thread_id="t1", ts=_iso(now - timedelta(hours=48)),
                   channel_values=_good_state("stale")),
        _FakeTuple(thread_id="t2", ts=_iso(now - timedelta(hours=1)),
                   channel_values=_good_state("fresh")),
    ])
    turns = list(iter_recent_turns(saver, window_hours=24, limit=None))
    assert [t["thread_id"] for t in turns] == ["t2"]


def test_iter_recent_turns_respects_limit():
    now = datetime.now(timezone.utc)
    saver = _FakeSaver([
        _FakeTuple(thread_id=f"t{i}", ts=_iso(now), channel_values=_good_state(f"q{i}"))
        for i in range(5)
    ])
    turns = list(iter_recent_turns(saver, window_hours=24, limit=2))
    assert len(turns) == 2


def test_iter_recent_turns_skips_partial_states():
    now = datetime.now(timezone.utc)
    saver = _FakeSaver([
        _FakeTuple(thread_id="t1", ts=_iso(now), channel_values={"messages": []}),
        _FakeTuple(thread_id="t2", ts=_iso(now), channel_values=_good_state("ok")),
    ])
    turns = list(iter_recent_turns(saver, window_hours=24, limit=None))
    assert [t["thread_id"] for t in turns] == ["t2"]


def test_iter_recent_turns_tolerates_missing_ts():
    saver = _FakeSaver([
        _FakeTuple(thread_id="t1", ts="", channel_values=_good_state("undated")),
    ])
    # No ts → cannot prove it's outside the window → kept.
    turns = list(iter_recent_turns(saver, window_hours=24, limit=None))
    assert len(turns) == 1


# ----------------------------- aggregate + payload -------------------------


def test_aggregate_returns_mean_and_n_per_judge():
    per_turn = [
        {"scores": {name: {"score": 1.0, "reason": ""} for name in JUDGE_NAMES}},
        {"scores": {name: {"score": 0.5, "reason": ""} for name in JUDGE_NAMES}},
    ]
    agg = aggregate(per_turn)
    for name in JUDGE_NAMES:
        assert agg[name]["n"] == 2
        assert agg[name]["mean"] == 0.75


def test_aggregate_empty_input_returns_zero_means():
    agg = aggregate([])
    for name in JUDGE_NAMES:
        assert agg[name]["mean"] == 0.0
        assert agg[name]["n"] == 0


def test_build_run_payload_shape():
    payload = build_run_payload([], window_hours=24)
    assert set(payload.keys()) == {"run_id", "run_at", "window_hours",
                                   "n_turns", "scores", "per_turn"}
    assert payload["window_hours"] == 24
    assert payload["n_turns"] == 0
    assert len(payload["run_id"]) == 32  # uuid4 hex


def test_score_turns_emits_one_record_per_turn_with_all_judges():
    chat = FakeChatProvider(replies=[
        '{"score": 0.9, "reason": "f"}',
        '{"score": 0.8, "reason": "a"}',
        '{"score": 0.7, "reason": "c"}',
    ])
    turns = [{
        "thread_id": "t1", "checkpoint_ts": "2026-06-21T00:00:00+00:00",
        "question": "q", "answer": "a", "context": ["c"],
    }]
    per_turn = score_turns(chat, turns)
    assert len(per_turn) == 1
    rec = per_turn[0]
    assert rec["thread_id"] == "t1"
    assert rec["context_chunks"] == 1
    assert set(rec["scores"].keys()) == set(JUDGE_NAMES)
    assert rec["scores"]["faithfulness"]["score"] == 0.9
