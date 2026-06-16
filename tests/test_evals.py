"""Unit tests for the eval harness metrics and runner."""
from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import action as action_metric
from evals.metrics import kg as kg_metric
from evals.metrics import latency as latency_metric
from evals.metrics import retrieval as retrieval_metric
from evals.metrics import routing as routing_metric
from evals.runner import diff_against_baseline, render_diff, render_markdown, run_suite
from evals.schemas import MetricResult, SuiteResult
from tests.fakes import (
    FakeChatProvider,
    FakeIntentRouter,
    FakeKnowledgeGraph,
    FakeKnowledgeRetriever,
)


def _write_jsonl(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_routing_metric_scores_match_expectation(tmp_path):
    dataset = _write_jsonl(tmp_path, "intents.jsonl", [
        {"case_id": "a", "user_message": "hi", "expected_intent": "fallback"},
        {"case_id": "b", "user_message": "hi", "expected_intent": "recommend_shipment"},
    ])
    router = FakeIntentRouter(decision={
        "intent_label": "fallback", "branches": ["ltm"], "rationale": "r",
    })
    result = routing_metric.run(router=router, dataset=str(dataset))
    assert result.n == 2
    assert result.passed == 1
    assert result.score == 0.5
    assert result.extras["confusion"] == {"fallback": {"fallback": 1}, "recommend_shipment": {"fallback": 1}}


def test_retrieval_metric_passes_when_all_expected_in_topk(tmp_path):
    dataset = _write_jsonl(tmp_path, "rag.jsonl", [
        {"case_id": "a", "query": "x", "expected_sources": ["doc1.pdf"], "k": 3},
    ])
    retriever = FakeKnowledgeRetriever(hits=[
        {"doc_type": "policy", "source": "doc1.pdf", "text": "t", "score": 0.9, "metadata": {}},
        {"doc_type": "policy", "source": "doc2.pdf", "text": "t", "score": 0.8, "metadata": {}},
    ])
    result = retrieval_metric.run(retriever=retriever, dataset=str(dataset), realm_id="r")
    assert result.n == 1
    assert result.passed == 1
    assert result.score == 1.0
    assert result.extras["mean_recall"] == 1.0


def test_retrieval_metric_partial_recall_fails_but_scores_fraction(tmp_path):
    dataset = _write_jsonl(tmp_path, "rag.jsonl", [
        {"case_id": "a", "query": "x", "expected_sources": ["doc1.pdf", "doc2.pdf"], "k": 1},
    ])
    retriever = FakeKnowledgeRetriever(hits=[
        {"doc_type": "policy", "source": "doc1.pdf", "text": "t", "score": 0.9, "metadata": {}},
    ])
    result = retrieval_metric.run(retriever=retriever, dataset=str(dataset), realm_id="r")
    assert result.passed == 0
    assert result.score == 0.0
    assert result.cases[0].score == 0.5


def test_kg_metric_row_match_via_fake_graph(tmp_path):
    dataset = _write_jsonl(tmp_path, "kg.jsonl", [
        {"case_id": "a", "query": "Which carriers serve TX-AZ?",
         "expected_rows": [{"carrier_id": "carrier_a", "lane_id": "TX-AZ", "hop": 1}]},
    ])
    from core.kg.extractor import RegexEntityExtractor
    graph = FakeKnowledgeGraph(subgraph={
        "edges": [
            {"kind": "serves", "from_id": "carrier_a", "to_id": "TX-AZ", "properties": {"hop": 1}},
        ],
    })
    result = kg_metric.run(
        extractor=RegexEntityExtractor(), graph=graph,
        dataset=str(dataset), realm_id="r",
    )
    assert result.passed == 1
    assert result.extras["mean_row_recall"] == 1.0


def test_action_metric_passes_when_action_and_approval_match(tmp_path):
    dataset = _write_jsonl(tmp_path, "action.jsonl", [
        {"case_id": "low", "user_message": "u", "agent_message": "a",
         "expected_action_type": "create_booking_draft", "expected_requires_approval": False},
        {"case_id": "high", "user_message": "u", "agent_message": "a",
         "expected_action_type": "create_booking_draft", "expected_requires_approval": True},
        {"case_id": "none", "user_message": "u", "agent_message": "a",
         "expected_action_type": "none", "expected_requires_approval": False},
    ])
    chat = FakeChatProvider(replies=[
        '{"action_type":"create_booking_draft","carrier":"A","lane":"TX-TX","weight_lb":1000,"estimated_cost_usd":500,"requires_approval":false,"rationale":"low"}',
        '{"action_type":"create_booking_draft","carrier":"C","lane":"TX-CA","weight_lb":45000,"estimated_cost_usd":18000,"requires_approval":true,"rationale":"high"}',
        '{"action_type":"none","requires_approval":false,"rationale":"none"}',
    ])
    result = action_metric.run(chat=chat, dataset=str(dataset))
    assert result.n == 3
    assert result.passed == 3
    assert result.score == 1.0
    assert result.extras["approval_accuracy"] == 1.0


def test_action_metric_flags_missed_approval(tmp_path):
    dataset = _write_jsonl(tmp_path, "action.jsonl", [
        {"case_id": "high", "user_message": "u", "agent_message": "a",
         "expected_action_type": "create_booking_draft", "expected_requires_approval": True},
    ])
    chat = FakeChatProvider(replies=[
        '{"action_type":"create_booking_draft","carrier":"C","lane":"TX-CA","estimated_cost_usd":18000,"requires_approval":false,"rationale":"missed gate"}',
    ])
    result = action_metric.run(chat=chat, dataset=str(dataset))
    assert result.passed == 0
    assert result.cases[0].actual["requires_approval"] is False
    assert result.extras["approval_accuracy"] == 0.0


def test_latency_metric_reports_percentiles_per_case_and_overall(tmp_path):
    dataset = _write_jsonl(tmp_path, "latency.jsonl", [
        {"case_id": "a", "user_message": "q1"},
        {"case_id": "b", "user_message": "q2"},
    ])
    samples = {
        "q1": [
            {"llm_ttft_ms": 100.0, "llm_ms": 200.0},
            {"llm_ttft_ms": 120.0, "llm_ms": 240.0},
            {"llm_ttft_ms": 110.0, "llm_ms": 220.0},
        ],
        "q2": [
            {"llm_ttft_ms": 300.0, "llm_ms": 600.0},
            {"llm_ttft_ms": 320.0, "llm_ms": 640.0},
            {"llm_ttft_ms": 310.0, "llm_ms": 620.0},
        ],
    }
    queue = {k: list(v) for k, v in samples.items()}

    def invoker(prompt: str):
        return queue[prompt].pop(0)

    result = latency_metric.run(invoker=invoker, dataset=str(dataset), runs=3)

    assert result.n == 2
    assert result.passed == 2
    assert result.score == 1.0
    assert result.extras["runs_per_case"] == 3
    assert result.extras["ttft_ms"]["p50"] == 210.0
    assert result.extras["ttft_ms"]["min"] == 100.0
    assert result.extras["ttft_ms"]["max"] == 320.0
    assert result.cases[0].actual["ttft_ms"]["p50"] == 110.0
    assert result.cases[1].actual["llm_ms"]["p95"] == 638.0


def test_latency_metric_fails_case_on_invoker_error(tmp_path):
    dataset = _write_jsonl(tmp_path, "latency.jsonl", [
        {"case_id": "flaky", "user_message": "q"},
    ])

    calls = {"n": 0}

    def invoker(_prompt: str):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"llm_ttft_ms": 100.0, "llm_ms": 200.0}

    result = latency_metric.run(invoker=invoker, dataset=str(dataset), runs=3)

    assert result.passed == 0
    assert result.cases[0].actual["runs_collected"] == 2
    assert "boom" in result.cases[0].notes


def test_runner_fast_mode_produces_all_four_metrics():
    suite = run_suite("fast", realm_id="r")
    names = [m.name for m in suite.metrics]
    assert names == [
        "intent_accuracy",
        "rag_recall_at_k",
        "kg_row_match",
        "action_planning_accuracy",
    ]
    assert all(m.n > 0 for m in suite.metrics)


def test_render_markdown_table_includes_metric_names():
    suite = run_suite("fast", realm_id="r")
    md = render_markdown(suite)
    assert "intent_accuracy" in md
    assert "rag_recall_at_k" in md
    assert "kg_row_match" in md
    assert "action_planning_accuracy" in md


def _suite(metrics: list[MetricResult]) -> SuiteResult:
    return SuiteResult(mode="test", metrics=metrics)


def _metric(name: str, score: float, extras: dict | None = None) -> MetricResult:
    return MetricResult(
        name=name, dataset=f"{name}.jsonl", n=1, passed=int(score >= 1.0),
        score=score, extras=extras or {},
    )


def test_diff_against_baseline_no_regression_within_tolerance():
    baseline = _suite([_metric("intent_accuracy", 0.95)])
    current = _suite([_metric("intent_accuracy", 0.945)])
    diff = diff_against_baseline(current, baseline, score_tolerance=0.01)
    assert diff["regressions"] == []
    assert diff["rows"][0]["ok"] is True


def test_diff_against_baseline_flags_score_regression():
    baseline = _suite([_metric("intent_accuracy", 0.95)])
    current = _suite([_metric("intent_accuracy", 0.80)])
    diff = diff_against_baseline(current, baseline, score_tolerance=0.01)
    assert len(diff["regressions"]) == 1
    name, kind, base_v, cur_v = diff["regressions"][0]
    assert name == "intent_accuracy" and kind == "score"
    assert base_v == 0.95 and cur_v == 0.80


def test_diff_against_baseline_flags_latency_p95_regression():
    base_extras = {"ttft_ms": {"p95": 100.0}, "llm_ms": {"p95": 200.0}}
    cur_extras = {"ttft_ms": {"p95": 120.0}, "llm_ms": {"p95": 600.0}}
    baseline = _suite([_metric("latency_p50_p95", 1.0, base_extras)])
    current = _suite([_metric("latency_p50_p95", 1.0, cur_extras)])
    diff = diff_against_baseline(current, baseline, latency_factor=1.5)
    names = {name for name, *_ in diff["regressions"]}
    assert names == {"latency_p50_p95.llm_ms.p95"}


def test_diff_against_baseline_treats_new_metrics_as_informational():
    baseline = _suite([_metric("intent_accuracy", 1.0)])
    current = _suite([_metric("intent_accuracy", 1.0), _metric("new_metric", 0.0)])
    diff = diff_against_baseline(current, baseline)
    assert diff["regressions"] == []
    kinds = {r["name"]: r["kind"] for r in diff["rows"]}
    assert kinds["new_metric"] == "new"


def test_render_diff_shows_regression_summary():
    baseline = _suite([_metric("intent_accuracy", 0.95)])
    current = _suite([_metric("intent_accuracy", 0.50)])
    diff = diff_against_baseline(current, baseline)
    md = render_diff(diff)
    assert "Baseline diff" in md
    assert "intent_accuracy" in md
    assert "regression" in md.lower()
    assert "0.95" in md and "0.5" in md
