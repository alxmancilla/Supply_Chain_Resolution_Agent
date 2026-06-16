"""Eval-suite runner.

Three modes:
- `fast` wires fakes from `tests/fakes.py` so the suite runs in CI
  without Atlas or any API key. Used to exercise the metric plumbing.
- `live` resolves the protocols against the real Atlas-backed stack.
  Used to refresh the committed baseline file.
- `latency` drives the compiled graph N times per prompt and reports
  p50/p95 of `llm_ttft_ms` and `llm_ms`. Kept separate so the standard
  modes stay quick.

Usage:
    python -m evals.runner --mode fast
    python -m evals.runner --mode live --baseline evals/baseline.json
    python -m evals.runner --mode latency --runs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from evals.metrics import action as action_metric
from evals.metrics import kg as kg_metric
from evals.metrics import latency as latency_metric
from evals.metrics import retrieval as retrieval_metric
from evals.metrics import routing as routing_metric
from evals.schemas import MetricResult, SuiteResult

DATASETS_DIR = Path(__file__).parent / "datasets"


_ACTION_FAST_REPLIES = [
    '{"action_type":"create_booking_draft","carrier":"Carrier A","lane":"TX-TX","origin":"Austin","destination":"Dallas","weight_lb":15000,"estimated_cost_usd":1850,"requires_approval":false,"rationale":"low-cost TX-TX"}',
    '{"action_type":"create_booking_draft","carrier":"Carrier A","lane":"TX-AZ","origin":"Houston","destination":"Phoenix","weight_lb":12000,"estimated_cost_usd":4200,"requires_approval":false,"rationale":"low-cost TX-AZ"}',
    '{"action_type":"create_booking_draft","carrier":"Carrier C","lane":"TX-CA","origin":"Austin","destination":"Los Angeles","weight_lb":45000,"estimated_cost_usd":18500,"requires_approval":true,"rationale":"team driver premium"}',
    '{"action_type":"create_booking_draft","carrier":"Carrier C","lane":"TX-CA","origin":"Dallas","destination":"San Diego","weight_lb":30000,"estimated_cost_usd":12400,"requires_approval":true,"rationale":"expedited >10k"}',
    '{"action_type":"none","carrier":null,"lane":null,"origin":null,"destination":null,"weight_lb":null,"estimated_cost_usd":null,"requires_approval":false,"rationale":"recall only"}',
    '{"action_type":"none","carrier":null,"lane":null,"origin":null,"destination":null,"weight_lb":null,"estimated_cost_usd":null,"requires_approval":false,"rationale":"policy lookup"}',
]


def _wire_fast() -> dict[str, Any]:
    from core.router import ChainedIntentRouter, HeuristicIntentRouter, LLMIntentRouter
    from core.kg.extractor import RegexEntityExtractor
    from tests.fakes import FakeChatProvider, FakeKnowledgeGraph, FakeKnowledgeRetriever

    router = ChainedIntentRouter(
        heuristic=HeuristicIntentRouter(),
        llm_router=LLMIntentRouter(
            chat=FakeChatProvider(
                reply='{"intent_label": "fallback", "branches": ["ltm","episodes","procedures","rag","kg"], "rationale": "fake"}'
            )
        ),
    )
    action_chat = FakeChatProvider(replies=list(_ACTION_FAST_REPLIES))
    retriever = FakeKnowledgeRetriever(
        hits=[
            {"doc_type": "carrier_sla", "source": "carrier_agreements/carrier_a_2026.pdf",
             "text": "fake", "score": 0.9, "metadata": {}},
        ]
    )
    extractor = RegexEntityExtractor()
    graph = FakeKnowledgeGraph(
        subgraph={
            "nodes": [], "edges": [
                {"kind": "serves", "from_id": "carrier_a", "to_id": "TX-AZ",
                 "properties": {"hop": 1}},
                {"kind": "serves", "from_id": "carrier_a", "to_id": "TX-TX",
                 "properties": {"hop": 1}},
                {"kind": "serves", "from_id": "carrier_b", "to_id": "TX-NM",
                 "properties": {"hop": 1}},
            ], "facts": [], "sources": [],
        }
    )
    return {
        "router": router,
        "retriever": retriever,
        "extractor": extractor,
        "graph": graph,
        "chat": action_chat,
    }


def _wire_live() -> dict[str, Any]:
    from core.kg import get_entity_extractor, get_knowledge_graph
    from core.providers.registry import get_chat_provider
    from core.rag.mongo import get_knowledge_retriever
    from core.router import get_intent_router

    return {
        "router": get_intent_router(),
        "retriever": get_knowledge_retriever(),
        "extractor": get_entity_extractor(),
        "graph": get_knowledge_graph(),
        "chat": get_chat_provider(),
    }


def _live_graph_invoker(realm_id: str):
    """Return an invoker that drives the compiled graph once and returns its latencies."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from agent.graph import get_graph
    from core.settings import AgentContext, get_settings

    graph = get_graph()
    settings = get_settings()

    def _invoke(user_message: str) -> dict[str, Any]:
        thread_id = f"latency-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        payload: Any = {
            "messages": [HumanMessage(content=user_message)],
            "context": AgentContext.from_settings(settings),
        }
        final_state: dict[str, Any] = {}
        while True:
            for mode, event in graph.stream(payload, config=config, stream_mode=["values", "custom"]):
                if mode == "values":
                    final_state = event
            if not final_state.get("__interrupt__"):
                break
            payload = Command(resume={"approved": True, "approver": "latency-eval"})
        return final_state.get("latency_ms", {}) or {}

    return _invoke


def run_suite(mode: str, realm_id: str, runs: int = 3) -> SuiteResult:
    if mode == "latency":
        invoker = _live_graph_invoker(realm_id)
        metrics: list[MetricResult] = [
            latency_metric.run(
                invoker=invoker,
                dataset=str(DATASETS_DIR / "latency.jsonl"),
                runs=runs,
            ),
        ]
        return SuiteResult(mode=mode, metrics=metrics)

    wiring = _wire_fast() if mode == "fast" else _wire_live()
    metrics = [
        routing_metric.run(
            router=wiring["router"],
            dataset=str(DATASETS_DIR / "intents.jsonl"),
        ),
        retrieval_metric.run(
            retriever=wiring["retriever"],
            dataset=str(DATASETS_DIR / "rag_retrieval.jsonl"),
            realm_id=realm_id,
        ),
        kg_metric.run(
            extractor=wiring["extractor"],
            graph=wiring["graph"],
            dataset=str(DATASETS_DIR / "kg_multi_constraint.jsonl"),
            realm_id=realm_id,
        ),
        action_metric.run(
            chat=wiring["chat"],
            dataset=str(DATASETS_DIR / "action_planning.jsonl"),
        ),
    ]
    return SuiteResult(mode=mode, metrics=metrics)


def diff_against_baseline(
    current: SuiteResult,
    baseline: SuiteResult,
    *,
    score_tolerance: float = 0.01,
    latency_factor: float = 1.5,
) -> dict[str, Any]:
    """Compare `current` to `baseline` and report regressions.

    A score regression fires when `current.score < baseline.score - score_tolerance`.
    For the `latency_p50_p95` metric, p95 of `ttft_ms` and `llm_ms` are also
    compared and a regression fires when `current.p95 > baseline.p95 * latency_factor`.
    New metrics (present in `current` but not `baseline`) are reported as
    informational rows and never trigger a regression.
    """
    base_by_name = {m.name: m for m in baseline.metrics}
    rows: list[dict[str, Any]] = []
    regressions: list[tuple[str, str, float, float]] = []
    for cur in current.metrics:
        prev = base_by_name.get(cur.name)
        if prev is None:
            rows.append({"name": cur.name, "kind": "new",
                         "baseline": None, "current": cur.score, "ok": True})
            continue
        score_ok = cur.score >= prev.score - score_tolerance
        if not score_ok:
            regressions.append((cur.name, "score", prev.score, cur.score))
        rows.append({"name": cur.name, "kind": "score",
                     "baseline": prev.score, "current": cur.score, "ok": score_ok})
        if cur.name == "latency_p50_p95":
            for key in ("ttft_ms", "llm_ms"):
                base_p95 = (prev.extras.get(key) or {}).get("p95")
                cur_p95 = (cur.extras.get(key) or {}).get("p95")
                if not (isinstance(base_p95, (int, float)) and isinstance(cur_p95, (int, float))):
                    continue
                ok = cur_p95 <= base_p95 * latency_factor
                if not ok:
                    regressions.append((f"{cur.name}.{key}.p95", "latency_p95",
                                        float(base_p95), float(cur_p95)))
                rows.append({"name": f"{cur.name}.{key}.p95", "kind": "latency_p95",
                             "baseline": float(base_p95), "current": float(cur_p95), "ok": ok})
    return {"rows": rows, "regressions": regressions,
            "score_tolerance": score_tolerance, "latency_factor": latency_factor}


def render_diff(diff: dict[str, Any]) -> str:
    lines = [
        "",
        "## Baseline diff "
        f"(score_tolerance={diff['score_tolerance']}, latency_factor={diff['latency_factor']})",
        "",
        "| Metric | Kind | Baseline | Current | OK |",
        "|---|---|---|---|---|",
    ]
    for r in diff["rows"]:
        if r["kind"] == "latency_p95":
            base = "—" if r["baseline"] is None else f"{r['baseline']:.1f}"
            cur = f"{r['current']:.1f}"
        else:
            base = "—" if r["baseline"] is None else f"{r['baseline']:.3f}"
            cur = f"{r['current']:.3f}"
        flag = "✅" if r["ok"] else "❌"
        lines.append(f"| `{r['name']}` | {r['kind']} | {base} | {cur} | {flag} |")
    if diff["regressions"]:
        lines.append("")
        lines.append(f"**{len(diff['regressions'])} regression(s):**")
        for name, kind, base_v, cur_v in diff["regressions"]:
            lines.append(f"- `{name}` ({kind}): {base_v} → {cur_v}")
    return "\n".join(lines)


def render_markdown(suite: SuiteResult) -> str:
    lines = [
        f"# Eval suite ({suite.mode} mode)",
        "",
        "| Metric | Dataset | N | Passed | Score | Extras |",
        "|---|---|---|---|---|---|",
    ]
    for m in suite.metrics:
        extras_pairs = []
        for k, v in m.extras.items():
            if k == "confusion":
                continue
            if isinstance(v, dict):
                inner = " ".join(f"{ik}={iv}" for ik, iv in v.items())
                extras_pairs.append(f"{k}=({inner})")
            else:
                extras_pairs.append(f"{k}={v}")
        extras = ", ".join(extras_pairs) or "-"
        lines.append(
            f"| `{m.name}` | `{Path(m.dataset).name}` | {m.n} | {m.passed} | "
            f"{m.score:.3f} | {extras} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suite.")
    parser.add_argument("--mode", choices=("fast", "live", "latency"), default="fast")
    parser.add_argument("--realm-id", default=None,
                        help="Override realm_id (defaults to settings).")
    parser.add_argument("--runs", type=int, default=3,
                        help="Invocations per prompt in --mode latency (default 3).")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the full JSON suite result.")
    parser.add_argument("--baseline", default=None,
                        help="Optional path to write the baseline JSON (live mode usually).")
    parser.add_argument("--against", default=None,
                        help="Optional path to a prior baseline JSON. Exits non-zero on regression.")
    parser.add_argument("--score-tolerance", type=float, default=0.01,
                        help="Allowed drop in metric score before flagging a regression (default 0.01).")
    parser.add_argument("--latency-factor", type=float, default=1.5,
                        help="Allowed multiplier on baseline p95 latency before flagging (default 1.5).")
    args = parser.parse_args(argv)

    if args.realm_id is None:
        try:
            from core.settings import get_settings
            args.realm_id = get_settings().realm_id
        except Exception:
            args.realm_id = "customer-tenant-001"

    suite = run_suite(args.mode, args.realm_id, runs=args.runs)
    payload = suite.model_dump()
    print(render_markdown(suite))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
    if args.baseline:
        Path(args.baseline).write_text(json.dumps(payload, indent=2))
    if args.against:
        baseline_suite = SuiteResult(**json.loads(Path(args.against).read_text()))
        diff = diff_against_baseline(
            suite, baseline_suite,
            score_tolerance=args.score_tolerance,
            latency_factor=args.latency_factor,
        )
        print(render_diff(diff))
        if diff["regressions"]:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
