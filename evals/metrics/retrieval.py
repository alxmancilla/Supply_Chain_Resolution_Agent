"""RAG recall@k metric.

A case passes if every expected source appears at least once in the
top-k hits returned by the `KnowledgeRetriever`. We also report the
mean per-case recall (fraction of expected sources present) for
finer-grained diffs against the baseline.
"""
from __future__ import annotations

from core.protocols import KnowledgeRetriever
from evals.metrics._io import load_jsonl
from evals.schemas import CaseOutcome, MetricResult


def run(*, retriever: KnowledgeRetriever, dataset: str, realm_id: str) -> MetricResult:
    cases: list[CaseOutcome] = []
    passed = 0
    recall_sum = 0.0

    for row in load_jsonl(dataset):
        expected = set(row["expected_sources"])
        k = int(row.get("k", 5))
        hits = retriever.query(realm_id=realm_id, text=row["query"], k=k)
        actual_sources = [h.source for h in hits]
        actual_set = set(actual_sources)
        hit_count = len(expected & actual_set)
        recall = hit_count / len(expected) if expected else 0.0
        ok = expected.issubset(actual_set)
        passed += int(ok)
        recall_sum += recall
        cases.append(
            CaseOutcome(
                case_id=row["case_id"],
                passed=ok,
                score=recall,
                expected=sorted(expected),
                actual=actual_sources,
                notes=f"recall={recall:.2f} k={k}",
            )
        )

    n = len(cases)
    score = passed / n if n else 0.0
    mean_recall = recall_sum / n if n else 0.0
    return MetricResult(
        name="rag_recall_at_k",
        dataset=str(dataset),
        n=n,
        passed=passed,
        score=score,
        extras={"mean_recall": round(mean_recall, 4)},
        cases=cases,
    )


__all__ = ["run"]
