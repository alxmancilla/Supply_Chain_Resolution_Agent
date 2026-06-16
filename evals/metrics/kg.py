"""KG row-match metric.

For each labeled multi-constraint question, the metric runs the entity
extractor + knowledge graph and verifies that every expected
(carrier_id, lane_id, hop) row is present in the returned subgraph edges.
"""
from __future__ import annotations

from core.protocols import EntityExtractor, KnowledgeGraph
from evals.metrics._io import load_jsonl
from evals.schemas import CaseOutcome, MetricResult


def run(
    *,
    extractor: EntityExtractor,
    graph: KnowledgeGraph,
    dataset: str,
    realm_id: str,
    limit: int = 8,
) -> MetricResult:
    cases: list[CaseOutcome] = []
    passed = 0
    row_recall_sum = 0.0

    for row in load_jsonl(dataset):
        expected_rows = row["expected_rows"]
        spec = extractor.extract(row["query"])
        subgraph = graph.query(realm_id, spec, limit=limit)
        actual = [
            {"carrier_id": e.from_id, "lane_id": e.to_id, "hop": int(e.properties.get("hop", 1))}
            for e in subgraph.edges
        ]
        actual_set = {(a["carrier_id"], a["lane_id"], a["hop"]) for a in actual}
        expected_set = {(r["carrier_id"], r["lane_id"], int(r.get("hop", 1))) for r in expected_rows}
        present = expected_set & actual_set
        row_recall = len(present) / len(expected_set) if expected_set else 0.0
        ok = expected_set.issubset(actual_set)
        passed += int(ok)
        row_recall_sum += row_recall
        cases.append(
            CaseOutcome(
                case_id=row["case_id"],
                passed=ok,
                score=row_recall,
                expected=expected_rows,
                actual=actual,
                notes=f"row_recall={row_recall:.2f} extracted_lanes={spec.lanes}",
            )
        )

    n = len(cases)
    score = passed / n if n else 0.0
    mean = row_recall_sum / n if n else 0.0
    return MetricResult(
        name="kg_row_match",
        dataset=str(dataset),
        n=n,
        passed=passed,
        score=score,
        extras={"mean_row_recall": round(mean, 4)},
        cases=cases,
    )


__all__ = ["run"]
