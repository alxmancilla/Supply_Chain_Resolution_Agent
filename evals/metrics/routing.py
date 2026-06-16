"""Intent-routing accuracy + confusion matrix."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.protocols import IntentRouter
from evals.metrics._io import load_jsonl
from evals.schemas import CaseOutcome, MetricResult


def run(*, router: IntentRouter, dataset: str) -> MetricResult:
    cases: list[CaseOutcome] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    passed = 0

    for row in load_jsonl(dataset):
        expected = row["expected_intent"]
        decision = router.route(row["user_message"])
        actual = decision.intent_label
        ok = actual == expected
        passed += int(ok)
        confusion[expected][actual] += 1
        cases.append(
            CaseOutcome(
                case_id=row["case_id"],
                passed=ok,
                score=1.0 if ok else 0.0,
                expected=expected,
                actual=actual,
                notes=decision.rationale,
            )
        )

    n = len(cases)
    score = passed / n if n else 0.0
    return MetricResult(
        name="intent_accuracy",
        dataset=str(dataset),
        n=n,
        passed=passed,
        score=score,
        extras={"confusion": _serialize_confusion(confusion)},
        cases=cases,
    )


def _serialize_confusion(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {expected: dict(actual_map) for expected, actual_map in confusion.items()}


__all__ = ["run"]
