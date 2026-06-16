"""Latency metric: p50/p95 of `llm_ttft_ms` and `llm_ms`.

Each labeled prompt is invoked `runs` times. `invoker(prompt)` must
return a mapping with at least `llm_ttft_ms` and `llm_ms` (the values
recorded by `generate_response` on `state['latency_ms']`). The metric
collects per-case percentiles and aggregates them across the suite.

A case `passes` when all `runs` invocations completed without raising;
the metric is intended as a tail-latency tripwire, not a correctness
check, so the score is the fraction of cases with `runs` clean
samples.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Callable, Mapping

from evals.metrics._io import load_jsonl
from evals.schemas import CaseOutcome, MetricResult

Invoker = Callable[[str], Mapping[str, Any]]


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "p50": round(median(samples), 1),
        "p95": round(_percentile(samples, 0.95), 1),
        "min": round(min(samples), 1),
        "max": round(max(samples), 1),
    }


def run(*, invoker: Invoker, dataset: str, runs: int = 3) -> MetricResult:
    if runs < 1:
        raise ValueError("runs must be >= 1")

    cases: list[CaseOutcome] = []
    all_ttft: list[float] = []
    all_llm: list[float] = []
    passed = 0

    for row in load_jsonl(dataset):
        ttft_samples: list[float] = []
        llm_samples: list[float] = []
        errors: list[str] = []
        for _ in range(runs):
            try:
                latency = invoker(row["user_message"])
                ttft = float(latency.get("llm_ttft_ms", 0.0) or 0.0)
                llm = float(latency.get("llm_ms", 0.0) or 0.0)
            except Exception as exc:
                errors.append(repr(exc))
                continue
            if ttft > 0:
                ttft_samples.append(ttft)
            if llm > 0:
                llm_samples.append(llm)

        all_ttft.extend(ttft_samples)
        all_llm.extend(llm_samples)
        ok = not errors and len(llm_samples) == runs
        passed += int(ok)
        cases.append(
            CaseOutcome(
                case_id=row["case_id"],
                passed=ok,
                score=1.0 if ok else 0.0,
                expected={"runs": runs},
                actual={
                    "runs_collected": len(llm_samples),
                    "ttft_ms": _summary(ttft_samples),
                    "llm_ms": _summary(llm_samples),
                },
                notes="; ".join(errors) or f"{len(llm_samples)}/{runs} clean samples",
            )
        )

    n = len(cases)
    score = passed / n if n else 0.0
    return MetricResult(
        name="latency_p50_p95",
        dataset=str(dataset),
        n=n,
        passed=passed,
        score=score,
        extras={
            "runs_per_case": runs,
            "ttft_ms": _summary(all_ttft),
            "llm_ms": _summary(all_llm),
        },
        cases=cases,
    )


__all__ = ["run", "Invoker"]
