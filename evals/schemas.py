"""Result schemas for the eval harness.

Every metric returns a `MetricResult`. The runner aggregates these into
a `SuiteResult` that's written to disk as JSON (the baseline file) and
rendered as a markdown table to stdout.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CaseOutcome(BaseModel):
    """Single labeled example after scoring."""
    model_config = ConfigDict(extra="allow")

    case_id: str
    passed: bool
    score: float = 0.0
    expected: Any = None
    actual: Any = None
    notes: str = ""


class MetricResult(BaseModel):
    """Aggregate result for one metric over its dataset."""
    model_config = ConfigDict(extra="allow")

    name: str
    dataset: str
    n: int
    passed: int
    score: float
    extras: dict[str, Any] = Field(default_factory=dict)
    cases: list[CaseOutcome] = Field(default_factory=list)


class SuiteResult(BaseModel):
    """All metric results from one runner invocation."""
    model_config = ConfigDict(extra="allow")

    mode: str
    metrics: list[MetricResult] = Field(default_factory=list)


__all__ = ["CaseOutcome", "MetricResult", "SuiteResult"]
