"""Heuristic RAG query planner.

Extracts structured filters (lanes, carriers, doc_types) from a free-text
user query so the downstream retriever can narrow `$vectorSearch` /
`$search` to the relevant corpus slice. Pure-Python regex — no LLM call.
"""
from __future__ import annotations

import re
from typing import Iterable

from core.schemas import RagQueryFilters

LANE_CODES: tuple[str, ...] = ("TX-AZ", "TX-TX", "TX-CA", "AZ-CA", "TX-NM")
CARRIER_LETTERS: tuple[str, ...] = ("A", "B", "C")

_LANE_PATTERNS = [(code, re.compile(rf"\b{code}\b", re.IGNORECASE)) for code in LANE_CODES]
_CARRIER_PATTERNS = [
    (f"Carrier {letter}", re.compile(rf"\bcarrier[\s-]?{letter}\b", re.IGNORECASE))
    for letter in CARRIER_LETTERS
]

_DOC_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "carrier_sla": ("sla", "agreement", "liability", "deductible"),
    "route_guide": ("route guide", "lane guide", "primary carrier", "transit"),
    "exception_playbook": (
        "playbook", "exception", "late delivery", "damaged goods",
        "no-show", "no show", "weight discrepancy",
    ),
    "shipping_policy": (
        "policy", "approval threshold", "vendor list", "hours of service",
        "fuel surcharge", "claims", "insurance", "appointment",
    ),
    "carrier_scorecard": ("scorecard", "on-time", "performance review", "quarterly"),
}


def _find_lanes(text: str) -> list[str]:
    found: list[str] = []
    for code, pattern in _LANE_PATTERNS:
        if pattern.search(text) and code not in found:
            found.append(code)
    return found


def _find_carriers(text: str) -> list[str]:
    found: list[str] = []
    for label, pattern in _CARRIER_PATTERNS:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def _find_doc_types(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords) and doc_type not in found:
            found.append(doc_type)
    return found


def _rationale(lanes: Iterable[str], carriers: Iterable[str], doc_types: Iterable[str]) -> str:
    parts: list[str] = []
    if lanes := list(lanes):
        parts.append(f"lanes={lanes}")
    if carriers := list(carriers):
        parts.append(f"carriers={carriers}")
    if doc_types := list(doc_types):
        parts.append(f"doc_types={doc_types}")
    return "heuristic: " + ", ".join(parts) if parts else "heuristic: no filters extracted"


def plan_query(text: str) -> RagQueryFilters:
    """Extract `RagQueryFilters` from the user's natural-language query."""
    if not text:
        return RagQueryFilters(rationale="heuristic: empty query")
    lanes = _find_lanes(text)
    carriers = _find_carriers(text)
    doc_types = _find_doc_types(text)
    return RagQueryFilters(
        lanes=lanes,
        carriers=carriers,
        doc_types=doc_types,
        rationale=_rationale(lanes, carriers, doc_types),
    )


__all__ = ["plan_query", "LANE_CODES", "CARRIER_LETTERS"]
