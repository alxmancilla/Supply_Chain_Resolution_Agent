"""Entity extraction for KG retrieval.

Pulls lane codes, carrier names, weight thresholds, and surcharge
constraints out of a user message into an `EntitySpec` the
`KnowledgeGraph` can use as seeds + filters.

`RegexEntityExtractor` is the default (deterministic, ~zero latency).
The LLM-backed extractor is intentionally not implemented yet — the regex
covers the demo's supply-chain entity vocabulary; a future swap to an
LLM extractor only has to satisfy the `EntityExtractor` protocol.
"""
from __future__ import annotations

import re
from functools import lru_cache

from core.schemas import EntitySpec


_LANE_RE = re.compile(r"\b([A-Z]{2})\s*[-/]\s*([A-Z]{2})\b")

_STATE_PAIRS = {
    ("austin", "dallas"): "TX-TX",
    ("dallas", "austin"): "TX-TX",
    ("houston", "san antonio"): "TX-TX",
    ("san antonio", "houston"): "TX-TX",
    ("el paso", "phoenix"): "TX-AZ",
    ("phoenix", "el paso"): "TX-AZ",
}

_CARRIER_RE = re.compile(r"\b[Cc]arrier\s+([A-Z])\b")

_WEIGHT_RE = re.compile(
    r"\b(?:over|above|more than|at least|>=?)\s*([0-9][0-9,]*)\s*(lb|lbs|pounds|kg|kilograms)\b",
    re.IGNORECASE,
)

_SURCHARGE_RE = re.compile(r"\b(no|zero|without)\s+(fuel\s+)?surcharge\b", re.IGNORECASE)


def _extract_lanes(text: str) -> list[str]:
    lanes: list[str] = []
    for o, d in _LANE_RE.findall(text):
        code = f"{o.upper()}-{d.upper()}"
        if code not in lanes:
            lanes.append(code)
    lower = text.lower()
    for (a, b), code in _STATE_PAIRS.items():
        if a in lower and b in lower and code not in lanes:
            lanes.append(code)
    return lanes


def _extract_carriers(text: str) -> list[str]:
    carriers: list[str] = []
    for letter in _CARRIER_RE.findall(text):
        cid = f"carrier_{letter.lower()}"
        if cid not in carriers:
            carriers.append(cid)
    return carriers


def _extract_weight_lb(text: str) -> float | None:
    match = _WEIGHT_RE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    unit = match.group(2).lower()
    if unit.startswith("kg") or unit.startswith("kilogram"):
        value *= 2.20462
    return value


def _extract_constraints(text: str, weight_lb: float | None) -> dict[str, float]:
    constraints: dict[str, float] = {}
    if _SURCHARGE_RE.search(text):
        constraints["surcharge_max"] = 0.0
    if weight_lb is not None:
        constraints["weight_threshold_lb_min"] = weight_lb
    return constraints


class RegexEntityExtractor:
    """Implements `core.protocols.EntityExtractor` via deterministic patterns."""

    def extract(self, user_message: str) -> EntitySpec:
        lanes = _extract_lanes(user_message)
        carriers = _extract_carriers(user_message)
        weight_lb = _extract_weight_lb(user_message)
        constraints = _extract_constraints(user_message, weight_lb)
        return EntitySpec(
            lanes=lanes,
            carriers=carriers,
            weight_lb=weight_lb,
            constraints=constraints,
        )


@lru_cache(maxsize=1)
def get_entity_extractor() -> RegexEntityExtractor:
    """Process-wide default extractor (regex)."""
    return RegexEntityExtractor()


__all__ = ["RegexEntityExtractor", "get_entity_extractor"]
