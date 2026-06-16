"""Intent router for the per-turn retrieval fan-out.

Classifies the user's last message into a `RoutingDecision` that names
which of the five retrieval branches (`ltm`, `episodes`, `procedures`,
`rag`, `kg`) should actually run for this turn. The graph's conditional
edges use that list to skip irrelevant retrievers.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from core.protocols import ChatProvider
from core.schemas import ALL_BRANCHES, RoutingDecision


INTENT_ROUTER_PROMPT = """You are an INTENT ROUTER for a supply chain agent.
Read the user's last message and decide which retrieval branches to activate.

Available branches:
- ltm         : semantic long-term memory (user preferences, durable facts)
- episodes    : past structured shipment interactions (vector-retrieved)
- procedures  : tenant-curated operating rules (always active for actionable
                requests; skip only for pure recall questions)
- rag         : knowledge corpus (route guides, carrier SLAs, policies)
- kg          : structured knowledge graph ($graphLookup over carriers, lanes,
                SLAs — use for multi-constraint or multi-hop questions like
                "carriers that serve TX-AZ over 18,000 lbs with no surcharge")

Intent labels (pick exactly one):
- recommend_shipment : a request to plan, recommend, or book a shipment
                       --> branches: ltm, episodes, procedures, rag, kg
- recall_preference  : asking about prior preferences or past decisions
                       --> branches: ltm, episodes
- recall_episode     : asking specifically about past shipments
                       --> branches: episodes
- lookup_policy      : asking for a policy, SLA, or route-guide fact
                       --> branches: rag, procedures
- structured_lookup  : multi-constraint or multi-hop carrier/lane/SLA question
                       --> branches: kg, rag
- list_rules         : asking what the operating rules / procedures are
                       --> branches: procedures
- propose_procedure  : user wants to ADD a new standing rule for this tenant
                       (e.g. "going forward, always...", "from now on, never...")
                       --> branches: procedures
- fallback           : unclear or off-topic
                       --> branches: ltm, episodes, procedures, rag, kg

Output a SINGLE JSON object on one line, no markdown fences, no prose:
{{"intent_label": "<label>", "branches": ["<branch>", ...], "rationale": "<10 words max>"}}

USER MESSAGE:
{user_message}

JSON:"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of the LLM response."""
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"router LLM produced no JSON object: {raw!r}")
    return json.loads(match.group(0))


def _sanitize_branches(branches: Any) -> list[str]:
    """Drop unknowns; preserve order; fall back to ALL_BRANCHES if empty."""
    if not isinstance(branches, list):
        return list(ALL_BRANCHES)
    seen: list[str] = []
    for b in branches:
        if isinstance(b, str) and b in ALL_BRANCHES and b not in seen:
            seen.append(b)
    return seen or list(ALL_BRANCHES)


_HEURISTIC_RULES: tuple[tuple[re.Pattern[str], str, list[str]], ...] = (
    (
        re.compile(r"^\s*(going forward|from now on|please always|always remember to|make it a rule( that)?|new rule:|add a rule:|set a rule:)\b", re.IGNORECASE),
        "propose_procedure",
        ["procedures"],
    ),
    (
        re.compile(r"\b(which|what|list).{0,40}\b(carriers?|lanes?)\b.{0,80}\b(over|under|above|below|no(\s+\w+){0,2}\s+surcharge|with(out)? fuel)\b", re.IGNORECASE),
        "structured_lookup",
        ["kg", "rag"],
    ),
    (
        re.compile(r"\b(ship|shipment|recommend|book|plan|move|haul|need to (ship|send))\b.*\b(lb|lbs|kg|pounds|kilograms)\b", re.IGNORECASE),
        "recommend_shipment",
        ["ltm", "episodes", "procedures", "rag", "kg"],
    ),
    (
        re.compile(r"\b(what carrier|which carrier|did i (prefer|pick|choose|use)|my preference|last time)\b", re.IGNORECASE),
        "recall_preference",
        ["ltm", "episodes"],
    ),
    (
        re.compile(r"\b(past shipment|previous shipment|prior shipment|recent shipment|history of)\b", re.IGNORECASE),
        "recall_episode",
        ["episodes"],
    ),
    (
        re.compile(r"\b(operating (rule|rules|procedures)|(what|list|show).{0,20}\b(rules|procedures)|active rules)\b", re.IGNORECASE),
        "list_rules",
        ["procedures"],
    ),
    (
        re.compile(r"\b(policy|policies|sla|surcharge|route guide|carrier agreement|tariff)\b", re.IGNORECASE),
        "lookup_policy",
        ["rag", "procedures"],
    ),
)


class HeuristicIntentRouter:
    """Deterministic regex router. Returns None when no rule matches."""

    def route_optional(self, user_message: str) -> RoutingDecision | None:
        for pattern, label, branches in _HEURISTIC_RULES:
            if pattern.search(user_message):
                return RoutingDecision(
                    intent_label=label,
                    branches=list(branches),
                    rationale=f"heuristic match: /{pattern.pattern[:40]}.../",
                )
        return None

    def route(self, user_message: str) -> RoutingDecision:
        decision = self.route_optional(user_message)
        if decision is None:
            return RoutingDecision(
                intent_label="fallback",
                branches=list(ALL_BRANCHES),
                rationale="no heuristic match",
            )
        return decision


class LLMIntentRouter:
    """Implements `core.protocols.IntentRouter` via a `ChatProvider`."""

    def __init__(self, *, chat: ChatProvider) -> None:
        self._chat = chat

    def route(self, user_message: str) -> RoutingDecision:
        raw = self._chat.invoke(INTENT_ROUTER_PROMPT.format(user_message=user_message))
        data = _extract_json(raw)
        data["branches"] = _sanitize_branches(data.get("branches"))
        return RoutingDecision.model_validate(data)


class ChainedIntentRouter:
    """Tries the heuristic router first; falls back to the LLM when it abstains."""

    def __init__(self, *, heuristic: HeuristicIntentRouter, llm_router: LLMIntentRouter) -> None:
        self._heuristic = heuristic
        self._llm_router = llm_router

    def route(self, user_message: str) -> RoutingDecision:
        decision = self._heuristic.route_optional(user_message)
        if decision is not None:
            return decision
        return self._llm_router.route(user_message)


@lru_cache(maxsize=1)
def get_intent_router() -> ChainedIntentRouter:
    """Process-wide default router: heuristic fast-path with LLM fallback."""
    from core.providers.registry import get_chat_provider

    return ChainedIntentRouter(
        heuristic=HeuristicIntentRouter(),
        llm_router=LLMIntentRouter(chat=get_chat_provider()),
    )


__all__ = [
    "HeuristicIntentRouter",
    "LLMIntentRouter",
    "ChainedIntentRouter",
    "get_intent_router",
    "INTENT_ROUTER_PROMPT",
]
