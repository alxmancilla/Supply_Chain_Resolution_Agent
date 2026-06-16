"""Per-node error isolation for the parallel retrieval fan-out.

Wraps a retriever so a backend failure (Atlas hiccup, Voyage outage,
malformed cursor) degrades that branch only — empty hits + a marker
appended to `state['degraded']` — instead of poisoning the whole turn.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable


NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def safe_retrieve(name: str, **default_fields: Any) -> Callable[[NodeFn], NodeFn]:
    """Catch any exception from a retriever and return a degraded result.

    `default_fields` is the safe fallback shape the downstream prompt
    can render without surprise — typically the hits list set to `[]`
    and the context string set to a "(retrieval degraded)" notice.
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001 — isolation is the whole point
                marker = f"{name}: {type(exc).__name__}: {exc}"
                return {**default_fields, "degraded": [marker]}

        return wrapper

    return decorator


__all__ = ["safe_retrieve"]
