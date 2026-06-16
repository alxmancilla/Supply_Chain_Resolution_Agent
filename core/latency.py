"""Cross-cutting latency + tracing decorator for graph nodes.

Records the wall-clock time of a node into `state["latency_ms"][key]` and
opens an OpenTelemetry span tagged with the correlation id + tenant/user/agent
triple. The latency channel uses a dict-merge reducer in `agent/nodes.py` so
parallel branches each writing their own key compose correctly. The span is
a no-op unless OTel is configured (see `core.observability`).
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from core.observability import node_span


NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def _context_attrs(state: dict[str, Any], key: str) -> dict[str, Any]:
    ctx = state.get("context")
    routing = state.get("routing") or {}
    attrs: dict[str, Any] = {"agent.node": key}
    if ctx is not None:
        attrs["agent.realm_id"] = getattr(ctx, "realm_id", None)
        attrs["agent.user_id"] = getattr(ctx, "user_id", None)
        attrs["agent.agent_id"] = getattr(ctx, "agent_id", None)
        attrs["agent.correlation_id"] = getattr(ctx, "correlation_id", None)
    intent = routing.get("intent_label")
    if intent:
        attrs["agent.intent"] = intent
    return attrs


def timed(key: str) -> Callable[[NodeFn], NodeFn]:
    """Decorate a graph node so its wall-clock time and span are recorded under `key`."""

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            attrs = _context_attrs(state, key)
            with node_span(key, attrs) as span:
                started = time.perf_counter()
                result = fn(state)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if span is not None:
                    span.set_attribute("agent.latency_ms", elapsed_ms)
                    degraded = result.get("degraded") if isinstance(result, dict) else None
                    if degraded:
                        span.set_attribute("agent.degraded", True)
            merged = dict(result.get("latency_ms") or {})
            merged[key] = elapsed_ms
            out = dict(result)
            out["latency_ms"] = merged
            return out

        return wrapper

    return decorator


__all__ = ["timed"]
