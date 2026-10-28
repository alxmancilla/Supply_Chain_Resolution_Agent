"""Optional, fail-open Langfuse tracing and evaluation-score publishing.

MongoDB Atlas remains the durable record for checkpoints, evidence, approvals,
and evaluation runs. This module only mirrors safe operational metadata and
explicitly requested evaluation scores to Langfuse.
"""
from __future__ import annotations

import hashlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

from core.settings import Settings

try:
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler
except ImportError:  # pragma: no cover - supports disabled deployments without the SDK
    Langfuse = None  # type: ignore[assignment,misc]
    propagate_attributes = None  # type: ignore[assignment]
    CallbackHandler = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class LangfuseTurn:
    """The trace identity and root observation for one graph invocation."""

    trace_id: str
    observation: Any


def trace_id_for(correlation_id: str) -> str:
    """Return the stable Langfuse trace ID for a checkpointed correlation ID."""
    if Langfuse is not None:
        return Langfuse.create_trace_id(seed=correlation_id)
    return hashlib.sha256(correlation_id.encode("utf-8")).hexdigest()[:32]


def score_id_for(run_id: str, trace_id: str, name: str) -> str:
    """Return a stable idempotency key for one evaluation-run trace score."""
    seed = f"{run_id}:{trace_id}:{name}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


@lru_cache(maxsize=1)
def get_langfuse_client(settings: Settings) -> Any | None:
    """Return the process-wide client only when explicitly enabled and configured."""
    if not settings.langfuse_enabled or Langfuse is None:
        return None
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.langfuse_environment,
            release=settings.langfuse_release or None,
        )
    except Exception:
        return None


def _content_or_none(value: str, settings: Settings) -> str | None:
    if not settings.langfuse_capture_content:
        return None
    return value[: settings.langfuse_capture_max_chars]


@contextmanager
def trace_agent_turn(
    *,
    settings: Settings,
    context: Any | None,
    thread_id: str,
    user_input: str,
    invocation: str,
) -> Iterator[LangfuseTurn | None]:
    """Trace a graph invocation without allowing SDK failures to block it."""
    if context is None:
        yield None
        return
    client = get_langfuse_client(settings)
    if client is None or propagate_attributes is None:
        yield None
        return

    correlation_id = getattr(context, "correlation_id", None)
    if not isinstance(correlation_id, str) or not correlation_id:
        yield None
        return
    trace_id = trace_id_for(correlation_id)
    metadata = {
        "correlation_id": correlation_id,
        "realm_id": getattr(context, "realm_id", None),
        "agent_id": getattr(context, "agent_id", None),
        "invocation": invocation,
    }
    attrs_cm = None
    observation_cm = None
    observation = None
    try:
        attrs_cm = propagate_attributes(
            user_id=getattr(context, "user_id", None),
            session_id=thread_id,
            metadata=metadata,
            trace_name="supply-chain-agent-turn",
            environment=settings.langfuse_environment,
            version=settings.langfuse_release or None,
            tags=["supply-chain-resolution-agent"],
        )
        attrs_cm.__enter__()
        observation_cm = client.start_as_current_observation(
            as_type="agent",
            name="agent.turn",
            trace_context={"trace_id": trace_id},
            input=_content_or_none(user_input, settings),
        )
        observation = observation_cm.__enter__()
    except Exception:
        if attrs_cm is not None:
            try:
                attrs_cm.__exit__(None, None, None)
            except Exception:
                pass
        yield None
        return

    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield LangfuseTurn(trace_id=trace_id, observation=observation)
    except BaseException:
        exc_info = sys.exc_info()
        try:
            observation.update(level="ERROR", status_message=str(exc_info[1])[:500])
        except Exception:
            pass
        raise
    finally:
        try:
            observation_cm.__exit__(*exc_info)
        except Exception:
            pass
        try:
            attrs_cm.__exit__(*exc_info)
        except Exception:
            pass


def complete_agent_turn(
    trace: LangfuseTurn | None,
    result: dict[str, Any],
    *,
    interrupted: bool,
    settings: Settings,
) -> None:
    """Attach a safe terminal summary to the active root observation."""
    if trace is None:
        return
    routing = result.get("routing") or {}
    metadata = {
        "interrupted": interrupted,
        "intent": routing.get("intent_label"),
        "branches": routing.get("branches"),
        "degraded": result.get("degraded") or [],
        "latency_ms": result.get("latency_ms") or {},
        "usage": result.get("usage") or {},
    }
    output = None
    if settings.langfuse_capture_content:
        for message in reversed(result.get("messages") or []):
            content = getattr(message, "content", None)
            if isinstance(content, str):
                output = _content_or_none(content, settings)
                break
    try:
        trace.observation.update(metadata=metadata, output=output)
    except Exception:
        pass


def with_langfuse_callbacks(
    config: dict[str, Any],
    *,
    trace: LangfuseTurn | None,
    settings: Settings,
    run_name: str,
) -> dict[str, Any]:
    """Return graph config enriched with a nested Langfuse callback when safe.

    The LangChain callback captures model prompts and completions.  Keep it
    behind the existing content-capture opt-in so metadata-only tracing never
    exports user or retrieved content.  A new handler per invocation avoids
    sharing callback state across Streamlit sessions and nests its runs under
    the active ``agent.turn`` observation opened by :func:`trace_agent_turn`.
    """
    enriched = dict(config)
    if trace is None or not settings.langfuse_capture_content or CallbackHandler is None:
        return enriched
    try:
        callbacks = list(enriched.get("callbacks") or [])
        callbacks.append(CallbackHandler())
        enriched["callbacks"] = callbacks
        enriched["run_name"] = run_name
    except Exception:
        # Callback creation must retain the adapter's fail-open contract.
        return dict(config)
    return enriched


def publish_evaluation_scores(
    per_turn: list[dict[str, Any]],
    *,
    run_id: str,
    settings: Settings,
    client: Any | None = None,
) -> None:
    """Best-effort publish of existing judge scores; records per-score outcomes."""
    client = client if client is not None else get_langfuse_client(settings)
    for turn in per_turn:
        correlation_id = turn.get("correlation_id")
        if not isinstance(correlation_id, str) or not correlation_id:
            turn["langfuse"] = {"status": "skipped", "reason": "missing_correlation_id"}
            continue
        if client is None:
            turn["langfuse"] = {"status": "skipped", "reason": "disabled_or_unconfigured"}
            continue
        trace_id = trace_id_for(correlation_id)
        scores: dict[str, dict[str, str]] = {}
        for name, score in (turn.get("scores") or {}).items():
            try:
                client.create_score(
                    name=name,
                    value=float(score["score"]),
                    trace_id=trace_id,
                    score_id=score_id_for(run_id, trace_id, name),
                    data_type="NUMERIC",
                    comment=str(score.get("reason") or "")[:1000],
                )
                scores[name] = {"status": "published"}
            except Exception as exc:
                scores[name] = {"status": "failed", "error": type(exc).__name__}
        turn["langfuse"] = {"trace_id": trace_id, "scores": scores}
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


__all__ = [
    "LangfuseTurn",
    "complete_agent_turn",
    "get_langfuse_client",
    "publish_evaluation_scores",
    "score_id_for",
    "trace_agent_turn",
    "trace_id_for",
    "with_langfuse_callbacks",
]