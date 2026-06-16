"""OpenTelemetry tracer for graph nodes (no-op when OTel is not installed).

`tracer` resolves to the global OpenTelemetry tracer when the API package is
available. With no `TracerProvider` configured (the default), OTel itself
returns a `NoOpTracer`, so spans created here cost essentially nothing until
an exporter is registered. Set `OTEL_ENABLED=1` plus the usual
`OTEL_EXPORTER_*` env vars to start emitting; everything else is automatic.

`@timed` in `core/latency` opens one span per graph node and tags it with the
correlation id + tenant/user/agent triple from `AgentContext`.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

try:  # opentelemetry-api is an optional dependency.
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only when dep is missing
    _otel_trace = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False


SERVICE_NAME = "supply-chain-resolution-agent"


def _get_tracer() -> Any:
    if not _OTEL_AVAILABLE:
        return None
    return _otel_trace.get_tracer(SERVICE_NAME)


_tracer = _get_tracer()


@contextmanager
def node_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Open a span for a graph node. Yields the span (or `None` if OTel is absent).

    Cheap no-op when OTel isn't installed or no exporter is configured; safe
    to call unconditionally from every node wrapper.
    """
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                span.set_attribute(key, value)
        yield span


def configure_sdk_from_env() -> None:
    """Opt-in: install a basic console SDK when `OTEL_ENABLED=1` and no provider exists.

    Real deployments should wire their own exporter (OTLP, Jaeger, etc.); this
    helper exists so the demo can show traces locally without extra setup.
    """
    if not _OTEL_AVAILABLE:
        return
    if os.environ.get("OTEL_ENABLED") != "1":
        return
    provider = _otel_trace.get_tracer_provider()
    if provider.__class__.__name__ != "ProxyTracerProvider":
        return  # caller already configured one
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:  # pragma: no cover
        return
    new_provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    new_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    _otel_trace.set_tracer_provider(new_provider)
    global _tracer
    _tracer = _get_tracer()


__all__ = ["node_span", "configure_sdk_from_env", "SERVICE_NAME"]
