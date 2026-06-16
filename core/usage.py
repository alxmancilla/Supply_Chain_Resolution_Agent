"""Token usage + cost accounting shared by every chat call.

LangChain attaches `usage_metadata = {"input_tokens", "output_tokens", ...}`
to the `AIMessage` (or final streamed chunk) when the upstream provider
returns usage. We extract just the input/output counts and convert them
to USD using rates from `Settings`. Per-turn totals flow through the
graph's `usage` channel via `merge_usage`.
"""
from __future__ import annotations

from typing import Any

from core.settings import Settings


def extract_usage_metadata(message: Any) -> dict[str, int] | None:
    """Return `{input_tokens, output_tokens}` from a langchain message, or None."""
    if message is None:
        return None
    meta = getattr(message, "usage_metadata", None)
    if not isinstance(meta, dict):
        return None
    inp = meta.get("input_tokens")
    out = meta.get("output_tokens")
    if not isinstance(inp, int) and not isinstance(out, int):
        return None
    return {
        "input_tokens": int(inp) if isinstance(inp, int) else 0,
        "output_tokens": int(out) if isinstance(out, int) else 0,
    }


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    settings: Settings,
) -> float:
    """Apply per-1k-token rates from `Settings` to a token count."""
    return (
        (input_tokens / 1000.0) * settings.chat_input_price_per_1k_usd
        + (output_tokens / 1000.0) * settings.chat_output_price_per_1k_usd
    )


def usage_payload(
    input_tokens: int,
    output_tokens: int,
    settings: Settings,
) -> dict[str, float]:
    """Build a `usage` channel delta from raw token counts."""
    return {
        "tokens_in": float(input_tokens),
        "tokens_out": float(output_tokens),
        "cost_usd": compute_cost_usd(input_tokens, output_tokens, settings),
        "calls": 1.0,
    }


def merge_usage(
    left: dict[str, float] | None,
    right: dict[str, float] | None,
) -> dict[str, float]:
    """Reducer for the `usage` channel — sums all numeric keys."""
    merged: dict[str, float] = {}
    for src in (left, right):
        if not src:
            continue
        for key, value in src.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0.0) + float(value)
    return merged


__all__ = [
    "extract_usage_metadata",
    "compute_cost_usd",
    "usage_payload",
    "merge_usage",
]
