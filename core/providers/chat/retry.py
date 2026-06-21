"""Self-correcting retry for structured-output chat calls.

Wraps `ChatProvider.invoke_typed` with a bounded re-prompt loop. When the
provider returns text that fails JSON extraction or schema validation, the
next attempt re-prompts the model with the previous error message appended
to the original prompt. The chat provider's `last_usage` / `last_fallback`
channels are forwarded unchanged from the surviving call (the fallback
chain runs *inside* each attempt — orthogonal to the retry budget).

After every call, `chat.last_structured_attempts` holds the number of
attempts actually used (1 == no retry) so graph nodes can emit a
`structured_retry:<node>` marker into the `degraded` channel.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from core.protocols import ChatProvider


_RETRY_INSTRUCTION = (
    "Your previous response could not be parsed: {error}\n"
    "Return ONLY a single JSON object that matches the requested schema, "
    "with no prose, comments, or markdown fences."
)


class StructuredOutputRetryError(ValueError):
    """Raised after `max_attempts` of `invoke_typed` failed parse/validation.

    Subclass of `ValueError` so existing `except ValueError` blocks in graph
    nodes continue to degrade gracefully without code changes.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: BaseException | None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def invoke_typed_with_retry(
    chat: ChatProvider,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_attempts: int = 3,
) -> BaseModel:
    """Call `chat.invoke_typed(prompt, schema)` with bounded self-correcting retries.

    Behaviour:
    - First attempt sends `prompt` unchanged.
    - On `ValueError` (the canonical signal from `GroveChatProvider.invoke_typed`
      for both JSON-extraction failures and Pydantic validation failures), the
      next attempt appends a short retry instruction containing the error text.
    - After `max_attempts` failures, raises `StructuredOutputRetryError`
      (a `ValueError` subclass), so existing call sites that do
      `except ValueError` continue to degrade unchanged.
    - On success, sets `chat.last_structured_attempts = N` (1..max_attempts).
    - On exhaustion, sets `chat.last_structured_attempts = max_attempts`.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last_error: BaseException | None = None
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        try:
            result = chat.invoke_typed(current_prompt, schema)
        except ValueError as exc:
            last_error = exc
            current_prompt = (
                f"{prompt}\n\n"
                + _RETRY_INSTRUCTION.format(error=str(exc)[:500])
            )
            continue
        _set_attempts(chat, attempt)
        return result
    _set_attempts(chat, max_attempts)
    raise StructuredOutputRetryError(
        f"structured-output retry exhausted after {max_attempts} attempts: {last_error}",
        attempts=max_attempts,
        last_error=last_error,
    )


def _set_attempts(chat: Any, attempts: int) -> None:
    try:
        setattr(chat, "last_structured_attempts", attempts)
    except AttributeError:
        # Provider uses __slots__ without the attribute — skip silently;
        # the helper still returns the parsed result.
        pass


__all__ = ["invoke_typed_with_retry", "StructuredOutputRetryError"]
