"""Cross-provider chat fallback wrapper.

Wraps a primary `ChatProvider` with one or more named secondaries.
On a retryable failure (rate limit, 5xx, timeout, connection error)
the wrapper advances to the next provider. Non-retryable errors are
re-raised immediately. After every successful call, `last_fallback`
holds the name of the secondary that handled it (or `None` when the
primary succeeded) so graph nodes can append a
`chat_fallback:<name>` marker to the `degraded` channel.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from pydantic import BaseModel

from core.protocols import ChatProvider

_RETRYABLE_CLASS_NAMES: frozenset[str] = frozenset({
    # openai-python (>=1.x)
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GatewayTimeoutError",
    # httpx
    "TimeoutException",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "ConnectError",
})


def is_retryable_chat_error(exc: BaseException) -> bool:
    """Classify whether a chat-provider exception should trigger fallback.

    Kept dependency-light: matches by class name + HTTP status code so we
    don't have to import openai/httpx at module load time.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if type(exc).__name__ in _RETRYABLE_CLASS_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 408 or status == 429 or status >= 500):
        return True
    return False


class FallbackChatProvider:
    """Implements `core.protocols.ChatProvider` with primary + secondaries.

    Each entry is a `(name, provider)` tuple so the surviving provider's
    name can be reported back via `last_fallback`. The wrapper forwards
    `last_usage` from whichever provider actually answered, keeping the
    token-accounting path in `agent/nodes.py` unchanged.
    """

    def __init__(
        self,
        primary: tuple[str, ChatProvider],
        *secondaries: tuple[str, ChatProvider],
        is_retryable: Callable[[BaseException], bool] | None = None,
    ) -> None:
        if not isinstance(primary, tuple) or len(primary) != 2:
            raise ValueError("primary must be a (name, provider) tuple")
        for entry in secondaries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("each secondary must be a (name, provider) tuple")
        self._providers: list[tuple[str, ChatProvider]] = [primary, *secondaries]
        self._is_retryable = is_retryable or is_retryable_chat_error
        self.last_fallback: str | None = None
        self.last_usage: dict[str, int] | None = None

    @property
    def model_name(self) -> str:
        return self._providers[0][1].model_name

    @property
    def provider_chain(self) -> Sequence[str]:
        return tuple(name for name, _ in self._providers)

    def invoke(self, prompt: str) -> str:
        return self._dispatch("invoke", (prompt,))

    def invoke_typed(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self._dispatch("invoke_typed", (prompt, schema))

    def underlying(self) -> Any:
        """Forward to the primary's raw client (used by `generate_response.stream`).

        Streaming bypasses the fallback chain by design: switching mid-stream
        is not supported. Reflection, plan_action, and memory extraction all
        go through `invoke` / `invoke_typed` and therefore *are* covered.
        """
        prov = self._providers[0][1]
        underlying = getattr(prov, "underlying", None)
        return underlying() if callable(underlying) else None

    def _dispatch(self, method: str, args: tuple[Any, ...]) -> Any:
        errors: list[tuple[str, BaseException]] = []
        for idx, (name, provider) in enumerate(self._providers):
            try:
                result = getattr(provider, method)(*args)
            except BaseException as exc:
                if not self._is_retryable(exc):
                    raise
                errors.append((name, exc))
                continue
            self.last_usage = getattr(provider, "last_usage", None)
            self.last_fallback = name if idx > 0 else None
            return result
        last_name, last_exc = errors[-1]
        raise RuntimeError(
            "all chat providers failed: "
            + ", ".join(f"{n}={type(e).__name__}" for n, e in errors)
        ) from last_exc


__all__ = ["FallbackChatProvider", "is_retryable_chat_error"]
