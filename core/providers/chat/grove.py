"""Grove gateway implementation of `core.protocols.ChatProvider`.

The Grove gateway speaks OpenAI's wire protocol but uses an `api-key`
header instead of Bearer auth. `ChatOpenAI` from langchain-openai is
still the HTTP client; the provider class is the boundary the rest of
the codebase sees.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

from core.usage import extract_usage_metadata

GROVE_BASE_URL = "https://grove-gateway-prod.azure-api.net/grove-foundry-prod/openai/v1"
GROVE_MODEL = "gpt-5.5"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class GroveChatProvider:
    """Implements `core.protocols.ChatProvider` via Grove (GPT-5.5)."""

    model_name: str = GROVE_MODEL

    def __init__(
        self,
        *,
        model_name: str | None = None,
        temperature: float = 0.2,
        base_url: str = GROVE_BASE_URL,
    ) -> None:
        if model_name is not None:
            self.model_name = model_name
        self._temperature = temperature
        self._base_url = base_url
        self._client: Any = None
        self.last_usage: dict[str, int] | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.environ.get("GROVE_API_KEY")
            if not api_key:
                raise RuntimeError("GROVE_API_KEY is required when CHAT_PROVIDER=grove.")
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.model_name,
                base_url=self._base_url,
                api_key="placeholder",
                default_headers={"api-key": api_key},
                temperature=self._temperature,
                stream_usage=True,
            )
        return self._client

    def invoke(self, prompt: str) -> str:
        reply = self._get_client().invoke([HumanMessage(content=prompt)])
        self.last_usage = extract_usage_metadata(reply)
        content = reply.content
        return content if isinstance(content, str) else str(content)

    def invoke_typed(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Plain-JSON-extraction structured output.

        Avoids `with_structured_output()` because the Grove pin lags
        OpenAI's tool-use schema. The prompt is expected to ask for a
        single JSON object on one line.
        """
        raw = self.invoke(prompt)
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError(f"chat provider produced no JSON object: {raw!r}")
        data = json.loads(match.group(0))
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"structured-output failed schema validation: {exc}") from exc

    def underlying(self) -> Any:
        """Escape hatch for callers that still need the raw langchain client."""
        return self._get_client()


__all__ = ["GroveChatProvider", "GROVE_MODEL", "GROVE_BASE_URL"]
