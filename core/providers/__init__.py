"""Pluggable embedding + chat-completion providers behind stable protocols.

The `registry` module is the single entry point — callers depend on
`get_embedding_provider()` / `get_chat_provider()` and let `Settings`
decide which concrete backend is wired in.
"""
from __future__ import annotations

from .registry import get_chat_provider, get_embedding_provider

__all__ = ["get_chat_provider", "get_embedding_provider"]
