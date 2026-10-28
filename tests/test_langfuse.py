"""Unit tests for the optional, fail-open Langfuse adapter."""
from __future__ import annotations

from core import langfuse
from core.langfuse import (
    LangfuseTurn,
    publish_evaluation_scores,
    score_id_for,
    trace_id_for,
    with_langfuse_callbacks,
)
from core.settings import Settings


class FakeLangfuseClient:
    def __init__(self, *, fail_name: str | None = None) -> None:
        self.fail_name = fail_name
        self.calls: list[dict] = []
        self.flushed = False

    def create_score(self, **kwargs) -> None:
        if kwargs["name"] == self.fail_name:
            raise RuntimeError("unavailable")
        self.calls.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def _settings() -> Settings:
    return Settings(mongodb_uri="mongodb://test", langfuse_enabled=True)


def _turn() -> dict:
    return {
        "correlation_id": "a" * 32,
        "scores": {
            "faithfulness": {"score": 0.9, "reason": "supported"},
            "answer_relevancy": {"score": 0.8, "reason": "relevant"},
        },
    }


def test_trace_and_score_ids_are_stable_and_valid_length():
    trace_id = trace_id_for("correlation-1")
    assert trace_id == trace_id_for("correlation-1")
    assert len(trace_id) == 32
    assert score_id_for("run", trace_id, "faithfulness") == score_id_for("run", trace_id, "faithfulness")


def test_langfuse_client_is_initialized_once(monkeypatch):
    class FakeClient:
        created = 0

        def __init__(self, **_kwargs) -> None:
            type(self).created += 1

    monkeypatch.setattr(langfuse, "Langfuse", FakeClient)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")
    langfuse.get_langfuse_client.cache_clear()

    first = langfuse.get_langfuse_client(_settings())
    second = langfuse.get_langfuse_client(_settings())

    assert first is second
    assert FakeClient.created == 1
    langfuse.get_langfuse_client.cache_clear()


def test_publish_scores_uses_trace_id_and_stable_score_ids():
    client = FakeLangfuseClient()
    turns = [_turn()]
    publish_evaluation_scores(turns, run_id="run-1", settings=_settings(), client=client)

    assert len(client.calls) == 2
    assert all(call["trace_id"] == trace_id_for("a" * 32) for call in client.calls)
    assert all(len(call["score_id"]) == 32 for call in client.calls)
    assert turns[0]["langfuse"]["scores"]["faithfulness"]["status"] == "published"
    assert client.flushed is True


def test_publish_scores_isolates_a_single_score_failure():
    client = FakeLangfuseClient(fail_name="answer_relevancy")
    turns = [_turn()]
    publish_evaluation_scores(turns, run_id="run-1", settings=_settings(), client=client)

    assert turns[0]["langfuse"]["scores"]["faithfulness"]["status"] == "published"
    assert turns[0]["langfuse"]["scores"]["answer_relevancy"] == {
        "status": "failed", "error": "RuntimeError",
    }


def test_publish_scores_skips_turn_without_checkpointed_correlation_id():
    turn = {"scores": {"faithfulness": {"score": 1.0, "reason": "ok"}}}
    publish_evaluation_scores([turn], run_id="run-1", settings=_settings(), client=FakeLangfuseClient())
    assert turn["langfuse"] == {"status": "skipped", "reason": "missing_correlation_id"}


def test_langgraph_callback_requires_content_capture(monkeypatch):
    class FakeCallbackHandler:
        pass

    monkeypatch.setattr(langfuse, "CallbackHandler", FakeCallbackHandler)
    config = {"configurable": {"thread_id": "thread-1"}}
    trace = LangfuseTurn(trace_id="trace-1", observation=object())

    result = with_langfuse_callbacks(
        config, trace=trace, settings=_settings(), run_name="initial-agent-turn"
    )

    assert result == config
    assert result is not config


def test_langgraph_callback_is_nested_per_invocation(monkeypatch):
    class FakeCallbackHandler:
        pass

    monkeypatch.setattr(langfuse, "CallbackHandler", FakeCallbackHandler)
    settings = Settings(mongodb_uri="mongodb://test", langfuse_enabled=True, langfuse_capture_content=True)
    config = {"configurable": {"thread_id": "thread-1"}, "callbacks": ["existing"]}
    trace = LangfuseTurn(trace_id="trace-1", observation=object())

    result = with_langfuse_callbacks(
        config, trace=trace, settings=settings, run_name="initial-agent-turn"
    )

    assert config["callbacks"] == ["existing"]
    assert result["callbacks"][0] == "existing"
    assert isinstance(result["callbacks"][1], FakeCallbackHandler)
    assert result["run_name"] == "initial-agent-turn"