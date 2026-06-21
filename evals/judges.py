"""LLM-as-judge scorers for live-traffic evaluation.

Three judges run per turn:

- `FaithfulnessJudge`     — does every claim in the answer trace back to
                            the retrieved context?
- `AnswerRelevancyJudge`  — does the answer actually address the
                            question?
- `ContextRelevancyJudge` — how much of the retrieved context is
                            relevant to the question?

Each judge returns a `JudgeScore` (0.0-1.0 + short rationale). Scores are
extracted through `chat.invoke_typed_with_retry` so a malformed reply is
re-prompted before falling back to a sentinel 0-score.
"""
from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field, field_validator

from core.protocols import ChatProvider
from core.providers.chat.retry import (
    StructuredOutputRetryError,
    invoke_typed_with_retry,
)


class JudgeScore(BaseModel):
    """Single judge verdict over one turn."""
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _clip(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


_FAITHFULNESS_PROMPT = """\
You are a strict factuality judge. Score whether every factual claim in
the AGENT REPLY is supported by the RETRIEVED CONTEXT.

Rules:
- 1.0 = every claim is directly supported.
- 0.5 = some claims supported, some unsupported but plausible.
- 0.0 = the reply invents facts not in the context.

Reply with strict JSON: {{"score": <float 0..1>, "reason": "<one sentence>"}}.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

AGENT REPLY:
{answer}
"""

_ANSWER_RELEVANCY_PROMPT = """\
You are a relevance judge. Score whether the AGENT REPLY actually
addresses the USER QUESTION (ignore correctness — only relevance).

Rules:
- 1.0 = directly answers the question.
- 0.5 = partially addresses it or answers a related question.
- 0.0 = off-topic.

Reply with strict JSON: {{"score": <float 0..1>, "reason": "<one sentence>"}}.

USER QUESTION:
{question}

AGENT REPLY:
{answer}
"""

_CONTEXT_RELEVANCY_PROMPT = """\
You are a retrieval-quality judge. Score how much of the RETRIEVED
CONTEXT is relevant to the USER QUESTION.

Rules:
- 1.0 = every chunk is on-topic.
- 0.5 = roughly half is relevant.
- 0.0 = nothing in the context relates to the question.

Reply with strict JSON: {{"score": <float 0..1>, "reason": "<one sentence>"}}.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""


def _join_context(chunks: Sequence[str]) -> str:
    parts = [c.strip() for c in chunks if c and c.strip()]
    return "\n---\n".join(parts) if parts else "(no context retrieved)"


def _score(chat: ChatProvider, prompt: str, *, max_attempts: int) -> JudgeScore:
    try:
        result = invoke_typed_with_retry(chat, prompt, JudgeScore, max_attempts=max_attempts)
    except StructuredOutputRetryError as exc:
        return JudgeScore(score=0.0, reason=f"judge_parse_failed: {exc}")
    assert isinstance(result, JudgeScore)
    return result


def judge_faithfulness(
    chat: ChatProvider, *, question: str, answer: str, context: Sequence[str],
    max_attempts: int = 3,
) -> JudgeScore:
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question.strip(), answer=answer.strip(),
        context=_join_context(context),
    )
    return _score(chat, prompt, max_attempts=max_attempts)


def judge_answer_relevancy(
    chat: ChatProvider, *, question: str, answer: str,
    max_attempts: int = 3,
) -> JudgeScore:
    prompt = _ANSWER_RELEVANCY_PROMPT.format(
        question=question.strip(), answer=answer.strip(),
    )
    return _score(chat, prompt, max_attempts=max_attempts)


def judge_context_relevancy(
    chat: ChatProvider, *, question: str, context: Sequence[str],
    max_attempts: int = 3,
) -> JudgeScore:
    prompt = _CONTEXT_RELEVANCY_PROMPT.format(
        question=question.strip(), context=_join_context(context),
    )
    return _score(chat, prompt, max_attempts=max_attempts)


JUDGE_NAMES: tuple[str, ...] = ("faithfulness", "answer_relevancy", "context_relevancy")


def run_all(
    chat: ChatProvider, *, question: str, answer: str, context: Sequence[str],
    max_attempts: int = 3,
) -> dict[str, JudgeScore]:
    """Run all three judges over one turn and return a name -> JudgeScore dict."""
    return {
        "faithfulness": judge_faithfulness(
            chat, question=question, answer=answer, context=context,
            max_attempts=max_attempts,
        ),
        "answer_relevancy": judge_answer_relevancy(
            chat, question=question, answer=answer, max_attempts=max_attempts,
        ),
        "context_relevancy": judge_context_relevancy(
            chat, question=question, context=context, max_attempts=max_attempts,
        ),
    }


__all__ = [
    "JudgeScore", "JUDGE_NAMES",
    "judge_faithfulness", "judge_answer_relevancy", "judge_context_relevancy",
    "run_all",
]
