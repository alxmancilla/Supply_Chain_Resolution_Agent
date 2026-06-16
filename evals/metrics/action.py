"""Action-planning accuracy metric.

For each labeled (user_message, agent_message) pair, the metric calls
the chat provider with the production `ACTION_PLANNING_PROMPT`, parses
the resulting `BookingProposal`, and verifies that both `action_type`
and `requires_approval` match the expected values. The high-cost cases
are the regression check for the human-in-the-loop interrupt path.
"""
from __future__ import annotations

from agent.prompts import ACTION_PLANNING_PROMPT
from core.protocols import ChatProvider
from core.schemas import BookingProposal
from evals.metrics._io import load_jsonl
from evals.schemas import CaseOutcome, MetricResult


def run(*, chat: ChatProvider, dataset: str) -> MetricResult:
    cases: list[CaseOutcome] = []
    passed = 0
    correct_approval = 0

    for row in load_jsonl(dataset):
        expected_action = row["expected_action_type"]
        expected_approval = bool(row["expected_requires_approval"])
        prompt = ACTION_PLANNING_PROMPT.format(
            user_message=row["user_message"],
            agent_message=row["agent_message"],
        )
        try:
            proposal = chat.invoke_typed(prompt, BookingProposal)
            assert isinstance(proposal, BookingProposal)
            actual_action = proposal.action_type
            actual_approval = proposal.requires_approval
            notes = ""
        except Exception as exc:
            actual_action = "error"
            actual_approval = False
            notes = f"extraction failed: {exc}"

        action_ok = actual_action == expected_action
        approval_ok = actual_approval == expected_approval
        ok = action_ok and approval_ok
        passed += int(ok)
        correct_approval += int(approval_ok)
        cases.append(
            CaseOutcome(
                case_id=row["case_id"],
                passed=ok,
                score=1.0 if ok else 0.0,
                expected={"action_type": expected_action, "requires_approval": expected_approval},
                actual={"action_type": actual_action, "requires_approval": actual_approval},
                notes=notes or f"action_ok={action_ok} approval_ok={approval_ok}",
            )
        )

    n = len(cases)
    score = passed / n if n else 0.0
    approval_score = correct_approval / n if n else 0.0
    return MetricResult(
        name="action_planning_accuracy",
        dataset=str(dataset),
        n=n,
        passed=passed,
        score=score,
        extras={"approval_accuracy": round(approval_score, 4)},
        cases=cases,
    )


__all__ = ["run"]
