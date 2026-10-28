"""Deterministic governance policy for durable procedural-memory rules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


ALLOWED_CATEGORIES = frozenset({"escalation", "formatting", "units", "policy", "general"})
PROTECTED_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "bypass approval",
    "disable approval",
    "override approval",
    "bypass safety",
    "disable safety",
    "disable grounding",
    "api key",
    "password",
    "secret",
)


class ProceduralPolicyError(ValueError):
    """A safe, stable reason for rejecting a procedure state transition."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ProcedureGovernancePolicy:
    """Role, content, and lifecycle constraints for a procedural rule."""

    proposer_roles: tuple[str, ...] = ("operator", "admin")
    approver_roles: tuple[str, ...] = ("operator", "admin")
    max_rule_chars: int = 500
    rule_ttl_days: int = 90

    def __post_init__(self) -> None:
        if self.max_rule_chars <= 0:
            raise ValueError("max_rule_chars must be positive")
        if self.rule_ttl_days <= 0:
            raise ValueError("rule_ttl_days must be positive")

    def validate_proposal(self, *, rule: str, category: str, proposer_role: str | None) -> None:
        if (proposer_role or "").strip().lower() not in self.proposer_roles:
            raise ProceduralPolicyError("proposer_role_not_authorized")
        if category not in ALLOWED_CATEGORIES:
            raise ProceduralPolicyError("procedure_category_not_allowed")
        text = (rule or "").strip()
        if not text:
            raise ProceduralPolicyError("procedure_rule_empty")
        if len(text) > self.max_rule_chars:
            raise ProceduralPolicyError("procedure_rule_too_long")
        folded = " ".join(text.casefold().split())
        if any(phrase in folded for phrase in PROTECTED_PHRASES):
            raise ProceduralPolicyError("procedure_rule_contains_protected_control")

    def validate_approval(
        self,
        *,
        proposed_by: str | None,
        approver: str | None,
        approver_role: str | None,
    ) -> None:
        identity = (approver or "").strip()
        if not identity:
            raise ProceduralPolicyError("procedure_approver_required")
        if (approver_role or "").strip().lower() not in self.approver_roles:
            raise ProceduralPolicyError("approver_role_not_authorized")
        if proposed_by and identity.casefold() == proposed_by.strip().casefold():
            raise ProceduralPolicyError("procedure_self_approval_forbidden")

    def expiry_at(self, *, now: datetime | None = None) -> str:
        issued_at = now or datetime.now(timezone.utc)
        return (issued_at + timedelta(days=self.rule_ttl_days)).isoformat(timespec="seconds")


__all__ = ["ProcedureGovernancePolicy", "ProceduralPolicyError"]