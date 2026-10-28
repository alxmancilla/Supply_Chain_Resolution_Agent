"""MongoDB-backed procedural LTM (tenant + agent scoped operating rules).

Reads are unconditional (`list_active`); writes go through a staged
propose -> commit | reject workflow. The agent never writes directly to
`agent_procedures` — only `commit()` (called after human approval in
`execute_action`) promotes a row from `procedure_proposals` into the
active rule set.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional, Sequence

from core.schemas import ProceduralRule

from .procedural_policy import ProcedureGovernancePolicy


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _stable_rule_id(rule_text: str) -> str:
    """Deterministic ID so re-proposing the same rule overwrites the same proposal."""
    digest = hashlib.sha1(_normalize(rule_text).encode("utf-8")).hexdigest()[:12]
    return f"proc_{digest}"


class MongoProceduralMemory:
    """Implements `core.protocols.ProceduralMemory` over raw pymongo collections."""

    def __init__(
        self,
        *,
        collection: Any,
        proposals: Optional[Any] = None,
        policy: ProcedureGovernancePolicy | None = None,
    ) -> None:
        self._collection = collection
        self._proposals = proposals
        self._policy = policy or ProcedureGovernancePolicy()

    def list_active(self, realm_id: str, agent_id: str) -> Sequence[ProceduralRule]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cursor = self._collection.find(
            {
                "realm_id": realm_id,
                "agent_id": agent_id,
                "active": True,
                "$or": [{"expires_at": {"$exists": False}}, {"expires_at": None}, {"expires_at": {"$gt": now}}],
            },
            {"_id": 0, "rule_id": 1, "rule": 1, "category": 1},
        ).sort("rule_id", 1)
        return [
            ProceduralRule(
                rule_id=doc["rule_id"],
                rule=doc["rule"],
                category=doc.get("category", "general"),
            )
            for doc in cursor
        ]

    def propose(
        self,
        realm_id: str,
        agent_id: str,
        *,
        rule: str,
        category: str = "general",
        rationale: str = "",
        proposed_by: Optional[str] = None,
        proposed_by_role: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upsert a candidate rule into the `procedure_proposals` staging collection.

        Idempotent on the normalized rule text — re-proposing the same rule
        overwrites the same `proposal_id`. Returns the staged document.
        """
        if self._proposals is None:
            raise RuntimeError("propose() requires a `proposals` collection")
        self._policy.validate_proposal(
            rule=rule,
            category=category,
            proposer_role=proposed_by_role,
        )
        rule_id = _stable_rule_id(rule)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        doc: dict[str, Any] = {
            "proposal_id": rule_id,
            "rule_id": rule_id,
            "realm_id": realm_id,
            "agent_id": agent_id,
            "rule": rule.strip(),
            "category": category,
            "rationale": rationale,
            "status": "pending_approval",
            "proposed_by": proposed_by,
            "proposed_by_role": proposed_by_role,
            "correlation_id": correlation_id,
            "updated_at": now,
        }
        self._proposals.update_one(
            {"realm_id": realm_id, "agent_id": agent_id, "proposal_id": rule_id},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return doc

    def commit(
        self,
        realm_id: str,
        agent_id: str,
        proposal_id: str,
        *,
        approver: Optional[str] = None,
        approver_role: Optional[str] = None,
    ) -> dict[str, Any]:
        """Promote an approved proposal into the active rule set. Idempotent."""
        if self._proposals is None:
            raise RuntimeError("commit() requires a `proposals` collection")
        proposal = self._proposals.find_one(
            {"realm_id": realm_id, "agent_id": agent_id, "proposal_id": proposal_id}
        )
        if proposal is None:
            raise LookupError(f"no proposal found: {proposal_id}")
        self._policy.validate_approval(
            proposed_by=proposal.get("proposed_by"),
            approver=approver,
            approver_role=approver_role,
        )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        expires_at = proposal.get("expires_at") or self._policy.expiry_at()
        rule_doc = {
            "rule_id": proposal["rule_id"],
            "rule": proposal["rule"],
            "category": proposal.get("category", "general"),
            "realm_id": realm_id,
            "agent_id": agent_id,
            "active": True,
            "source_proposal_id": proposal_id,
            "proposed_by": proposal.get("proposed_by"),
            "proposed_by_role": proposal.get("proposed_by_role"),
            "approver": approver,
            "approver_role": approver_role,
            "correlation_id": proposal.get("correlation_id"),
            "approved_at": now,
            "expires_at": expires_at,
            "updated_at": now,
        }
        self._collection.update_one(
            {"realm_id": realm_id, "agent_id": agent_id, "rule_id": proposal["rule_id"]},
            {"$set": rule_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        self._proposals.update_one(
            {"realm_id": realm_id, "agent_id": agent_id, "proposal_id": proposal_id},
            {"$set": {
                "status": "approved",
                "approver": approver,
                "approver_role": approver_role,
                "approved_at": now,
                "expires_at": expires_at,
            }},
        )
        return rule_doc

    def reject(
        self,
        realm_id: str,
        agent_id: str,
        proposal_id: str,
        *,
        approver: Optional[str] = None,
        approver_role: Optional[str] = None,
    ) -> None:
        """Mark a staged proposal as rejected. Leaves the active rule set untouched."""
        if self._proposals is None:
            raise RuntimeError("reject() requires a `proposals` collection")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._proposals.update_one(
            {"realm_id": realm_id, "agent_id": agent_id, "proposal_id": proposal_id},
            {"$set": {
                "status": "rejected",
                "approver": approver,
                "approver_role": approver_role,
                "rejected_at": now,
            }},
        )


@lru_cache(maxsize=1)
def get_procedural_memory() -> MongoProceduralMemory:
    """Process-wide default procedural memory wired to the shared Atlas collections."""
    from agent.memory import get_procedure_proposals_collection, get_procedures_collection
    from core.settings import get_settings

    settings = get_settings()
    policy = ProcedureGovernancePolicy(
        proposer_roles=settings.procedure_proposer_roles,
        approver_roles=settings.procedure_approver_roles,
        max_rule_chars=settings.procedure_max_rule_chars,
        rule_ttl_days=settings.procedure_rule_ttl_days,
    )

    return MongoProceduralMemory(
        collection=get_procedures_collection(),
        proposals=get_procedure_proposals_collection(),
        policy=policy,
    )


__all__ = ["MongoProceduralMemory", "get_procedural_memory", "_stable_rule_id"]
