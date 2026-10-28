# Procedural Memory Governance Specification

## Purpose

Protect durable `agent_procedures` rules, which are injected into future
prompts, from unsafe content, unauthorized activation, self-approval, and
unbounded lifetime.

## Scope

Applies to `propose_procedure` only. Booking approval, RAG retrieval, and
legacy active rules are unchanged.

## Identity and authorization model

`AgentContext` carries an actor role derived from `USER_ROLE` (default:
`operator`). The policy accepts configurable comma-separated role sets:

- `PROCEDURE_PROPOSER_ROLES` — default `operator,admin`.
- `PROCEDURE_APPROVER_ROLES` — default `operator,admin` for the demo.

The approval resume payload carries the configured UI actor role, not a
free-text role. A proposal must have a non-empty approver identity different
from `proposed_by`, and that role must be an approver role.

This is a demo authorization boundary only. The Streamlit UI does not
authenticate people; production must populate actor identity and role from an
SSO/authorization provider and must not trust browser-supplied identities.

## Proposal policy

Before persistence, a proposed rule must:

1. originate from an allowed proposer role;
2. use an allowed existing `ProcedureCategory` value;
3. be non-empty and no longer than `PROCEDURE_MAX_RULE_CHARS` (default 500);
4. not contain a protected control phrase, case-insensitively.

Protected phrases block attempts to alter prompt hierarchy, bypass approval or
safety controls, disable grounding, or request credentials/secrets. The list
is centralized, deterministic, and intentionally conservative.

Blocked proposals are never written to `procedure_proposals`. The graph
returns a `procedure_proposal` with `status="blocked"` and a safe reason.

## Approval and lifecycle policy

Every new approved rule receives an `expires_at` timestamp based on
`PROCEDURE_RULE_TTL_DAYS` (default 90). `list_active()` returns only active,
unexpired rules. Existing records without `expires_at` remain readable for
backward compatibility; teams must review or migrate them separately.

Approval validates role and separation of duties again at commit time. Reject
records are retained; the active rule set is not changed.

## Audit contract

Staged proposals retain `proposed_by`, `proposed_by_role`, `correlation_id`,
category, rationale, timestamps, status, approver, and approver role.
Promoted rules retain source proposal, proposer metadata, approver metadata,
correlation id, approval time, expiry, and timestamps.

## Acceptance criteria

- Unauthorized or protected proposals produce no staging write.
- A proposer cannot approve their own proposal.
- Unauthorized approvers cannot activate rules.
- An approved rule receives expiry and is excluded after expiry.
- Legacy active rules without expiry remain available.
- Existing tenant and agent scope remains mandatory on every read/write.