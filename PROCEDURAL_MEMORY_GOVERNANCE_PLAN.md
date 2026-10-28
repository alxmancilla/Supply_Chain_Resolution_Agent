# Procedural Memory Governance Implementation Plan

## Phase 1 — Domain policy

1. Add a small deterministic policy object under `core/memory/`.
2. Define a typed policy error with safe, stable reasons.
3. Validate proposer/approver roles, rule length, protected phrases, and
   separation of duties without any LLM call.

## Phase 2 — Configuration and context

1. Add role and policy settings with safe demo defaults.
2. Add `actor_role` to `AgentContext` after existing positional fields to
   preserve checkpoint and retry construction compatibility.
3. Document all settings in `.env.example`.

## Phase 3 — Persistence enforcement

1. Pass the policy to `MongoProceduralMemory`.
2. Validate on both `propose()` and `commit()`.
3. Persist proposal/approval audit metadata and calculated expiry.
4. Filter expired rules in `list_active()` while allowing legacy records
   without `expires_at`.

## Phase 4 — Graph and UI wiring

1. Pass proposer role from graph context to staged proposals.
2. Convert policy failures into an observable blocked proposal without an
   `interrupt()` or database write.
3. Pass the configured UI actor role on approval resume.
4. Return lifecycle and audit details in the graph result.

## Phase 5 — Tests and validation

1. Extend in-memory fakes for role metadata and expiry.
2. Cover blocked content, unauthorized roles, self-approval, approval role,
   expiry filtering, legacy compatibility, and successful audit data.
3. Run focused tests, the full test suite, diagnostics, and `git diff --check`.

## Rollout and rollback

The new fields are additive and require no data migration. Deploy with the
default 90-day TTL, then review old no-expiry rules and re-seed or migrate them
with an explicit expiry. Rollback is code-only: existing documents remain
readable because their original fields are retained.