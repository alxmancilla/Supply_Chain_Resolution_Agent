# Langfuse Evaluation Integration Plan

**Status:** Proposed. This plan intentionally contains no implementation.

## Guiding delivery decision

Use the Langfuse Python SDK and its LangChain/LangGraph callback integration as
the primary tracing path. It provides model-level observations in addition to
the project's existing OpenTelemetry node spans. Do not independently configure
an unrelated OTLP exporter until trace-context propagation has been verified;
otherwise one turn can produce duplicate traces.

## Preconditions and decisions

1. Obtain explicit approval to add the external `langfuse` Python dependency.
2. Confirm Cloud region versus self-hosted Langfuse, customer project, retention
policy, access model, and permitted prompt/context capture.
3. Define which identifiers require pseudonymization and the redaction policy.
4. Create score configurations in the customer Langfuse project for the three
current numeric judges and the categorical approval decision.
5. Agree that MongoDB `eval_runs` stays as the internal/audit record while
Langfuse is the observability and evaluator workflow surface.

## Phase 1 — Optional configuration and integration boundary

**Objective:** Add a no-op-by-default Langfuse boundary without changing agent
behavior.

- Add the approved SDK through the package manager and update the lock state if
  the project introduces one.
- Extend `Settings` with enabled flag, base URL, environment, and release
  metadata. Read Langfuse credentials lazily from environment only.
- Add a focused `core/langfuse.py` adapter that creates clients/callbacks only
  when enabled, derives a deterministic trace ID, and safely degrades on error.
- Initialize the integration before application modules acquire OTel tracers so
  existing node spans can share context.
- Document env names in `.env.example` and README without supplying values.

**Exit gate:** Starting the app with Langfuse disabled has no new network calls,
no changed user-visible behavior, and no required credentials.

## Phase 2 — Trace the agent execution lifecycle

**Objective:** Produce one correlated trace per user turn.

- Build a shared graph-invocation configuration helper rather than adding
  callback logic separately to Streamlit, `tools.demo`, smoke, and evaluators.
- At a new turn, create a root Langfuse observation using the deterministic
  trace ID derived from `AgentContext.correlation_id`.
- Propagate `thread_id`, pseudonymous user, realm, agent, environment, release,
  intent, and safe operational metadata.
- Pass the Langfuse callback to `graph.stream` and equivalent graph invocations.
- Reuse the originating trace ID for interrupt resume and retry execution.
- Add an optional Streamlit link or displayed trace ID after a finalized turn.

**Exit gate:** One standard request has a single trace containing graph and
model observations; its correlation ID maps back to the terminal checkpoint.

## Phase 3 — Publish live-evaluation scores

**Objective:** Attach current Atlas-backed judge results to production traces.

- Extend checkpoint extraction to retain the correlation ID required for trace
  derivation while preserving current tenant-scope filtering.
- Add `--publish-langfuse` to `tools.eval_live_traffic`; no publishing occurs
  by default or with `--dry-run`.
- Publish the three existing numeric scores and comments with stable idempotency
  identifiers.
- Preserve the full `eval_runs` payload and include Langfuse trace/score
  references where useful for audit and drill-down.
- Capture per-turn publish failures without stopping scoring or Atlas writes.

**Exit gate:** Repeating the same evaluation job updates/retains the intended
Langfuse score identities and does not create unintended duplicates.

## Phase 4 — Human approvals and review workflow

**Objective:** Make consequential decisions visible and reviewable.

- Record `approval_requested`, `approval_decision`, and action-result events
  on the stable trace.
- Export approved/rejected as a categorical score and retain Atlas as the
  decision authority.
- Add a low-score link or trace reference from the Streamlit Evaluation tab.
- Document how customer reviewers use a Langfuse annotation queue for weak
  traces, including the rule that feedback is not auto-promoted to agent memory.

**Exit gate:** A high-cost booking displays a trace that shows the request,
human decision, and final execution outcome alongside the Atlas draft.

## Phase 5 — Dataset experiments and operational rollout

**Objective:** Extend from production monitoring to repeatable improvement.

- Mirror selected existing JSONL cases as a Langfuse Dataset; retain repository
  JSONL fixtures as the canonical CI-friendly test inputs.
- Define experiment evaluators for routing, retrieval, KG constraints, action
  extraction, approval gating, and answer quality.
- Establish a staged rollout: local development -> non-sensitive staging ->
  sampled production -> agreed production coverage.
- Configure alert thresholds only after enough baseline traces exist.
- Update the demo slides and presenter script with the trace/evaluation/HIL
  walkthrough after the implementation has passed validation.

## Test and validation matrix

| Area | Required validation |
| --- | --- |
| Disabled mode | No import-time credential requirement or network activity |
| Trace identity | Same correlation ID always derives the same valid trace ID |
| Graph paths | New turn, stream, retry, and approval resume retain correlation |
| Scope | Realm/agent metadata mirrors checkpoint scope; no cross-tenant mixing |
| Evaluation export | Three scores/comments have correct trace IDs and idempotency keys |
| Failure isolation | Callback/export errors do not block graph, Atlas, or approvals |
| Privacy | Redactor removes configured sensitive fields before export |
| Existing behavior | Focused unit tests and full `pytest tests/ -v` remain green |
| Live smoke | Opt-in non-production trace and score visible in Langfuse |

Use a fake Langfuse adapter in unit tests. Live validation must use a dedicated
non-production project and synthetic shipping prompts, never customer data.

## Proposed demo walkthrough after rollout

1. Run a grounded shipment recommendation and open its Langfuse trace.
2. Show the matching Atlas checkpoint with the evidence the agent actually saw.
3. Run the evaluation CLI with explicit Langfuse publishing enabled.
4. Refresh the trace to show faithfulness, answer relevancy, and context
   relevancy scores with judge rationale.
5. Trigger a high-cost booking, show the approval pause, then approve/reject.
6. Show the approval decision on the trace and the durable Atlas booking draft.
7. Filter low-scoring traces in Langfuse and explain the annotation-to-remediation
   workflow.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Duplicate/incomplete traces | One SDK callback path; explicit trace-context tests |
| Sensitive data export | Approved redaction policy, sampling, regional hosting |
| Evaluation outages affect agent | Best-effort asynchronous export; Atlas remains authoritative |
| Score duplication | Deterministic trace IDs and idempotent score IDs |
| Reviewer feedback changes behavior unsafely | Human-approved promotion process only |
| Vendor lock-in concerns | Atlas retains checkpoints/evidence/eval records; adapter is isolated |

## Files expected to change during a later implementation

- `requirements.txt`, `.env.example`, `README.md`
- `core/settings.py`, `core/observability.py`, and new `core/langfuse.py`
- `app.py`, `tools/demo.py`, `tools/smoke_turn.py`, `tools/eval_live_traffic.py`
- Targeted tests for the adapter, Streamlit invocation config, live evaluator,
  approval lifecycle, and disabled-mode behavior
- `DEMO_SLIDES.md` and presenter documentation after validated functionality

## Explicit stop point

Do not start Phase 1 until the customer confirms Langfuse hosting/region,
privacy policy, score ownership, and approval to add the Langfuse dependency.