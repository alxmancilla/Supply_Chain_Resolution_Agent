# Langfuse Evaluation Integration Specification

**Status:** Proposed — planning only; no implementation is authorized by this document.

## 1. Purpose

Add Langfuse as the observability, evaluation, and human-review experience for
the Supply Chain Resolution Agent while retaining MongoDB Atlas as the durable,
tenant-scoped source of truth for agent state, grounding evidence, approvals,
and evaluation audit records.

## 2. Desired outcome

For each real agent turn, an operator can move between a Langfuse trace and its
MongoDB evidence using a stable identifier. Automated evaluations and human
decisions appear on the trace and remain auditable in Atlas.

```text
Agent turn -> Atlas checkpoint/evidence -> Langfuse trace -> automated scores
          -> human review or approval -> Atlas audit record and Langfuse event
```

## 3. Goals

1. Trace every interactive, CLI-demo, smoke, and evaluation graph invocation
   when Langfuse is explicitly enabled.
2. Correlate one user turn with its Atlas checkpoint, Langfuse trace, and any
   evaluator scores using the existing `correlation_id`.
3. Retain `realm_id`, `agent_id`, `thread_id`, routing, latency, usage, and
   degraded markers as filterable, non-secret trace metadata.
4. Publish the existing faithfulness, answer-relevancy, and context-relevancy
   scores to the matching Langfuse trace on an explicit opt-in.
5. Record high-cost action approval requested/approved/rejected events against
   the relevant trace without bypassing the existing LangGraph interrupt.
6. Remain fully functional, testable, and silent when Langfuse is disabled.

## 4. Non-goals

- Replacing Atlas checkpoints, `eval_runs`, RAG, memory, or booking-draft data.
- Sending credentials, MongoDB connection strings, or unrestricted customer
  content to Langfuse.
- Automatically promoting an evaluator result or reviewer feedback to memory,
  a procedure, or an executable action.
- Changing the current judge prompts, scoring semantics, or approval threshold
  in the first delivery.
- Adding a production scheduler; the existing evaluator remains CLI-invocable.

## 5. Current integration seams

| Concern | Current source | Required extension |
| --- | --- | --- |
| Turn identity | `AgentContext.correlation_id` | Deterministic Langfuse trace ID/join key |
| Conversation | Streamlit `thread_id` | Langfuse `session_id` |
| Graph invocation | `app.py` `_drive_turn` / `_stream_invocation` | Callback and root trace context |
| Retries/resumes | `app.py` approval and retry paths | Reuse the originating trace ID |
| Evaluation source | `tools/eval_live_traffic.py` | Include correlation and publish scores |
| Approval evidence | `execute_action` interrupt/checkpoint | Trace events and categorical result |
| Existing instrumentation | `core/latency.py` OTel node spans | Ensure it shares the Langfuse trace context |

## 6. Architecture and ownership

### MongoDB Atlas remains authoritative

Atlas holds the completed graph checkpoint, user/assistant messages, retrieved
RAG/KG/LTM context, action plans, booking drafts, tenant scope, and the
`eval_runs` audit history. It is the reliable source for replay, scoped batch
evaluation, and compliance/audit queries.

### Langfuse is the experience layer

Langfuse receives trace observations, LangChain/LangGraph callback events,
model usage, selected safe metadata, automated scores, and reviewer feedback.
It provides trace inspection, dashboards, alerts, datasets, experiments, and
annotation queues.

### Correlation contract

Each new user turn creates one `correlation_id`. A deterministic 32-hex
Langfuse trace ID must be derived from it using the Langfuse SDK helper. The
same derived ID is used by initial execution, approval resume, retry, and
post-hoc evaluation. The evaluator must be able to derive it from checkpointed
context; persisting it redundantly is optional, not required for correctness.

## 7. Trace and metadata contract

| Field | Source | Langfuse placement | Sensitivity |
| --- | --- | --- | --- |
| Trace ID | Derived from `correlation_id` | Trace context | Internal identifier |
| Session ID | `thread_id` | Session | Internal identifier |
| User ID | `user_id` | User | Pseudonymize when needed |
| Realm and agent | `AgentContext` | Metadata/tags | Tenant metadata |
| Intent, branches, degraded markers | Terminal graph state | Metadata/events | Operational |
| Latency and usage | Existing state | Observations/metadata | Operational |
| Prompt/reply/context | Graph and callbacks | Configurable capture | Potentially sensitive |

Trace metadata must include the application release and deployment environment
when available. Metadata is not an authorization boundary; cross-tenant access
must use Langfuse project and access-control configuration.

## 8. Automated evaluation contract

`tools/eval_live_traffic` remains responsible for selecting only terminal,
tenant-scoped checkpoints and running the existing judges. With an explicit
`--publish-langfuse` flag, it must publish one numeric trace-level score per
judge plus the judge rationale as the score comment.

Score names are `faithfulness`, `answer_relevancy`, and `context_relevancy`.
Each publish uses a stable idempotency key composed from the evaluation run,
trace, and score name. Publishing failures must not discard the Atlas
`eval_runs` record or fail the rest of the batch; they are reported per turn.

## 9. Human-in-the-loop contract

The existing high-cost booking interruption remains the approval authority.
Langfuse records, but does not decide, these lifecycle events:

1. `approval_requested` with policy reason and redacted proposal summary.
2. `approval_decision` with approved/rejected outcome and pseudonymous
   approver identity.
3. `action_execution_result` with draft outcome or withheld execution.

The decision is also exported as a categorical trace score. Low-quality traces
may be placed in a Langfuse annotation queue, but reviewer corrections require
separate approved changes to procedures, prompts, data, or test cases.

## 10. Security, privacy, and resilience

- Enable only with `LANGFUSE_ENABLED=1` and deployment-managed Langfuse keys.
- Support the customer's selected cloud region or self-hosted base URL.
- Define redaction/truncation for prompts, replies, documents, approver data,
  and booking payloads before production enablement.
- Never put secrets in trace metadata, logs, slide decks, test fixtures, or CLI
  arguments.
- Langfuse failures are non-blocking: graph execution, MongoDB writes, and
  approval handling continue with a structured local diagnostic.
- Unit tests run without Atlas, model credentials, or Langfuse credentials.

## 11. Acceptance criteria

1. A completed demo turn appears as one navigable Langfuse trace with the
   matching Atlas correlation ID, session, scope, node timing, and model calls.
2. Approval resume and retry appear on the same trace as the originating turn.
3. The evaluation CLI can publish all three judge scores idempotently to the
   trace while preserving the existing `eval_runs` record.
4. A disabled or unavailable Langfuse integration never blocks a user turn,
   CLI demo, test suite, evaluator run, or human approval.
5. Tests verify correlation, callback configuration, score payloads,
   idempotency, tenant preservation, redaction behavior, and failure isolation.
6. The demo can show trace -> score -> human decision -> Atlas evidence in
   under ten minutes.