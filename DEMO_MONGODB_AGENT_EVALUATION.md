# MongoDB Automated Agent Evaluation Demo

## Purpose

Demonstrate that MongoDB is the durable evaluation substrate for an AI agent:
it retains the agent's completed execution state, supplies evaluation inputs
without manual labeling, and stores quality history for trend analysis and
failure investigation.

This is **not** a claim that MongoDB judges model quality. An LLM-as-a-judge
scores the turn; MongoDB makes that process repeatable, scoped, queryable, and
auditable.

## Audience and outcome

- **Audience:** technical buyers, platform teams, and AI-application owners.
- **Takeaway:** one MongoDB deployment can support agent state, grounded
  context, checkpointed execution traces, live-traffic evaluation, and a
  quality dashboard.
- **Duration:** 7–10 minutes, plus optional drill-down discussion.

## Demo narrative

1. **Run real agent work.** Use two representative prompts in the Streamlit
   chat: a grounded shipment recommendation and a policy / KG lookup. LangGraph
   persists checkpoints to MongoDB as each turn completes.
2. **Show the durable trace.** Explain that a terminal checkpoint contains the
   question, final reply, retrieval context, route, citations, latency, usage,
   and degraded markers. No separate transcript-export pipeline is needed.
3. **Automate the evaluation.** Run the live evaluator against a recent window:

   ```bash
   python -m tools.eval_live_traffic --window-hours 24 --limit 20
   ```

   It selects one terminal checkpoint per `thread_id`, reconstructs the turn,
   runs Faithfulness, Answer Relevancy, and Context Relevancy judges, and writes
   an immutable evaluation-run record.
4. **Show aggregate quality.** Open the **Evaluation** tab in the demo app and
   show run count, evaluated-turn count, and the three aggregate scores over a
   selected time window.
5. **Diagnose a weak result.** Select one run / low score and reveal its thread
   reference, timestamp, question, answer preview, context count, judge score,
   and short rationale. The presenter connects it back to the stored agent
   trace, retrieval decision, and source context.
6. **Operationalize it.** State that the evaluator can run on a daily scheduler
   (CI scheduled workflow, Kubernetes CronJob, or equivalent) and that the
   persisted `eval_runs` history can back alerts and a seven-day trend view.

## Architecture shown

```text
Agent turn → LangGraph MongoDB checkpoint → live evaluator → LLM judges
     ↑              │                              │
     └── MongoDB ───┴────────────── eval_runs ─────┴── Evaluation dashboard
```

### Collections and responsibilities

| MongoDB data | Role in the demo |
| --- | --- |
| LangGraph `checkpoints` | Source of truth for completed agent execution state. |
| `knowledge_corpus`, `kg_*`, memories | Grounding data that produced the turn. |
| `eval_runs` | Append-only evaluation history: metric means and per-turn verdicts. |

## Implementation scope

### Evaluation record contract

New runs contain the existing payload plus ownership metadata:

```text
{ run_id, run_at, realm_id, agent_id, window_hours, n_turns,
  scores: { faithfulness, answer_relevancy, context_relevancy }, per_turn }
```

- `realm_id` and `agent_id` are copied from the terminal checkpoint context.
- The evaluator accepts optional `--realm-id` / `--agent-id` filters, defaulting
  to application settings; it never combines tenants in a run.
- A compound `{realm_id, agent_id, run_at}` descending index supports dashboard
  filtering. The existing `run_at` index remains valid for legacy operations.

### Dashboard contract

- A read-only **Evaluation** tab is added to the existing right-hand inspector.
- A 1 / 7 / 30-day window fetches only the selected realm + agent's runs.
- It renders aggregate score cards, a per-run trend table, and an expandable
  low-score drill-down. The UI never runs judges or writes data.
- Empty state tells the presenter exactly which evaluator command to run.

### PII and safety boundaries

- The dashboard shows question / answer previews already persisted in
  `eval_runs`; production deployments must apply their organization’s redaction
  policy before checkpoints or evaluation records are stored.
- Raw credentials, model prompts, and full checkpoint payloads are never
  rendered by the dashboard.
- All reads and writes are tenant-and-agent scoped. Existing legacy eval runs
  without scope metadata are intentionally not shown in the scoped dashboard.

## Delivery plan

1. **Evaluator contract and storage:** derive scope from checkpoint context,
   add filters, carry it into the run payload, and provision the compound index.
2. **Read model:** create pure helpers for safe scoped retrieval, score cards,
   trend rows, and low-score turn selection.
3. **UI:** integrate the Evaluation tab with the read model and an explicit
   operator runbook / empty state.
4. **Verification:** cover scoping, payload shape, index plan, dashboard
   helpers, and full existing unit suite; update the README and Roadmap.

## Acceptance criteria

- A terminal checkpoint becomes a scored, persisted `eval_runs` record through
  one CLI command.
- New evaluation records have `realm_id` and `agent_id`; filtered runs never
  mix owners.
- The UI displays scoped aggregates and a per-turn judge-rationale drill-down.
- No dashboard action creates agent data or invokes an LLM.
- Unit tests run without Atlas or provider credentials.

## Presenter checklist

1. Provision indexes: `python -m db.indexes`.
2. Start Streamlit and produce at least two completed turns.
3. Run the evaluator with the same realm / agent as the demo app.
4. Refresh the Evaluation tab and select a run.
5. Keep `--dry-run --out /tmp/eval-run.json` available for a no-write rehearsal.