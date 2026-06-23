---
name: langgraph-mongodb-agent
description: Opinionated blueprint for building production-style stateful agents on LangGraph + MongoDB Atlas. Use whenever the user asks to "build an agent", "create an agent", "design a new agent", "scaffold an agent", or discusses agent architecture, memory layers, RAG over Atlas, knowledge graph + agent, procedural rules, human-in-the-loop approval, multi-tenant agents, or tool/action planning. Defaults: LangGraph 1.x StateGraph, MongoDB Atlas (vector + b-tree), Voyage embeddings, Streamlit UI, parallel retrieval fan-out, per-node error isolation, self-correcting structured output, cross-provider chat fallback, opt-in Writer/Reviewer split, per-sentence citation binder, context-discipline prompt assembler (skipped/empty branches dropped), failure recovery via checkpoint time-travel, governed procedural memory via interrupt(), citation enforcement, tenant scoping via realm_id, in-memory fakes for tests. Do NOT invent a different stack unless the user explicitly overrides one of these defaults.
license: MIT
---

# LangGraph + MongoDB Atlas Agent Blueprint

This is the **default architecture and tech stack** for any new agent built in this workspace. Deviate only when the user explicitly says so.

## When to apply

- Building any conversational, task, or workflow agent that needs memory, retrieval, and action-taking.
- The agent must persist across turns, be multi-tenant, and have human-in-the-loop checkpoints.
- The user does not specify a stack — these are the defaults.

## Non-negotiable architecture rules

1. **Single MongoDB Atlas cluster** holds everything: short-term checkpoints, long-term memory (3 kinds), knowledge graph, RAG corpus, action drafts, agent registry. Do not split storage backends.
2. **Six layers, top-down only.** Each layer talks only to the one directly below:
   `Entry points → Orchestration (agent/) → Domain (core/protocols, schemas, settings) → Capabilities (core/router, memory, rag, kg, resilience) → Providers (core/providers/) → Storage + external APIs`.
3. **`agent/` nodes are thin.** They orchestrate; capability logic lives one layer down behind a protocol in `core/protocols.py`.
4. **Vendor SDK imports live only in `core/providers/`.** Feature code never imports `voyageai`, `openai`, etc. directly.
5. **Tenant scoping is mandatory.** Every persisted row carries `realm_id`. User-state collections also carry `user_id`; agent-config collections carry `agent_id`. `thread_id` scopes only `checkpoints`. A `correlation_id` ties one turn's spans, drafts, and resumes.

## Default tech stack

| Layer | Default | Notes |
|---|---|---|
| Orchestration | `langgraph==1.2.5` | `StateGraph`, `interrupt()`, `MongoDBSaver` checkpointer, custom-channel streaming via `get_stream_writer`. |
| Storage / Search | MongoDB Atlas, one cluster | Vector Search for embeddings, b-tree for KG joins, `$graphLookup` for traversals. |
| Embeddings | `voyageai==0.4.0`, model `voyage-4` (1024 dim) | Call `voyageai.Client(...).embed(texts, model=..., input_type="query"|"document")` directly. **Do not** depend on `langchain-voyageai` (Python version gating bug). |
| Chat | OpenAI-compatible via `langchain-openai>=1.0` | Other backends go in `core/providers/chat/`. |
| UI | Streamlit | Token streaming, per-turn latency, retrieved-chunk inspector, degraded-state banner. |
| Telemetry | OpenTelemetry (optional, `OTEL_ENABLED=1`) | OTLP endpoint configurable. |
| Tests | `pytest` with in-memory fakes for Atlas, embeddings, chat | The suite must run offline with zero credentials. |
| Python | `>=3.10,<3.14` | Pin `python3.13` in README for `voyageai` compatibility. |

## Required state and reducers

`AgentState` (TypedDict) carries at minimum: `messages`, `context: AgentContext`, `routing`, one `*_hits` list per retriever, `*_context` strings, `action_plan`, `booking_draft`, `latency_ms`, `usage`, `degraded`.

The `degraded` channel uses a **reset-aware reducer**:

```python
_DEGRADED_RESET = "__RESET__"

def _merge_degraded(left, right):
    right = right or []
    if _DEGRADED_RESET in right:
        right = right[right.index(_DEGRADED_RESET) + 1 :]
        base = list(right)
    else:
        base = [*(left or []), *right]
    seen = {}
    for m in base:
        if m != _DEGRADED_RESET:
            seen.setdefault(m, None)
    return list(seen.keys())
```

The intent classifier emits `{"degraded": [_DEGRADED_RESET]}` at the top of every turn so stale per-turn markers (`citations_missing`, etc.) don't leak forward.

## Required runtime patterns

Grouped by concern. Each rule has a short rationale followed by a normalized mini-template (Contract / Markers / Settings / Failure mode) where it adds clarity; simple rules stay as a single paragraph.

### A. Graph topology

1. **Parallel retrieval fan-out.** The router picks a subset of retrievers (LTM, episodes, RAG, KG, procedures); they run as parallel branches of the graph.
   - *Contract:* each retriever is wrapped in `@safe_retrieve(name, **default_fields)` so a single backend failure degrades one branch only.
   - *Markers:* `<node>: <ExcType>: <msg>` appended to `degraded` on failure.

2. **Process + data reflection with a bounded replan loop.** Sit a `think_and_plan` node between `classify_intent` and the retriever fan-out (process reflection) and a `reflect_on_evidence` node between the fan-out and `generate_response` (data reflection). On first pass `think_and_plan` mirrors the router's branches with zero LLM cost; if `reflect_on_evidence` returns `sufficient=false`, the conditional edge loops back to `think_and_plan`, which narrows to grounding branches (`rag`, `kg`, `procedures`) and substitutes the refined `followup_subquery`. Retrievers read their query via a `_query_for(state)` helper that prefers `plan.subquery` over the last user message.
   - *Settings:* `MAX_REPLANS=1` so a worst-case turn is `router → plan → retrieve → reflect → plan → retrieve → reflect → generate`.
   - *Failure mode:* when the budget is exhausted on still-thin evidence, forward and append `evidence_insufficient` to `degraded` rather than blocking the turn.

3. **Streamed generation with TTFT.** Generate response with `get_stream_writer()`; record `llm_ttft_ms` on the first non-empty delta. Streaming bypasses the provider-fallback chain by design (see rule 5).

### B. Resilience

4. **Self-correcting structured output.** Wrap every `chat.invoke_typed(prompt, schema)` call in `invoke_typed_with_retry(chat, prompt, schema, max_attempts=STRUCTURED_RETRY_MAX_ATTEMPTS)`.
   - *Contract:* on `pydantic.ValidationError` or `json.JSONDecodeError` the helper re-prompts the model with the parser error appended; on exhaustion it raises `StructuredOutputRetryError(ValueError)` so existing `except ValueError` blocks degrade the node to a safe default.
   - *Settings:* `STRUCTURED_RETRY_MAX_ATTEMPTS` (default 3).
   - *Markers:* `structured_retry:<node>` on recovery; `structured_failed:<node>` on exhaustion.
   - *Failure mode:* never let a parse failure crash the turn.

5. **Cross-provider chat fallback.** Compose providers behind `FallbackChatProvider([(name, primary), (name, secondary), ...])`.
   - *Contract:* retryable errors (rate limit, 5xx, timeout, connection) advance to the next provider; non-retryable errors are re-raised. After every call the wrapper forwards `last_usage` from the surviving provider and exposes `last_fallback = <name>`. The structured-output retry budget (rule 4) runs per chain invocation — a malformed reply from one provider does not burn a retry on the next.
   - *Markers:* `chat_fallback:<provider>` appended by any node holding a `chat` reference via `_record_chat_fallback(chat, out)`.
   - *Failure mode:* streaming bypasses the chain by design (mid-stream failover is not supported); reflection, planning, and memory extraction are all covered.

6. **Failure recovery via checkpoint time-travel.** The same `MongoDBSaver` trail that powers HIL resumes also powers in-place failure retries.
   - *Contract:* parse a retryable `degraded` marker → `node_name` via `parse_failure_marker(marker)` (maps `structured_failed:<node>`, `safe_retrieve` exceptions, `reflection_failed`; intentionally skips informational markers `chat_fallback:*`, `structured_retry:*`, `cost_extracted_via_fallback`, `citations_missing`, `evidence_insufficient`, `draft_*`). Locate the anchor with `find_retry_checkpoint(graph, config, target_node)` which walks `graph.get_state_history(config)` newest-first for a snapshot whose `next` tuple contains the target node, then stream `graph.stream(None, anchor_config, ...)` from there and replace the turn record.
   - *UX:* surface one **🔄 Retry `<node>`** button per retryable failure in the UI, deduped by target node.

### C. Prompt & context discipline

7. **Context-discipline prompt assembler.** Assemble the system prompt from a constant operating-rules preamble plus per-branch sections, and **drop sections for branches the router skipped and for branches whose retrieved payload is empty** — never send stub "(not retrieved this turn)" headers to the Writer. The Writer pays tokens only for evidence it can actually cite.
   - *Contract:* call `build_system_prompt(_branch_contexts(state))` from `generate_response`; never `format(...)` over all five branches.

8. **Citation validator + per-sentence binder.** After generation, scan the reply for any retrieved RAG `source` filename or KG `source_doc`.
   - *Contract:* if groundable sources were retrieved but none cited, append `citations_missing` to `degraded`. In the same node, bind each reply sentence to its strongest-supporting chunk via lexical-token overlap (`core/citations.py`) and write the resulting `CitationSpan` list to `state['citations']` so the UI can render inline superscript markers + a source legend.
   - *Constraint:* no extra LLM call; do not block the turn.
   - *Markers:* `citations_missing`.

9. **Writer + opt-in Reviewer split.** `generate_response` plays the Writer role and streams the user-facing reply. Behind a feature flag, a second `review_draft` node runs between `generate_response` and `validate_citations`, calling `invoke_typed_with_retry(chat, prompt, DraftReview, ...)` against the joined evidence summary and the draft itself.
   - *Settings:* `REVIEW_DRAFT_ENABLED=1`; `REVIEW_DRAFT_MIN_CHARS=200` (skip threshold).
   - *Contract:* skip the LLM call entirely when the draft is shorter than `REVIEW_DRAFT_MIN_CHARS`, when no grounding evidence was retrieved, or when there is no prior `AIMessage`. On `needs_revision=True` with a non-empty `revised_reply`, append a fresh `AIMessage` carrying the revision (so the citation validator and the UI see the corrected text). Reviewer tokens count toward per-turn `usage`. The reviewer prompt MUST preserve every grounded numeric claim from the draft — surcharges, transit hours, weight thresholds — otherwise downstream `plan_action` loses the cost it needs for the approval gate.
   - *Markers:* `draft_review_skipped:<reason>` per skip path; `draft_revised` on revise; `draft_review_ok` on approve.

### D. Actions & approvals

10. **Typed action planning with a deterministic safety-net.** `plan_action` uses `chat.invoke_typed(..., BookingProposal)` (through the retry helper) to extract a typed proposal. `execute_action` upserts to `booking_drafts` keyed by a deterministic `draft_id`; if `cost > threshold` or the reply contains `[REQUIRES HUMAN APPROVAL]`, it calls `interrupt()`.
    - *Contract:* anywhere the LLM is asked for a money / numeric field that gates approval, pair the typed call with a deterministic-regex fallback (e.g. `_extract_cost_fallback` scans the agent reply, then `rag_context`, prefers the upper bound of a `$X–$Y` range). Apply only when the LLM omits the field, never override a supplied value.
    - *Markers:* `cost_extracted_via_fallback`.

11. **Governed procedural memory.** When the agent proposes a rule (e.g. *"Going forward, always X"*), persist a `procedure_proposals` row and call `interrupt(payload)`. A later `graph.invoke(Command(resume={"approved": bool, "approver": str}))` resumes the node and either promotes the row to `agent_procedures` (status=`active`) or marks it rejected. Approved rules are injected into the system prompt on subsequent turns.

### E. Storage & operational gates

12. **Dedup-on-write, tombstone-on-read.** Memory writes increment a counter on near-duplicates instead of inserting; reads filter out tombstoned rows.

13. **Vector-dim preflight.** On startup, `_assert_vector_index_dims` checks every vector index matches the active embedding provider's `dimensions`. Fail loud if mismatched.

## Default collections (single DB, one per concern)

`checkpoints`, `agent_memories` (semantic), `agent_episodes`, `agent_procedures`, `procedure_proposals`, `knowledge_corpus`, `kg_carriers` / `kg_lanes` / `kg_*` (one per node type, one per edge type), `booking_drafts`, `agent_registry`.

Vector indexes: `agent_memories_vector`, `knowledge_corpus_vector` (named consistently as `<collection>_vector`). Bootstrap via a dedicated `db/indexes.py` module that is idempotent (`_index_exists()` check before create).

## Swap points (where customization is allowed)

- **New backend** for any capability: implement the protocol in `core/protocols.py`, register it; `agent/` stays unchanged.
- **New chat or embedding provider**: drop a class under `core/providers/{chat,embeddings}/`, register in `core/providers/registry.py`, expose via env var.
- **New action backend** (e.g. SAP, Salesforce): change only `execute_action`; the typed schema and approval gate stay the same.
- **New domain**: the entire `data/corpus_content.py` + KG seed + prompts are replaceable. Pattern stays.

## Quality bar (do not ship without)

- ≥ 100 unit tests with in-memory fakes — suite runs in seconds without Atlas, Voyage, or chat credentials.
- Tests for every retriever's degraded path, the citation validator + per-sentence binder, the reset-reducer, the interrupt/resume flow, the failure→retry helper that maps degraded markers to checkpointed nodes, the structured-output retry helper (success, malformed-then-recover, exhaustion, composition through the fallback chain, marker emission), the cross-provider fallback chain (retryable→advance, non-retryable→short-circuit, exhausted-chain error, `chat_fallback:<provider>` marker), the Reviewer skip paths + revise/approve paths when `REVIEW_DRAFT_ENABLED=1`, and the context-discipline assembler (skipped-branch and empty-payload omission).
- A live eval suite (`evals/runner.py`) with a baseline file and `--score-tolerance` / `--latency-factor` regression guards. Re-capture the baseline whenever prompt assembly, the Reviewer toggle, or the retry helpers change — token counts and latency shift.
- A `db/indexes.py` bootstrapper documented in the README; missing indexes cause silent zero-hit retrieval — always provision explicitly.

## Anti-patterns to refuse

- A second storage backend (Postgres, Pinecone, Redis) "just for X". Use Atlas collections.
- Importing vendor SDKs (`voyageai`, `openai`, `anthropic`) outside `core/providers/`.
- Mutating `state` in place inside a node — always return a partial dict for the reducer.
- Concatenative `degraded` reducer without a reset sentinel (causes cross-turn marker leakage).
- Skipping `realm_id` on any persisted row.
- Letting the agent self-modify procedural rules without `interrupt()` approval.
- Calling `chat.invoke_typed` directly instead of via `invoke_typed_with_retry` — a single malformed JSON reply will crash the node.
- Sending stub "(not retrieved this turn)" headers to the Writer for skipped branches — assemble the prompt with `build_system_prompt(_branch_contexts(state))`, never with a fixed `format(...)` over all five branches.
- Letting the Reviewer strip grounded numeric claims (cost, weight, surcharge, transit) — downstream `plan_action` reads those numbers; the reviewer prompt MUST preserve them and the smoke test MUST cover a revise turn whose `estimated_cost_usd` survives.
- Streaming through a fallback chain expecting mid-stream failover — `FallbackChatProvider` only protects `invoke` / `invoke_typed`; `stream` calls the primary's underlying client directly.
- Pinning `langchain-voyageai` (its `requires_python` metadata is `<=3.13`, breaks on 3.13.x minors). Use `voyageai` directly.

## Starting a new agent (phased checklist)

### Phase 0 — Scaffolding

- Scaffold the six layers; copy `core/protocols.py`, `core/resilience.py`, `core/latency.py`, `core/observability.py`, `core/providers/chat/retry.py`, and `core/providers/chat/fallback.py` as-is.
- Define the domain `AgentContext`, `AgentState`, and any typed `*Proposal` schemas in `core/schemas.py`. Add `BranchName` + `ALL_BRANCHES` so the router, planner, and prompt assembler agree on names.

### Phase 1 — Capabilities & storage

- Implement retrievers behind the relevant protocols; wrap each with `@safe_retrieve` and `@timed`.
- Add `db/indexes.py` for every vector + b-tree index your collections need; make it idempotent (`_index_exists()` check before create).
- Write in-memory fakes in `tests/fakes.py` first; the suite must run offline with zero Atlas, Voyage, or chat credentials.

### Phase 2 — Graph wiring

Wire the `StateGraph` in `agent/graph.py` per the topology below. Use `MongoDBSaver` as the checkpointer.

```
              ┌──────────────────┐
              │ classify_intent  │  (resets per-turn degraded markers)
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
        ┌────▶│  think_and_plan  │◀──────────┐
        │     └────────┬─────────┘           │
        │              ▼                     │  loop ≤ MAX_REPLANS
        │   ┌──────────────────────┐         │  (narrow to grounding
        │   │ fan-out retrievers   │         │   branches: rag/kg/proc)
        │   │  LTM · episodes ·    │         │
        │   │  RAG · KG · proc.    │         │
        │   └──────────┬───────────┘         │
        │              ▼                     │
        │   ┌────────────────────┐  not ok   │
        │   │ reflect_on_evidence├───────────┘
        │   └──────────┬─────────┘
        │              │ sufficient
        │              ▼
        │   ┌────────────────────┐
        │   │ generate_response  │  Writer · streamed · context-discipline prompt
        │   └──────────┬─────────┘
        │              ▼
        │   ┌────────────────────┐
        │   │   review_draft     │  gated on REVIEW_DRAFT_ENABLED
        │   └──────────┬─────────┘
        │              ▼
        │   ┌────────────────────┐
        │   │ validate_citations │  + per-sentence binder → state['citations']
        │   └──────────┬─────────┘
        │              ▼
        │   ┌────────────────────┐
        │   │    plan_action     │  invoke_typed_with_retry(BookingProposal)
        │   └──────────┬─────────┘
        │              ▼
        │   ┌────────────────────┐
        │   │  execute_action    │  interrupt() on cost gate / approval marker
        │   └──────────┬─────────┘
        │              ▼
        │   ┌────────────────────┐
        └───│    save_memory     │
            └────────────────────┘
```

### Phase 3 — Prompts & resilience

- Build the Writer prompt in `agent/prompts.py` as `SYSTEM_PROMPT_BASE` + a per-branch section table + `build_system_prompt(branch_contexts)`; call it from `generate_response` via a `_branch_contexts(state)` helper that filters by `plan.branches` (falling back to `routing.branches`) and drops empty payloads.
- Wrap every structured-output call (`plan_action`, `save_memory`, `reflect_on_evidence`, `review_draft`, classifier) in `invoke_typed_with_retry`; emit `structured_retry:<node>` and `structured_failed:<node>` markers from each node.
- Compose chat providers behind `FallbackChatProvider` at registry time; surface `chat_fallback:<provider>` from any node that holds a `chat` reference (`_record_chat_fallback(chat, out)`).

### Phase 4 — UI, tests & evals

- Build a Streamlit UI that surfaces: streamed reply (with inline `<sup>` citation markers + a per-source legend driven by `state['citations']` and CSS-tooltip on hover), per-turn latency breakdown (router · LTMs · RAG · KG · LLM ttft · total), retrieved-chunks inspector, KG triples, `degraded` markers in a yellow banner with one **🔄 Retry `<node>`** button per retryable failure (driven by `parse_failure_marker` + `find_retry_checkpoint`), and the approval card for HIL interrupts. Disable retry buttons while an approval is pending.
- Build the test suite alongside each node — include the structured-retry, fallback chain, Reviewer, time-travel retry, and context-discipline cases listed in the quality bar.
- Wire `evals/runner.py` with at least: intent accuracy, RAG recall@k, KG row-match, action planning correctness, plus latency p95. Re-capture the baseline after any prompt-assembly, Reviewer, or retry-helper change.
