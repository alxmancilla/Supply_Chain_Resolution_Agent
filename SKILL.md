---
name: langgraph-mongodb-agent
description: Opinionated blueprint for building production-style stateful agents on LangGraph + MongoDB Atlas. Use whenever the user asks to "build an agent", "create an agent", "design a new agent", "scaffold an agent", or discusses agent architecture, memory layers, RAG over Atlas, knowledge graph + agent, procedural rules, human-in-the-loop approval, multi-tenant agents, or tool/action planning. Defaults: LangGraph 1.x StateGraph, MongoDB Atlas (vector + b-tree), Voyage embeddings, Streamlit UI, parallel retrieval fan-out, per-node error isolation, governed procedural memory via interrupt(), citation enforcement, tenant scoping via realm_id, in-memory fakes for tests. Do NOT invent a different stack unless the user explicitly overrides one of these defaults.
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

1. **Parallel retrieval fan-out.** The router picks a subset of retrievers (LTM, episodes, RAG, KG, procedures); they run as parallel branches of the graph. Each retriever is wrapped in `@safe_retrieve(name, **default_fields)` so a single backend failure degrades one branch only.
2. **Streamed generation with TTFT + context discipline.** Generate response with `get_stream_writer()`; record `llm_ttft_ms` on the first non-empty delta. Assemble the system prompt from a constant operating-rules preamble plus per-branch sections, and **drop sections for branches the router skipped and for branches whose retrieved payload is empty** — never send stub "(not retrieved this turn)" headers to the Writer. The Writer pays tokens only for evidence it can actually cite.
3. **Citation validator + per-sentence binder.** After generation, scan the reply for any retrieved RAG `source` filename or KG `source_doc`; if groundable sources were retrieved but none cited, append `"citations_missing"` to `degraded`. In the same node, bind each reply sentence to its strongest-supporting chunk via lexical-token overlap (`core/citations.py`) and write the resulting `CitationSpan` list to `state['citations']` so the UI can render inline superscript markers + a source legend. No extra LLM call. Do not block the turn.
4. **Governed procedural memory.** When the agent proposes a rule (e.g. *"Going forward, always X"*), persist a `procedure_proposals` row and call `interrupt(payload)`. A later `graph.invoke(Command(resume={"approved": bool, "approver": str}))` resumes the node and either promotes the row to `agent_procedures` (status=`active`) or marks it rejected. Approved rules are injected into the system prompt on subsequent turns. The same checkpoint trail also powers in-place failure recovery: parse a retryable `degraded` marker → `node_name`, walk `graph.get_state_history(config)` newest-first for a snapshot whose `next` contains that node, then stream `graph.stream(None, anchor_config, ...)` and replace the turn record.
5. **Typed action planning.** `plan_action` uses `chat.invoke_typed(..., BookingProposal)` to extract a typed proposal. `execute_action` upserts to `booking_drafts` keyed by a deterministic `draft_id`; if `cost > threshold` or the reply contains `[REQUIRES HUMAN APPROVAL]`, it calls `interrupt()`.
6. **Dedup-on-write, tombstone-on-read.** Memory writes increment a counter on near-duplicates instead of inserting; reads filter out tombstoned rows.
7. **Vector-dim preflight.** On startup, `_assert_vector_index_dims` checks every vector index matches the active embedding provider's `dimensions`. Fail loud if mismatched.
8. **Process + data reflection with a bounded replan loop.** Sit a `think_and_plan` node between `classify_intent` and the retriever fan-out (process reflection) and a `reflect_on_evidence` node between the fan-out and `generate_response` (data reflection). On first pass `think_and_plan` mirrors the router's branches with zero LLM cost; if `reflect_on_evidence` returns `sufficient=false`, the conditional edge loops back to `think_and_plan`, which narrows to grounding branches (`rag`, `kg`, `procedures`) and substitutes the refined `followup_subquery`. Cap the loop at `MAX_REPLANS=1` so a worst-case turn is `router → plan → retrieve → reflect → plan → retrieve → reflect → generate`. Retrievers read their query via a `_query_for(state)` helper that prefers `plan.subquery` over the last user message. When the budget is exhausted on still-thin evidence, forward and append `"evidence_insufficient"` to `degraded` rather than blocking the turn.

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
- Tests for every retriever's degraded path, the citation validator, the reset-reducer, the interrupt/resume flow, and the failure→retry helper that maps degraded markers to checkpointed nodes.
- A live eval suite (`evals/runner.py`) with a baseline file and `--score-tolerance` / `--latency-factor` regression guards.
- A `db/indexes.py` bootstrapper documented in the README; missing indexes cause silent zero-hit retrieval — always provision explicitly.

## Anti-patterns to refuse

- A second storage backend (Postgres, Pinecone, Redis) "just for X". Use Atlas collections.
- Importing vendor SDKs (`voyageai`, `openai`, `anthropic`) outside `core/providers/`.
- Mutating `state` in place inside a node — always return a partial dict for the reducer.
- Concatenative `degraded` reducer without a reset sentinel (causes cross-turn marker leakage).
- Skipping `realm_id` on any persisted row.
- Letting the agent self-modify procedural rules without `interrupt()` approval.
- Pinning `langchain-voyageai` (its `requires_python` metadata is `<=3.13`, breaks on 3.13.x minors). Use `voyageai` directly.

## Starting a new agent (checklist)

1. Scaffold the six layers; copy `core/protocols.py`, `core/resilience.py`, `core/latency.py`, `core/observability.py` as-is.
2. Define the domain `AgentContext`, `AgentState`, and any typed `*Proposal` schemas in `core/schemas.py`.
3. Implement retrievers behind the relevant protocols; wrap each with `@safe_retrieve` and `@timed`.
4. Wire the `StateGraph` in `agent/graph.py`: `classify_intent → think_and_plan → parallel retrievers → reflect_on_evidence → {think_and_plan (loop, capped) | generate_response} → validate_citations → plan_action → execute_action → save_memory`. Use `MongoDBSaver` as the checkpointer.
5. Add `db/indexes.py` for every vector + b-tree index your collections need.
6. Write in-memory fakes in `tests/fakes.py` first; build the test suite alongside each node.
7. Build a Streamlit UI that surfaces: streamed reply (with inline `<sup>` citation markers + a per-source legend driven by `state['citations']`), per-turn latency breakdown, retrieved-chunks inspector, KG triples, `degraded` markers, and the approval card for interrupts.
8. Wire `evals/runner.py` with at least: intent accuracy, RAG recall@k, KG row-match, action planning correctness, plus latency p95.
