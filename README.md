# Supply Chain Resolution Agent

A LangGraph demo that runs a supply-chain assistant on **one MongoDB Atlas
cluster**. The same cluster holds the RAG corpus, all three long-term
memory types (semantic, episodic, procedural), a structured knowledge
graph, and the short-term chat checkpoints — no separate vector database
needed.

The scenario: a shipping specialist that recommends carriers, looks up
route and SLA facts, recalls past shipments, follows tenant rules, and
remembers user preferences across sessions.

> See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a layered diagram and
> component walk-through.

## What it demonstrates

- **One cluster, seven workloads.** A single `MongoClient` reaches all of
  RAG (`knowledge_corpus`), the three LTMs (`agent_memories`,
  `agent_episodes`, `agent_procedures`), the knowledge graph (four
  `kg_*` collections), and short-term memory (`checkpoints`).
- **All three LTM types**, per the CoALA / LangGraph taxonomy:

  | Type | What it stores | Retrieval | Example |
  |---|---|---|---|
  | Semantic | Durable user facts/preferences | Vector top-k | "User prefers Carrier A on TX-AZ lanes" |
  | Episodic | Past interactions, summarized | Vector top-k on summary | "Shipped 18k lbs El Paso→Phoenix; booked Carrier A" |
  | Procedural | Curated operating rules | Find-all per tenant | "Always express weights in both lb and kg" |

- **Cross-session recall.** LTM keys are scoped to user + tenant (not
  to `thread_id`), so a brand-new chat still sees the user's preferences
  and past shipments.
- **Per-turn intent routing.** A cheap regex router picks which retrievers
  to run for each question; an LLM router handles anything the regex
  doesn't recognize. Recall questions skip RAG + KG; multi-constraint
  questions skip LTM.
- **Structured knowledge-graph retrieval.** A multi-hop `$graphLookup`
  over a small carrier / lane / SLA graph answers questions like *"which
  carriers serve TX-AZ with no surcharge above 18,000 lbs?"* — same
  cluster, just a different query shape than vector RAG.
- **Per-branch error isolation.** If one retriever fails the rest still
  run and the UI shows a ⚠️ badge; the agent never goes silent because
  one backend hiccuped.
- **Per-turn latency view.** Router, each retriever, LLM (with
  time-to-first-token), plan, execute, and save timings are displayed
  live in the Streamlit UI.
- **Action layer with human-in-the-loop approval.** After the reply,
  `plan_action` extracts a structured `BookingProposal` and
  `execute_action` upserts a row into `booking_drafts`. Bookings over
  `$10,000` (or any reply containing `[REQUIRES HUMAN APPROVAL]`) call
  LangGraph's `interrupt()`, pausing the turn until the operator clicks
  Approve or Reject in the Streamlit UI; `Command(resume=…)` then
  finalizes the draft as `executed` or `rejected`. Draft ids are
  derived deterministically from the correlation id + shipment fields,
  so resuming never creates a duplicate.
- **Response streaming.** `generate_response` streams token deltas via
  LangGraph's `get_stream_writer`; the UI renders them live and records
  `llm_ttft_ms` (wall time to the first non-empty token) alongside
  total `llm_ms`.
- **OpenTelemetry instrumentation.** Each graph node opens an OTel span
  tagged with the correlation id and the tenant / user / agent triple
  from `AgentContext`. A no-op tracer is used until `OTEL_ENABLED=1`
  plus the usual `OTEL_EXPORTER_*` vars are set, so traces cost nothing
  in dev.
- **Vector-index dimension preflight.** On startup, each store reads its
  Atlas Search index definition and fails fast if `numDimensions`
  disagrees with the active embedding provider — a common
  silent-corruption mode when swapping models.
- **Pluggable model providers.** Embedding and chat backends sit behind
  protocols in `core/providers/`. Defaults are Voyage `voyage-4` and
  Grove `gpt-5.5`; switching vendors means writing one class.
- **Cross-provider chat fallback.** When `CHAT_PROVIDERS` lists more
  than one backend (e.g. `grove,openai`), the registry composes a
  `FallbackChatProvider` that advances to the next entry on retryable
  failures (rate limit, 5xx/408/429, openai/httpx timeouts, connection
  errors). Non-retryable errors short-circuit. Each fallback emits a
  `chat_fallback:<provider>` marker into the `degraded` channel for
  the turn so the UI can surface that a backup model answered.
- **Eval harness with a baseline gate.** `python -m evals.runner` scores
  four suites (intent accuracy, RAG recall, KG row-match, and action
  planning) and exits non-zero if any score drops vs. the committed
  baseline. Runs against fakes in CI or against live Atlas to refresh
  the baseline.
- **Memory dedup + periodic consolidation.** When the agent writes a new
  fact, it first checks for a near-duplicate and just bumps a counter if
  one exists. A separate `python -m tools.reflect` pass clusters similar
  facts and asks the LLM to merge each cluster into a single canonical
  row, soft-deleting the originals so retrieval stays clean.

## Architecture

```
                                       ┌─> retrieve_ltm         ──> agent_memories    (semantic)
                                       ├─> retrieve_episodes    ──> agent_episodes    (episodic)
START ──> classify_intent ──(conditional fan-out)─┤
                                       ├─> retrieve_procedures  ──> agent_procedures  (procedural)         ──> generate_response ──> plan_action ──> execute_action ──> save_memory ──> END
                                       ├─> retrieve_rag         ──> knowledge_corpus  (RAG)                     (streamed)         (BookingProposal)   (interrupt if      │
                                       └─> retrieve_kg          ──> kg_* (4 colls, $graphLookup)                                                        > $10k)            ▼
                                                                                                                                                                      writes to agent_memories,
                                                                                                                                                                      agent_episodes, booking_drafts
```

Every turn: the router picks which retrievers should run, the selected
ones fan out in parallel, results flow into the LLM, the reply is
streamed back to the UI, then `plan_action` extracts a structured
booking proposal and `execute_action` upserts a draft (pausing for
human approval when required). Finally, semantic + episodic facts are
written.

## Reusable layers

The agent is split into two parts:
- `agent/` — the LangGraph wiring (specific to this demo).
- `core/` — reusable retrieval, memory, router, and provider modules
  that another agent could import as-is.

Graph nodes only talk to `core/` through protocols (interfaces). Each
protocol has a real Mongo-backed implementation and an in-memory fake
for tests, so backends and providers are swappable.

| Layer | Module | What it does |
|---|---|---|
| Schemas | `core/schemas.py` | Pydantic types passed between nodes |
| Protocols | `core/protocols.py` | Interfaces for memory, RAG, KG, router, and providers |
| Providers | `core/providers/` | Voyage embeddings + Grove chat behind a registry |
| RAG | `core/rag/mongo.py` | `$vectorSearch` over the knowledge corpus |
| KG | `core/kg/` | Multi-hop `$graphLookup` + a regex entity extractor |
| Memory | `core/memory/{semantic,episodic,procedural}.py` | One class per LTM type; dedup on write, tombstone-aware on read |
| Reflection | `core/memory/reflector.py` | Clusters near-duplicate facts and merges each cluster into a canonical row |
| Router | `core/router.py` | Heuristic regex first, LLM fallback |
| Cross-cutting | `core/latency.py`, `core/resilience.py`, `core/observability.py` | Per-node timing + OTel spans + per-branch failure isolation |
| Settings | `core/settings.py` | Env loader + per-turn `AgentContext` (tenant, user, agent, correlation ids) |

## Project layout

```
.
├── app.py                       # Streamlit UI
├── agent/                       # LangGraph wiring for this demo
│   ├── memory.py                #   Shared MongoClient + stores + checkpointer + booking drafts
│   ├── prompts.py               #   System + extraction + action-planning + consolidation prompts
│   ├── nodes.py                 #   Graph nodes (thin adapters over core/)
│   └── graph.py                 #   StateGraph: router → fan-out → response → plan → execute → save
├── core/                        # Reusable layers (no demo-specific code)
│   ├── schemas.py               #   Pydantic types (incl. BookingProposal)
│   ├── protocols.py             #   Interfaces (memory, RAG, KG, router, providers)
│   ├── settings.py              #   Env loader + AgentContext (with correlation_id)
│   ├── latency.py               #   @timed decorator (wraps each node in an OTel span)
│   ├── observability.py         #   OpenTelemetry tracer (no-op until OTEL_ENABLED=1)
│   ├── resilience.py            #   @safe_retrieve decorator
│   ├── usage.py                 #   Token-usage extraction + per-1k cost reducer
│   ├── router.py                #   Heuristic + LLM intent routers
│   ├── providers/               #   Model backends
│   │   ├── registry.py          #     Provider dispatch by env (single or fallback chain)
│   │   ├── embeddings/voyage.py #     Voyage voyage-4 (1024-dim)
│   │   └── chat/
│   │       ├── grove.py         #     Grove GPT-5.5
│   │       └── fallback.py      #     FallbackChatProvider + retryable-error classifier
│   ├── rag/mongo.py             #   $vectorSearch over knowledge_corpus
│   ├── kg/                      #   $graphLookup + entity extractor
│   └── memory/
│       ├── semantic.py          #   Semantic LTM (dedup on write)
│       ├── episodic.py          #   Episodic LTM (dedup on write)
│       ├── procedural.py        #   Procedural LTM (curated rules)
│       └── reflector.py         #   Consolidation pass
├── data/                        # One-shot seed scripts
│   ├── corpus_content.py        #   17 docs → 103 chunks
│   ├── seed_corpus.py / seed_memories.py / seed_episodes.py
│   ├── seed_procedures.py / seed_kg.py
├── db/indexes.py                # Creates vector + b-tree indexes
├── evals/                       # Offline eval suite
│   ├── runner.py                #   CLI: --mode {fast,live,latency}
│   ├── metrics/                 #   intent_accuracy, rag_recall_at_k, kg_row_match,
│   │                            #   action_planning_accuracy, latency_p50_p95
│   ├── baseline.json            #   Committed live scores (4-metric suite)
│   ├── latency_baseline.json    #   Committed latency-mode p50/p95 sample
│   └── datasets/*.jsonl         #   20 + 13 + 6 + 6 + 5 labeled cases
├── tests/                       # 148 unit tests (in-memory fakes, no Atlas)
├── tools/
│   ├── demo.py                  #   One-turn end-to-end demo
│   ├── smoke_turn.py            #   End-to-end 3-turn smoke
│   ├── reflect.py               #   Run the consolidation pass
│   └── cleanup_memories.py      #   Delete LTM rows
├── requirements.txt
└── .env.example
```

## Prerequisites

- Python 3.11+ (tested on 3.13)
- MongoDB Atlas cluster with Vector Search enabled (M10+ recommended; M30 + S20
  search nodes for a snappy demo)
- API keys: Voyage AI (`voyage-4`), Grove gateway (GPT-5.5)

## Setup

```bash
# 1. Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and fill in MONGODB_URI, GROVE_API_KEY, VOYAGE_API_KEY
```

## Provision and seed

Run these once per cluster:

```bash
python -m db.indexes            # 3 vector indexes + 4 KG b-tree indexes
python -m data.seed_corpus      # 103 chunks across 17 docs (RAG corpus)
python -m data.seed_memories    # Semantic LTM (user preferences)
python -m data.seed_episodes    # Episodic LTM (past shipments)
python -m data.seed_procedures  # Procedural LTM (tenant rules)
python -m data.seed_kg          # Knowledge graph (3 carriers, 3 lanes, 6 serves, 6 SLAs)
```

`db.indexes` creates the three vector indexes plus four compound b-tree
indexes that back the KG joins. Procedural memory is a flat rule list
with no index.

## Run

### Streamlit UI

```bash
streamlit run app.py
```

Chat on the left, a tabbed memory inspector on the right (Semantic,
Episodic, Procedural, Knowledge Graph). Each turn shows the router's
decision, the hits from each retriever, the streamed reply, and a
latency strip (router · LTMs · RAG · KG · LLM with `ttft` · total).
When a turn proposes a booking over `$10,000`, the chat input is
disabled and an inline approval card with **Approve & execute** /
**Reject** buttons appears. The graph state is held in the LangGraph
checkpointer; resuming with `Command(resume=…)` finalizes the draft.
**New Session** starts a fresh chat but keeps the user's LTM, the KG,
and prior `booking_drafts`.

### Quick one-turn demo

```bash
python -m tools.demo
python -m tools.demo "Which carriers serve TX-AZ under 18000 lbs?"
```

Runs a single question through the full graph and prints the router's
decision, per-branch hit counts and latencies, and the agent's reply.
With no argument it uses a default shipment question.

### Headless smoke test

```bash
python -m tools.smoke_turn
```

Three scripted turns:
1. **Recommend a shipment** — all five retrievers run; the agent picks
   a carrier and saves a new semantic fact + episode.
2. **Recall a preference** (new chat) — only LTM + episodes run;
   cross-session recall works.
3. **Multi-constraint KG lookup** — only KG + RAG run; the graph query
   returns the matching `(carrier, lane, SLA)` row.

Output includes per-phase latencies (router · LTMs · RAG · KG ·
`llm_ttft_ms` · `llm_ms` · save) and hit counts. The Turn 1 summary
also prints live-vs-raw memory counts (e.g. `5 live (of 50 raw)`) so
the dedup + consolidation effect is visible. When a turn proposes a
booking that needs approval, the script auto-approves with
`Command(resume={"approved": True})` and prints the resulting draft.
A ⚠️ line appears if any retriever degrades.

### Eval harness

```bash
python -m evals.runner --mode fast                                          # CI; no Atlas
python -m evals.runner --mode live --baseline evals/baseline.json           # against Atlas
python -m evals.runner --mode latency --runs 3 \
    --baseline evals/latency_baseline.json                                  # tail-latency sample
python -m evals.runner --mode fast --against evals/baseline.json            # regression gate
```

Four correctness suites run in `fast` / `live`: intent accuracy (20
cases), RAG recall (13), KG row-match (6), and action planning
accuracy (6 — checks both the extracted `action_type` and the
`requires_approval` gate). Current live baseline: 1.000 across all
four.

`--mode latency` drives the compiled graph `--runs` times per prompt
(default 3) over `evals/datasets/latency.jsonl` (5 representative
shipments / recall / KG questions) and reports per-case + overall p50
/ p95 / min / max of `llm_ttft_ms` and `llm_ms`. The committed
`evals/latency_baseline.json` is a sample of one such run against the
live cluster, useful as a sanity floor when comparing latency after a
provider or prompt change.

`--against <baseline.json>` loads a previously committed suite result
and compares it to the current run. The runner prints a diff table and
exits **2** if any metric's score drops by more than `--score-tolerance`
(default 0.01), or — for the latency metric — if `ttft_ms.p95` or
`llm_ms.p95` exceeds the baseline by more than `--latency-factor`
(default 1.5×). New metrics (present in the current run but not the
baseline) are reported informationally and never trigger a regression.

### Memory reflection

```bash
python -m tools.reflect                              # defaults: cosine ≥ 0.92
python -m tools.reflect --threshold 0.88 --dry-run   # preview merges first
```

Runs the consolidation pass over live semantic + episodic memory:
groups near-duplicates, asks the chat model for one canonical phrasing
per group, writes that as a `canon_…` row, and soft-deletes the
originals. Prints how many clusters, canonicals, and tombstones were
produced.

### Unit tests

```bash
python -m pytest tests/ -v
```

148 tests covering the retrievers, the router, the KG layer, the
provider protocols (including settings-driven model-name overrides
and the cross-provider `FallbackChatProvider` chain — classifier,
retryable→fallback, non-retryable short-circuit, exhausted-chain
error, registry composition, and the `chat_fallback:<provider>`
marker emitted by `reflect_on_evidence`), the hybrid RAG path
(vector + BM25 RRF + reranker), the think-and-plan / reflection
loop, memory dedup + tombstones, the reflector (both the standalone
clustering pass and the in-graph `REFLECT_EVERY_N_TURNS` trigger),
the streamed `generate_response` path (TTFT included), the citation
validator, the token/cost accounting helpers + per-node usage
plumbing, and the eval harness in fast mode (including latency
percentile math and `--against` baseline diffing). Runs against
in-memory fakes — no Atlas, Voyage, or Grove credentials needed.

## Configuration

| Env var                     | Purpose                                                    | Default                  |
|-----------------------------|------------------------------------------------------------|--------------------------|
| `MONGODB_URI`               | Atlas connection string                                    | *(required)*             |
| `GROVE_API_KEY`             | Grove gateway key (only if using the Grove chat provider)  | *(required if `CHAT_PROVIDER=grove`)* |
| `VOYAGE_API_KEY`            | Voyage AI key (only if using the Voyage embedding provider) | *(required if `EMBEDDING_PROVIDER=voyage`)* |
| `EMBEDDING_PROVIDER`        | Which embedding backend to use                             | `voyage`                 |
| `CHAT_PROVIDER`             | Which chat backend to use                                  | `grove`                  |
| `CHAT_PROVIDERS`            | Optional ordered, comma-separated fallback chain (e.g. `grove,openai`). When set with >1 entry, `get_chat_provider()` returns a `FallbackChatProvider` that advances on retryable errors; emits `chat_fallback:<name>` into `degraded`. Unset → use `CHAT_PROVIDER` alone | *(unset)* |
| `EMBEDDING_MODEL`           | Model name passed to the embedding provider                | `voyage-4`               |
| `CHAT_MODEL`                | Model name passed to the chat provider                     | `gpt-5.5`                |
| `REALM_ID`                  | Tenant scope (all collections)                             | `customer-tenant-001`    |
| `USER_ID`                   | User scope (semantic + episodic LTM)                       | `user-demo`              |
| `AGENT_ID`                  | Agent scope (procedural rules + KG)                        | `supply-chain-resolution-agent` |
| `SEMANTIC_DEDUP_THRESHOLD`  | Cosine threshold above which semantic writes bump a counter instead of inserting | `0.92` |
| `EPISODIC_DEDUP_THRESHOLD`  | Same, for episodic writes                                  | `0.92`                   |
| `CHAT_INPUT_PRICE_PER_1K_USD`  | Per-1k input-token price used to convert per-turn usage_metadata into `cost_usd` (0 disables) | `0.0` |
| `CHAT_OUTPUT_PRICE_PER_1K_USD` | Per-1k output-token price (same)                        | `0.0`                    |
| `REFLECT_EVERY_N_TURNS`     | When > 0, `save_memory` runs `LLMMemoryReflector` against semantic + episodic LTM every N successful turns per (realm, user); 0 keeps reflection a manual `tools/reflect.py` job | `0` |
| `REFLECT_THRESHOLD`         | Cosine threshold passed to the in-graph reflector when scheduling is on | `0.88`                   |
| `OTEL_ENABLED`              | When `1`, configure an OTLP exporter on startup; otherwise the per-node spans are no-ops | `0` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Standard OTel collector endpoint (used when `OTEL_ENABLED=1`) | *(unset)*               |

Unknown provider names fail at startup (validated against an allow-list
in `core/settings.py`).

## Maintenance

Delete LTM rows. By default the cleanup hits both user-scoped collections
(semantic + episodic). Procedural rules are tenant-scoped, not user-scoped.

```bash
# Clear semantic + episodic for one user (interactive confirmation)
python -m tools.cleanup_memories --user <user_id>

# Just one type
python -m tools.cleanup_memories --user <user_id> --type semantic --yes
python -m tools.cleanup_memories --user <user_id> --type episodic --yes

# Wipe all procedural rules for a tenant
python -m tools.cleanup_memories --type procedural --realm <realm_id> --yes
```

## Notes

- **Provider boundary.** Vendor SDK imports (`langchain-voyageai`,
  `langchain-openai`) live only inside `core/providers/`. Everything
  else talks to the provider protocols, so swapping vendors does not
  touch `agent/`, `core/memory/`, `core/rag/`, or `core/kg/`.
- **Cross-provider chat fallback.** Off by default — when
  `CHAT_PROVIDERS` is unset the registry returns the single configured
  chat provider (no wrapper allocated). When set with >1 entry,
  `get_chat_provider()` returns a `FallbackChatProvider` that sequences
  the chain on every `invoke` / `invoke_typed`; retryable failures
  (`is_retryable_chat_error`: rate limit, 5xx/408/429, openai/httpx
  timeout + connection classes, builtin `TimeoutError` /
  `ConnectionError`) advance to the next provider, non-retryable
  errors re-raise immediately, and an exhausted chain raises a
  `RuntimeError` summarizing every failure. The classifier matches by
  exception class name + `status_code` attribute, so the wrapper has
  no hard import-time dependency on `openai` or `httpx`. Whichever
  provider answers has its `last_usage` forwarded, so per-turn token
  accounting is unchanged. `agent/nodes.py` calls
  `_record_chat_fallback` after each chat invocation in
  `reflect_on_evidence`, `plan_action`, and `save_memory` and appends
  a single de-duped `chat_fallback:<provider>` marker into the
  `degraded` channel for the turn. The streaming path
  (`generate_response`) uses `chat.underlying()` and bypasses the
  wrapper — mid-stream failover is not supported.
- **Memory write path.** Each turn, `save_memory` runs **one**
  structured-output call against the chat provider to extract facts +
  one episode summary, then writes them through the dedup-aware `put`:
  if a near-duplicate already exists, bump its `seen_count` instead of
  inserting. Procedural memory is curated, not auto-extracted.
- **Action layer.** After the reply, `plan_action` uses
  `chat.invoke_typed(..., BookingProposal)` to extract a typed proposal.
  `execute_action` upserts it into `booking_drafts` with a deterministic
  `draft_id` derived from `(correlation_id, realm, user, carrier, lane,
  weight, cost)` so retries after an interrupt update the existing row
  instead of creating a duplicate. When `requires_approval` is true the
  node calls LangGraph's `interrupt()`; resuming with
  `Command(resume={"approved": …, "approver": …})` flips the row to
  `executed` or `rejected`.
- **Streaming + TTFT.** `generate_response` streams token deltas through
  LangGraph's custom-channel `get_stream_writer`. The first non-empty
  delta also records `llm_ttft_ms` into `state['latency_ms']`, which
  the Streamlit latency strip and the `demo.py` / `smoke_turn.py`
  output surface alongside `llm_ms`.
- **Token + cost accounting.** Every chat call (`generate_response`,
  `plan_action`, `save_memory`) records `usage_metadata` from the
  underlying provider. A reducer on the `usage` channel sums
  `tokens_in`, `tokens_out`, `calls`, and `cost_usd` per turn;
  `cost_usd` is computed from `CHAT_INPUT_PRICE_PER_1K_USD` and
  `CHAT_OUTPUT_PRICE_PER_1K_USD` in `Settings` (both default to 0, in
  which case `cost_usd` stays at 0 but token counts are still surfaced).
  The Streamlit latency strip, `demo.py`, and `smoke_turn.py` all show
  the per-turn totals.
- **Observability.** `@timed` opens an OTel span per node tagged with
  the correlation id, the tenant/user/agent triple, the intent label,
  and `agent.latency_ms`. A fresh `correlation_id` is minted into
  `AgentContext` per turn (in `_run_turn`). Without
  `OTEL_ENABLED=1`, the global tracer is a no-op, so the spans cost
  nothing.
- **Index dimension preflight.** `_assert_vector_index_dims` reads the
  existing Atlas Search index definition for each store (`agent_memories`,
  `agent_episodes`, `knowledge_corpus`) and raises at startup if the
  declared `numDimensions` doesn't match the active embedding provider —
  the failure mode that previously surfaced as silently empty
  `$vectorSearch` results.
- **Memory read path.** `search` filters out rows marked `tombstoned`,
  so consolidated originals disappear from retrieval automatically.
- **Consolidation.** `tools.reflect` groups near-duplicate live rows
  by cosine similarity, asks the chat model to merge each group into
  one canonical row, writes that row, and tombstones the originals.
  The smoke test surfaces the effect (`5 live (of 50 raw)`). For
  unattended deployments, set `REFLECT_EVERY_N_TURNS=N` to have
  `save_memory` run the same reflector inline every N successful
  turns per `(realm, user)`; failures degrade the turn but do not
  crash it.
- **Eval as a regression gate.** Fast mode runs against in-memory fakes
  (CI-friendly); live mode runs against Atlas. With `--baseline`, the
  runner exits non-zero if any score drops — safe to land dedup /
  reflection / prompt changes without silently breaking retrieval.
- **Per-branch resilience.** If one retriever raises, the others still
  run, the LLM still answers (with reduced grounding), and the UI
  shows a ⚠️ badge for that turn.
- **Intent router.** Cheap regex first (sub-millisecond). If nothing
  matches, an LLM call decides which retrievers to run.
- **Knowledge graph.** A `$graphLookup` from the queried lane out to
  carriers, then a second join to find other lanes those carriers
  also serve. Hop-1 rows are filtered by the user's constraints (max
  surcharge, min weight threshold) extracted by a regex; hop-2 rows
  only ride along for carriers that survived hop-1. Every row carries
  back-references to its source RAG document so the LLM can cite it.
- **Versioning quirk.** `langchain-voyageai 0.1.3` pins
  `langchain-core<0.4`, but `langgraph 1.x` wants `>=1.4`. Pip prints a
  resolver warning; the `Embeddings` interface is stable and
  end-to-end works at runtime.
- **Cold-start indexes.** First provision on a fresh cluster can take
  30–60 s; `MongoDBStore` is configured with `auto_index_timeout=70`.
