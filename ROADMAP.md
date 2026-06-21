# Roadmap

Post-`v0.1.0` improvements, prioritized. Each item lists the change in scope,
the primary files touched, and acceptance criteria. The ordering reflects
leverage × risk — pick from the top.

The shape of these items is informed by the Bayer / Thoughtworks PRINCE case
study ([Fowler, 2026](https://martinfowler.com/articles/reliable-llm-bayer.html)),
which independently arrived at a very similar harness (LangGraph + checkpointed
state + multi-agent retrieval). We adopt the patterns we don't have, and skip
the ones that don't apply at our scale.

## P0 — Done

### 1. Think & Plan node + Reflection Agent (bounded loop) ✅
- **Why:** Today's router picks a retriever set in one shot; on a thin retrieval the LLM answers anyway and `validate_citations` only flags it post hoc. PRINCE's split between *process reflection* (Think & Plan) and *data reflection* (Reflection Agent) catches both failure modes earlier.
- **Scope:** New `think_and_plan` and `reflect_on_evidence` nodes; bounded re-plan loop (`MAX_REPLANS=1`); typed `ResearchPlan` and `EvidenceReflection` schemas; LLM-rescue path triggered only when total RAG+KG hits == 0 on a grounding-required intent.
- **Files:** `core/schemas.py`, `agent/prompts.py`, `agent/nodes.py`, `agent/graph.py`, `tests/test_nodes.py`, `SKILL.md`.
- **Acceptance:** ✅ 11 new tests; 127/127 pass (was 116); loop bounded at `MAX_REPLANS=1`.

## P1 — Next

### 2. RAG pipeline upgrade — hybrid + reranker
- **Why:** `$vectorSearch` alone leaves recall on the table. PRINCE: metadata filter extraction + n=5 query expansion + hybrid weighted (0.7 vector / 0.3 keyword) + cross-encoder rerank top-20 → top-7.
- **Scope:**
  1. Extract metadata filters (e.g. `lane`, `carrier`, `doc_type`) via LLM and inject into `$vectorSearch.filter`.
  2. Add Atlas `$search` (BM25) phase; fuse via `$rankFusion` (Atlas 8.1+) or manual weighted union.
  3. Add reranker stage using `voyageai.Client.rerank(model="rerank-2-lite")` after retrieval.
  4. Tag corpus chunks at ingest with structured metadata (`lane`, `doc_type`, `section`).
- **Files:** `core/rag/mongo.py`, `data/corpus_content.py`, `data/seed_corpus.py`, `db/indexes.py`, new `core/rag/rerank.py`, new `core/rag/query_planner.py`.
- **Acceptance:** RAG-recall eval score improves vs. baseline; reranker can be disabled via env var for cost control.

### 3. Cross-provider LLM fallback
- **Why:** Provider outages and rate limits happen. PRINCE switches providers after retries.
- **Scope:** `FallbackChatProvider(primary, *secondaries)` wrapper implementing `ChatProvider`; switches on `RateLimitError`, 5xx, `httpx.TimeoutException`; logs fallback to `degraded` channel.
- **Files:** `core/providers/chat/fallback.py` (new), `core/providers/registry.py`, `core/settings.py`, `tests/test_providers.py`.
- **Acceptance:** Synthetic primary-failure test triggers secondary and produces a response; `degraded` carries `chat_fallback:<provider>` marker.

### 4. Feed error context back into the agent on structured-output retry
- **Why:** `plan_action` swallows `ValueError` on bad JSON; PRINCE feeds the error + invalid output back to the model for self-correction (capped at 3 attempts).
- **Scope:** New `invoke_typed_with_retry(prompt, schema, max_attempts=3)` on `ChatProvider`; passes previous error message and bad output into the next prompt iteration.
- **Files:** `core/protocols.py`, `core/providers/chat/grove.py`, `agent/nodes.py` (`plan_action`).
- **Acceptance:** Test where the first JSON is malformed but the second succeeds; production success rate on `plan_action` improves.

### 5. Live-traffic evaluation (daily batch)
- **Why:** Baseline evals catch known regressions; live-traffic evals catch drift.
- **Scope:** `tools/eval_live_traffic.py` reads last 24h of `checkpoints`, replays retrieved chunks + responses through judge prompts (Faithfulness, Answer Relevancy, Context Relevancy), writes scores to `eval_runs`. Cron / GitHub Action friendly.
- **Files:** `tools/eval_live_traffic.py` (new), `evals/judges.py` (new), `db/indexes.py`.
- **Acceptance:** Script produces a JSON summary; dashboard query returns last 7 days of scores.

## P2 — Quality of life

### 6. Writer Agent split + draft review loop
- **Why:** Today's `generate_response` handles both synthesis and formatting. A dedicated Writer with an optional 1-pass draft reviewer enables completeness checks for complex outputs.
- **Scope:** Rename + reshape `generate_response` into a Writer-role node; add optional `review_draft` node behind a feature flag.
- **Files:** `agent/nodes.py`, `agent/prompts.py`, `agent/graph.py`.

### 7. Per-sentence citations in the Streamlit UI
- **Why:** Granular citations (hover to source + page) significantly raise the trust ceiling. PRINCE's UX is the bar.
- **Scope:** Inline superscript markers in the streamed reply; side panel rendering the matched chunk + source link.
- **Files:** `app.py`, `core/citations.py` (new), `agent/nodes.py` (annotate citation positions).

### 8. Resume-from-failed-node UX
- **Why:** Already have checkpoint-based resume for interrupts; extend it to failure recovery.
- **Scope:** "Retry from failure" button in Streamlit using `graph.invoke(Command(resume=...))` against the persisted checkpoint at the failed node.
- **Files:** `app.py`, `agent/graph.py`.

### 9. Context-discipline audit
- **Why:** `generate_response` currently receives every `*_context` blob regardless of router choice (modulo `_ctx_for`). Trim more aggressively per-intent.
- **Scope:** Per-intent prompt assembly in `agent/prompts.py`; remove unused branches' contexts entirely (not just stub them).
- **Files:** `agent/prompts.py`, `agent/nodes.py`.

## P3 — Deferred (not justified yet)

- **Domain sub-agent hierarchy** — Only justified when a second vertical is added beyond logistics.
- **NER data-quality utility with confidence scoring** — Only justified when the corpus moves from curated `data/corpus_content.py` to messy real-world PDFs.
- **Langfuse-style trace store** — Current OTel setup is sufficient; revisit if a richer trace UX becomes a need.

## Sequencing rationale

P0/P1 deliver the highest user-visible quality gains (T&P + Reflection, hybrid
RAG, cross-provider fallback). P2 sharpens trust and operability without
reshaping the agent. P3 is on standby until product scope demands it.
