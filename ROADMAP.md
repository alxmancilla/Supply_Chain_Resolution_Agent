# Roadmap

Post-`v0.1.0` improvements, prioritized. Each item lists the change in scope,
the primary files touched, and acceptance criteria. The ordering reflects
leverage × risk — pick from the top.

The shape of these items is informed by the Bayer / Thoughtworks PRINCE case
study ([Fowler, 2026](https://martinfowler.com/articles/reliable-llm-bayer.html)),
which independently arrived at a very similar harness (LangGraph + checkpointed
state + multi-agent retrieval). We adopt the patterns we don't have, and skip
the ones that don't apply at our scale.

## Done

### 1. Think & Plan node + Reflection Agent (bounded loop) ✅
- **Why:** Today's router picks a retriever set in one shot; on a thin retrieval the LLM answers anyway and `validate_citations` only flags it post hoc. PRINCE's split between *process reflection* (Think & Plan) and *data reflection* (Reflection Agent) catches both failure modes earlier.
- **Scope:** New `think_and_plan` and `reflect_on_evidence` nodes; bounded re-plan loop (`MAX_REPLANS=1`); typed `ResearchPlan` and `EvidenceReflection` schemas; LLM-rescue path triggered only when total RAG+KG hits == 0 on a grounding-required intent.
- **Files:** `core/schemas.py`, `agent/prompts.py`, `agent/nodes.py`, `agent/graph.py`, `tests/test_nodes.py`, `SKILL.md`.
- **Acceptance:** ✅ 11 new tests; 127/127 pass (was 116); loop bounded at `MAX_REPLANS=1`.

### 2. RAG pipeline upgrade — hybrid + reranker ✅
- **Why:** `$vectorSearch` alone leaves recall on the table. PRINCE: metadata filter extraction + hybrid weighted vector/keyword + cross-encoder rerank.
- **Scope (shipped):**
  1. Heuristic metadata filter extraction (`core/rag/query_planner.py`) — regex over lanes (`TX-AZ`, `TX-TX`, `TX-CA`, `AZ-CA`, `TX-NM`), carriers (`Carrier A/B/C`), and doc-type keywords.
  2. Hybrid path in `MongoKnowledgeRetriever`: parallel `$vectorSearch` + `$search` aggregations, fused via weighted reciprocal-rank fusion (RRF, `k=60`); vector-only mode preserved as default.
  3. Optional `VoyageReranker` (`rerank-2-lite`) wired behind `RAG_RERANK_ENABLED`; `NullReranker` is the no-op default.
  4. Per-chunk metadata enrichment at ingest (`metadata.lanes`, `metadata.carriers`) so post-filters work without re-scanning text.
  5. `knowledge_corpus_search` Atlas Search (BM25) index bootstrapped from `db.indexes`; vector index gains `metadata.lanes` / `metadata.carriers` filter fields.
- **Files:** `core/schemas.py`, `core/protocols.py`, `core/rag/{mongo,query_planner,rerank}.py`, `core/settings.py`, `data/seed_corpus.py`, `db/indexes.py`, `.env.example`, `tests/test_nodes.py`.
- **Acceptance:** ✅ 9 new tests; 136/136 pass. Vector-only path unchanged when flags off; hybrid + rerank opt-in via env (`RAG_HYBRID_ENABLED`, `RAG_RERANK_ENABLED`).

### 3. Cross-provider LLM fallback ✅
- **Why:** Provider outages and rate limits happen. PRINCE switches providers after retries.
- **Scope (shipped):**
  1. `FallbackChatProvider((name, primary), *secondaries)` wrapper implementing `ChatProvider`; advances on retryable errors only (rate limit, 5xx/408/429, openai/httpx timeout + connection classes, builtin `TimeoutError` / `ConnectionError`); non-retryable errors re-raise immediately.
  2. `is_retryable_chat_error(exc)` classifier matches by exception class name + `status_code` attribute, so the wrapper has no hard dependency on `openai` or `httpx` at import time.
  3. `CHAT_PROVIDERS` env (comma-separated, ordered) overrides single-`CHAT_PROVIDER` mode; registry composes the chain transparently and validates each name against `CHAT_PROVIDERS` allowlist.
  4. Surviving provider's `last_usage` is forwarded so the per-turn token-accounting path is unchanged.
  5. `agent/nodes.py` adds `_record_chat_fallback(chat, out)` and calls it after each chat invocation in `reflect_on_evidence`, `plan_action`, and `save_memory`; emits `chat_fallback:<provider>` into the `degraded` channel for the turn.
- **Files:** `core/providers/chat/fallback.py` (new), `core/providers/registry.py`, `core/settings.py`, `agent/nodes.py`, `.env.example`, `tests/test_providers.py`.
- **Acceptance:** ✅ 12 new tests; 148/148 pass. Synthetic-primary-failure test through `reflect_on_evidence` produces a reply *and* a `chat_fallback:openai` marker; non-retryable errors short-circuit; exhausted chain raises a `RuntimeError` summarizing every failure.

### 4. Feed error context back into the agent on structured-output retry ✅
- **Why:** `plan_action` swallowed `ValueError` on bad JSON; PRINCE feeds the error + invalid output back to the model for self-correction (capped at 3 attempts).
- **Scope (shipped):**
  1. `invoke_typed_with_retry(chat, prompt, schema, max_attempts)` in `core/providers/chat/retry.py`; loops up to `max_attempts`, catches `pydantic.ValidationError` and `json.JSONDecodeError`, and appends the prior error message + offending output into the next prompt iteration.
  2. `StructuredOutputRetryError(ValueError)` raised on exhaustion so node-level `except ValueError` paths keep degrading cleanly; `last_structured_attempts` exposed on the provider for observability.
  3. Wired into `reflect_on_evidence`, `plan_action`, and `save_memory` in `agent/nodes.py`; emits `structured_retry:<node>` when a retry occurred and `structured_failed:<node>` when all attempts exhausted (state degrades, turn does not crash).
  4. `STRUCTURED_RETRY_MAX_ATTEMPTS` (default 3) added to `core/settings.py` and `.env.example`.
  5. Composes with `FallbackChatProvider` — the retry budget runs per chain invocation, not per provider.
- **Files:** `core/providers/chat/retry.py` (new), `agent/nodes.py`, `core/settings.py`, `.env.example`, `tests/test_providers.py`.
- **Acceptance:** ✅ 8 new tests; 156/156 pass. Malformed-then-valid case returns the parsed object on attempt 2 with `structured_retry:plan_action` marker; exhausted case returns `_NO_ACTION` with `structured_failed:plan_action` marker; retry budget forwards through `FallbackChatProvider` chains.

### 5. Live-traffic evaluation (daily batch) ✅
- **Why:** Baseline evals catch known regressions; live-traffic evals catch drift.
- **Scope (shipped):**
  1. `evals/judges.py` — three LLM-as-judge scorers (`judge_faithfulness`, `judge_answer_relevancy`, `judge_context_relevancy`) returning a clipped `JudgeScore(score 0-1, reason)`; each runs through `invoke_typed_with_retry` so malformed replies self-correct, with a sentinel 0-score on exhaustion.
  2. `tools/eval_live_traffic.py` — CLI that iterates `MongoDBSaver.list(None)` newest-first, dedupes to one terminal checkpoint per `thread_id` inside a rolling window (default 24h), extracts `(last HumanMessage, last AIMessage, joined *_context channels)`, runs the three judges per turn, and writes one `eval_runs` document `{run_id, run_at, window_hours, n_turns, scores: {judge: {mean, n}}, per_turn: [...]}`. Supports `--window-hours`, `--limit`, `--dry-run`, `--out`.
  3. `agent/memory.py` — new `EVAL_RUNS_COLLECTION` constant; `db/indexes.py` — `ensure_eval_runs_index()` provisions the collection and creates `run_at_-1` for the last-7-days dashboard query (wired into `db.indexes` main).
- **Files:** `evals/judges.py` (new), `tools/eval_live_traffic.py` (new), `agent/memory.py`, `db/indexes.py`, `tests/test_evals_live_traffic.py` (new).
- **Acceptance:** ✅ 20 new tests; 176/176 pass. Extractor drops partial states, dedupes by `thread_id`, filters outside the window, and respects `--limit`. Aggregator returns per-judge mean + n (zero-safe on empty input). Judges parse fake LLM output and fall back to a 0-score sentinel after retries exhausted.

### 6. Writer Agent split + draft review loop ✅
- **Why:** `generate_response` synthesizes the user-facing reply *and* is the only chance to catch hallucinations or missing sub-question coverage before `validate_citations` flags them post hoc. A dedicated reviewer that re-reads the draft against the same retrieved context lets the agent self-correct in one bounded pass, without changing the streaming UX when the flag is off.
- **Scope (shipped):**
  1. `generate_response` docstring + role clarified as the Writer; node name kept stable because the Streamlit UI and smoke harness subscribe on `{"node": "generate_response", ...}` stream payloads.
  2. New `review_draft` node behind `REVIEW_DRAFT_ENABLED` (default off). Bypasses the LLM when the draft is shorter than `REVIEW_DRAFT_MIN_CHARS` (default 200), when no grounding evidence was retrieved, or when no prior `AIMessage` exists — emits `draft_review_skipped:<reason>` for observability.
  3. When invoked, calls `invoke_typed_with_retry(chat, prompt, DraftReview, ...)` against `DRAFT_REVIEW_PROMPT`; on `needs_revision=True` with a non-empty `revised_reply`, appends a new `AIMessage` (so `validate_citations` and the UI see the revised text) and emits `draft_revised`. On approve: `draft_review_ok`. On parse exhaustion: `structured_failed:review_draft`. Reviewer tokens count toward per-turn `usage`.
  4. `agent/graph.py` adds the node + `_route_after_writer` conditional edge so topology stays flat (`generate_response -> validate_citations`) when the flag is off.
  5. `DraftReview` schema (`needs_revision`, `revised_reply`, `reasons`); `REVIEW_DRAFT_ENABLED` and `REVIEW_DRAFT_MIN_CHARS` documented in `.env.example`.
- **Files:** `core/schemas.py`, `core/settings.py`, `agent/prompts.py`, `agent/nodes.py`, `agent/graph.py`, `.env.example`, `tests/test_draft_review.py` (new).
- **Acceptance:** ✅ 10 new tests; 186/186 pass. Flag-off path is a true no-op (no chat calls, no markers); enabled path covers short-reply bypass, no-evidence bypass, approve, revise (replaces draft via new AIMessage), blank-revision guard, and `structured_failed:review_draft` on unparseable output. Graph test confirms node wiring and conditional routing.

## P2 — Quality of life

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

Items 1-6 deliver the highest user-visible quality gains (T&P + Reflection,
hybrid RAG, cross-provider fallback, self-correcting structured output,
live-traffic drift detection, and the opt-in draft reviewer). P2 sharpens
trust and operability without reshaping the agent. P3 is on standby until
product scope demands it.
