# MongoDB-Powered Agent Evaluation Demo

---

## Slide 1 — From Real Agent Traffic to Automated Quality Scores

**The challenge:** Traditional agent evaluation depends on manually labeled
conversations and disconnected observability tools. That is slow, incomplete,
and hard to operate continuously.

**The demo:**

```text
Agent turn → MongoDB checkpoint → live evaluation job → LLM judges → eval_runs → dashboard
```

1. A user asks the supply-chain agent a real question.
2. LangGraph persists the completed turn in MongoDB — including the question,
   final answer, retrieved RAG / knowledge-graph context, citations, routing,
   latency, and degraded-state markers.
3. A scheduled evaluator selects the latest completed checkpoint for each
   conversation, then scores the turn for:
   - **Faithfulness** — are answer claims supported by retrieved evidence?
   - **Answer relevancy** — did the answer address the question?
   - **Context relevancy** — was the retrieved evidence useful?
4. It writes aggregate metrics and per-turn rationale to the `eval_runs`
   collection.
5. The Streamlit **Evaluation** dashboard trends quality and lets an operator
   drill from a low score to its thread, answer, context count, and judge reason.

**Presenter moment:** Run `python -m tools.eval_live_traffic --window-hours 24`
after two completed agent turns, then open the Evaluation tab and investigate a
low-scoring result.

---

## Slide 2 — Why MongoDB Is the Evaluation Platform, Not Just Storage

| MongoDB capability | Benefit for agent evaluation |
| --- | --- |
| **Durable LangGraph checkpoints** | Evaluates what the agent actually did, not only a static benchmark or an exported transcript. |
| **One operational data plane** | Agent memory, RAG/KG grounding data, execution checkpoints, and evaluation history live in one governed platform. |
| **Scoped `eval_runs` history** | Every evaluation run is tied to a `realm_id` and `agent_id`, preventing cross-tenant metric mixing and enabling customer-level dashboards. |
| **Indexed time-window queries** | The `{ realm_id, agent_id, run_at }` index supports efficient 1 / 7 / 30-day quality trends and scheduled monitoring. |
| **Per-turn auditability** | Operators can move from an aggregate score to the exact thread reference, answer preview, context count, and judge rationale. |
| **Automation-ready collections** | A daily CI job, Kubernetes CronJob, or equivalent can continuously write evaluation results, drive alerts, and inform remediation. |

**Business outcome:** MongoDB converts evaluation from a periodic manual audit
into a continuous quality loop: **observe real behavior → score it → retain the
evidence → identify weak turns → improve the agent**.

**Important nuance:** MongoDB does not judge the model itself. LLM judges score
the turn; MongoDB makes the workflow durable, tenant-safe, queryable, and
operationally repeatable.

---

## Slide 3 — Voyage Improves the Evidence; MongoDB Proves the Outcome

**High-quality evaluation starts with high-quality retrieval.** The agent uses
Voyage AI models to find and rank the evidence before it writes an answer:

```text
User question → voyage-4 query embedding → Atlas Vector Search
              → optional rerank-2-lite → grounded context → agent answer
              → MongoDB checkpoints → LLM-as-a-judge → eval_runs
```

| Voyage AI capability | Why it matters in this demo |
| --- | --- |
| **`voyage-4` embeddings** | Places questions and logistics documents in the same semantic space, so the agent can retrieve relevant evidence beyond exact keyword matches. |
| **Query/document retrieval modes** | Uses `input_type="query"` for the user's question and `input_type="document"` for corpus chunks — optimizing each side of semantic retrieval for its job. |
| **Optional `rerank-2-lite` cross-encoder** | Reorders initial RAG candidates so the most relevant evidence reaches the agent prompt first. |
| **Hybrid retrieval compatibility** | Complements Atlas Vector Search with optional keyword retrieval, then reranks the fused candidates for grounded answers. |

**The combined value:**

- **Voyage AI** improves the *quality of the retrieved evidence*.
- **MongoDB Atlas** stores that evidence, checkpointed agent behavior, and
  longitudinal evaluation outcomes in one operational data plane.
- **The evaluator** closes the loop: weak faithfulness or context-relevancy
  scores identify where retrieval, reranking, content, or prompting should be
  improved.

**Key message:** Better retrieval is measurable. Use Voyage to improve the
context; use MongoDB to retain the trace and prove whether answer quality
improves over time.