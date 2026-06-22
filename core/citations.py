"""Sentence-level citation matcher.

Maps each sentence of the agent's reply to the single best-supporting
retrieved chunk via lexical token overlap. Pure-Python and synchronous —
no extra LLM call — so it can run inside `validate_citations` without
adding meaningful latency or token cost.

The output `CitationSpan` records carry enough context for the Streamlit
UI to render a superscript marker after the sentence and an expandable
side panel showing the source chunk text + filename.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

# Sentence boundary: end-of-sentence punctuation followed by whitespace and
# a capital letter / digit / opening bracket. Conservative on purpose —
# better to over-merge than to fragment numeric claims like "$410.0 all-in".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(`*\-])")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")

# Stopwords + extremely common shipping-domain filler that would otherwise
# match every chunk and drown the signal.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "but", "you", "your", "our", "their", "they", "them", "have", "has",
    "had", "not", "any", "all", "can", "may", "via", "per", "into",
    "about", "above", "below", "between", "under", "over", "than",
    "when", "where", "what", "which", "who", "whom", "why", "how",
    "also", "such", "only", "still", "just", "very", "more", "most",
    "use", "using", "used", "see", "based", "would", "should", "could",
    "will", "shall", "must", "let", "lets", "let's", "make", "made",
    "get", "got", "one", "two", "three",
})

DEFAULT_MIN_OVERLAP = 2


class CitationSpan(TypedDict):
    """A single citation: sentence -> best-matching retrieved chunk."""

    sentence_idx: int
    start: int
    end: int
    sentence: str
    kind: str  # "rag" | "kg"
    source: str
    doc_type: str
    chunk_id: str
    score: float
    overlap: int
    evidence: str


def _tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text or "")
        if m.group(0).lower() not in _STOPWORDS
    }


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split `text` into (start, end, sentence) spans preserving offsets."""
    if not text:
        return []
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for part in _SENTENCE_SPLIT_RE.split(text):
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        out.append((start, end, part))
        cursor = end
    return out


def _index_chunks(
    rag_hits: list[dict[str, Any]] | None,
    kg_hits: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, hit in enumerate(rag_hits or []):
        text = (hit or {}).get("text") or ""
        if not text:
            continue
        source = hit.get("source", "")
        chunks.append({
            "kind": "rag",
            "tokens": _tokens(text),
            "text": text,
            "source": source,
            "doc_type": hit.get("doc_type", ""),
            "chunk_id": hit.get("chunk_id") or f"{source}#{idx}",
            "score": float(hit.get("score") or 0.0),
        })
    for idx, hit in enumerate(kg_hits or []):
        fact = (hit or {}).get("fact") or ""
        if not fact:
            continue
        chunks.append({
            "kind": "kg",
            "tokens": _tokens(fact),
            "text": fact,
            "source": hit.get("source_doc") or "knowledge_graph",
            "doc_type": "kg_fact",
            "chunk_id": hit.get("edge_id") or f"kg#{idx}",
            "score": float(hit.get("score") or 1.0),
        })
    return chunks


def match_citations(
    reply: str,
    rag_hits: list[dict[str, Any]] | None,
    kg_hits: list[dict[str, Any]] | None,
    *,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> list[CitationSpan]:
    """Return per-sentence citations whose token overlap >= `min_overlap`.

    Strategy: for each sentence pick the single chunk with the highest
    distinct-token overlap; break ties by retrieval `score`. Sentences
    that fail the overlap floor are skipped (the reply may contain
    framing / boilerplate that need not be cited; `validate_citations`
    already handles the fully uncited case via `degraded`).
    """
    sentences = split_sentences(reply)
    if not sentences:
        return []
    chunks = _index_chunks(rag_hits, kg_hits)
    if not chunks:
        return []
    out: list[CitationSpan] = []
    for idx, (start, end, sentence) in enumerate(sentences):
        s_tokens = _tokens(sentence)
        if not s_tokens:
            continue
        best: dict[str, Any] | None = None
        best_overlap = 0
        for chunk in chunks:
            overlap = len(s_tokens & chunk["tokens"])
            if overlap > best_overlap or (
                overlap == best_overlap
                and overlap > 0
                and best is not None
                and chunk["score"] > best["score"]
            ):
                best = chunk
                best_overlap = overlap
        if best is None or best_overlap < min_overlap:
            continue
        out.append(CitationSpan(
            sentence_idx=idx,
            start=start,
            end=end,
            sentence=sentence,
            kind=best["kind"],
            source=best["source"],
            doc_type=best["doc_type"],
            chunk_id=best["chunk_id"],
            score=best["score"],
            overlap=best_overlap,
            evidence=best["text"],
        ))
    return out
