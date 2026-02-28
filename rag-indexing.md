# rag-indexing.md
`votez-activity-analyzer`

This document defines the RAG (Retrieval-Augmented Generation) indexing and retrieval strategy.

The goal is to provide grounded, session-specific context to the LLM when classifying interventions and extracting topics.

This design is intentionally:
- Local-only
- Deterministic
- Session-scoped (v0)
- Compatible with MCP tool boundaries

## See also

- `architecture.md`
- `classification-rubric.md`
- `mcp-tools.md`

---

# 1) RAG Scope (v0)

In version 0:

- Retrieval is limited to the SAME session as the intervention.
- No cross-session retrieval.
- No external documents.
- No external knowledge base.

This ensures:
- High grounding quality
- Predictable behavior
- Easier debugging

---

# 2) What Gets Indexed

For each session, we build a retrieval index consisting of structured chunks.

Chunk types (v0):

1) session_notes
2) debate_context
3) agenda_or_topic_candidate

---

## 2.1 Session Notes Chunk

Source:
- `initial_notes` field from stenogram JSON

Structure:

Chunk {
    chunk_id
    session_id
    type: "session_notes"
    text
    embedding
}

Purpose:
- Provide high-level context about session purpose
- Anchor relevance decisions

Rule:
- Always create exactly one session_notes chunk per session (if text exists).

---

## 2.2 Debate Context Chunks

Source:
- Speech bodies inside the session

Strategy:

For each intervention:
- Create contextual chunks representing nearby speeches

Two possible strategies (choose one for v0):

Option A (simpler):
- One chunk per speech (after merging text/text2/text3)

Option B (better grounding):
- Sliding window of N speeches (e.g., 3–5 speeches per chunk)

Recommendation for learning:
Start with Option A.
Later upgrade to sliding window and compare behavior.

Chunk structure:

Chunk {
    chunk_id
    session_id
    type: "debate_context"
    text
    embedding
}

---

## 2.3 Agenda / Topic Candidate Chunks (Optional v0+)

Derived from:
- Frequent terms in session
- Early speeches
- Title metadata (if available)

These chunks represent inferred high-level themes.

They help the LLM anchor to session-level subjects.

Can be added in v1 after baseline retrieval works.

---

# 3) Chunk Creation Rules

- All chunks must:
  - Belong to exactly one session_id
  - Be deterministic (same input → same chunk IDs)
- chunk_id format suggestion:
  "{session_id}:{type}:{sequence_number}"

Example:
  "8845:debate_context:17"

- Text must be normalized consistently (same rules as intervention text merge).

---

# 4) Embeddings

Each chunk receives an embedding vector.

Embedding constraints:

- Deterministic model choice (same model across runs)
- No dynamic temperature or randomness
- Stored persistently in vector store

Storage:
- Local persistent vector index (disk-backed)

Important:
Embeddings are part of internal state, not frontend contract.

---

# 5) Retrieval Strategy

Retrieval is performed via MCP tool: `retrieve_context`.

Given an intervention:

1) Embed the intervention text.
2) Search vector index limited to:
   - same session_id
3) Return top_k chunks ordered by:
   - descending similarity score
   - tie-breaker: chunk_id ascending

Output includes:
- chunk_id
- type
- text
- score

---

# 6) Context Passed to LLM

When classifying an intervention, the LLM receives:

- Intervention text
- Retrieved chunks (ordered)
- Explicit classification rubric

Important:
The model must not invent session context.
All context must come from retrieved chunks.

---

# 7) Evidence Linking

When the LLM classifies an intervention, it must:

- Provide evidence_chunk_ids[]
- Reference only chunk IDs returned by retrieval

Validation rule:
All evidence_chunk_ids must:
- Exist
- Belong to the same session
- Be part of the retrieved set

---

# 8) Performance Considerations

v0 (implemented):
- Index is persisted per session under `state/vectorstore/{session_id}.index`
- Rebuild skipped if chunk content hash matches (incremental by default)
- Index rebuilt only when a session is reprocessed or state is reset

v1+:
- Incremental embedding of new sessions only (already works via hash check)
- Batch embedding for faster initial index build across many sessions

---

# 9) Known Tradeoffs

Limiting retrieval to same session:

Pros:
- Prevents hallucinated cross-session context
- Keeps model grounded
- Easier debugging

Cons:
- Cannot capture broader thematic continuity across sessions

This is acceptable for v0.

---

# 10) Future Extensions

Possible upgrades after stable baseline:

- Cross-session retrieval
- Party-specific thematic retrieval
- Temporal weighting (recent speeches higher weight)
- Hybrid search (keyword + vector)
- Re-ranking layer

---

# 11) v0 Implementation

## Embedding model

- `paraphrase-multilingual-MiniLM-L12-v2` (via `sentence-transformers`)
- Chosen for: multilingual support (Romanian), small size, fast inference
- Embeddings are L2-normalised; inner product = cosine similarity

## Vector index

- One FAISS `IndexFlatIP` per session (exact search, no approximation)
- Stored at `state/vectorstore/{session_id}.index`
- Companion metadata at `state/vectorstore/{session_id}.meta.json`
- Metadata includes chunk texts, types, and source speech indexes

## Retrieval strategy (hybrid)

For each intervention, chunks are selected in priority order:

1. **session_notes** (always included) — provides session framing
2. **Neighbors** (always included) — 3 speech chunks before and after the
   intervention's speech index (6 slots total) — provides local debate context
3. **Similarity** (fill remaining slots up to top_k=10) — cosine similarity
   search for semantic context (typically 3 slots)

Budget: 1 session_notes + 6 neighbors + 3 similarity = 10 chunks ≈ 2,500 tokens.

Rationale: parliamentary debate is highly sequential — a speaker almost always
responds to what was said in the preceding 3–5 speeches. Using 3 neighbors on
each side captures that context reliably. The 3 remaining similarity slots can
still surface thematically relevant content from earlier in the session (e.g.
the rapporteur's opening speech on the bill being debated).

## Why not send the full session

Sessions average ~18,000 tokens (median), up to ~65,000 tokens at P90.
With ~130 speeches per session, sending the full session per intervention
would cost tens to hundreds of millions of tokens per pipeline run.
RAG selects the ~8 most relevant chunks (~1,000–2,000 tokens), eliminating
noise while preserving the critical context.

## Files

- `scripts/rag_store.py` — embedding, index build/load/query
- `scripts/inspect_retrieval.py` — CLI tool to inspect retrieval for any intervention

---

End of RAG indexing document.
