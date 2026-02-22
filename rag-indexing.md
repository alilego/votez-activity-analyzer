# rag-indexing.md
`votez-activity-analyzer`

This document defines the RAG (Retrieval-Augmented Generation) indexing and retrieval strategy.

The goal is to provide grounded, session-specific context to the LLM when classifying interventions and extracting topics.

This design is intentionally:
- Local-only
- Deterministic
- Session-scoped (v0)
- Compatible with MCP tool boundaries

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

v0:
- Rebuild index per run (acceptable for learning phase)

v1+:
- Persist embeddings
- Only embed new sessions
- Incremental indexing

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

# 11) Minimal v0 RAG Setup

To have a working RAG system that still teaches the core concepts:

Required:

- One chunk per speech
- One session_notes chunk
- Local embedding model
- Vector similarity search
- retrieve_context MCP tool
- Evidence linking

Anything beyond that is optimization, not requirement.

---

End of RAG indexing document.
