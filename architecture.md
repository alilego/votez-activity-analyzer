# architecture.md
`votez-activity-analyzer`

This document describes the architecture of the application.

The system is a **local, periodically-run analyzer** that processes Romanian Parliament stenograms and exports structured datasets for frontend consumption.

A key goal of this project is hands-on learning and practical implementation of:

- **RAG (Retrieval-Augmented Generation)** for grounded classification and topic extraction
- **MCP (Model Context Protocol)** patterns for structured, auditable tool use
- clear explanations for each important architectural and implementation decision

## See also

- `goal.md`
- `input-data.md`
- `rag-indexing.md`
- `mcp-tools.md`
- `classification-rubric.md`
- `output-contract.md`

---

# 0) Tech Stack

## Language & Runtime

- **Python 3.11+**
  - Chosen for fast iteration, strong ecosystem for NLP, embeddings, and LLM orchestration.
  - Clear fit for experimentation with RAG and MCP patterns.

Execution model:
- CLI / batch job
- Runs locally
- Triggered manually or periodically (cron / scheduler later)

---

## Storage Strategy (Local-First)

### A) Filesystem (authoritative contract boundary)

Inputs:
- `input/toti_deputatii.json`
- `input/toti_senatorii.json`
- `input/stenograme/*.json`

Outputs:
- `outputs/members/interventions_index.json`
- `outputs/members/interventions_{member_id}.json`
- `outputs/parties/interventions_index.json`
- `outputs/parties/interventions_{party_id}.json`

The filesystem is:
- the source of truth for inputs
- the contract boundary for the frontend

---

### B) Persistent State Store

Use **SQLite** for local persistent state:

File example:
- `state/state.sqlite`

Stored data may include:
- intervention-level analysis results
- unmatched speakers
- run metadata and summaries
- incremental processing metadata (required)
- evaluation traces

SQLite is chosen because:
- no server required
- single file
- deterministic
- easy to inspect

Incremental requirement:
- when new files are added to `input/stenograme/`, the analyzer should process only new or changed files
- previously processed stenograms should not be reprocessed by default in periodic runs

---

### C) Local Vector Store (RAG)

RAG requires embeddings and similarity search.

The vector store will:
- persist locally on disk
- store embeddings for session chunks
- be rebuildable if needed

Location example:
- `state/vectorstore/`

No remote vector DB is used in v0.

---

# 1) High-Level Architecture

The system has three conceptual layers:

1. **Data Layer** (Input ingestion + local storage)
2. **Intelligence Layer** (RAG + LLM reasoning)
3. **Action Layer** (MCP tools for structured effects)

---

# 2) Pipeline Overview

```
Load registries + stenograms
        |
Build retrieval index (RAG store)
        |
Normalize & resolve speakers
        |
Build interventions
        |
For each intervention:
    - Retrieve context (RAG)
    - Classify + extract topics (LLM)
    - Store result via MCP tool
        |
Aggregate per member + per party
        |
Export frontend JSON artifacts
```

---

# 3) Core Components

## A) Ingest Layer

Responsibilities:
- Load deputies and senators registry files
- Load all stenogram JSON files
- Handle parsing errors safely

Outputs:
- `MemberRegistry`
- `Session[]`

---

## B) Speaker Normalization & Identity Resolution

Responsibilities:
- Remove prefixes: "Domnul", "Doamna"
- Remove text inside parentheses
- Trim whitespace
- Normalize case
- (Future) Normalize diacritics

Resolve normalized name against:
- `toti_deputatii.json`
- `toti_senatorii.json`

Output:
- Resolved speech objects with `member_id` or `unmatched`

Party affiliation is determined using the latest known registry snapshot.

---

## C) Intervention Builder

Responsibilities:
- Merge `text`, `text2`, `text3` into a single body
- Attach session metadata
- Generate deterministic `intervention_id`

Output:
- `Intervention[]` records

---

# 4) RAG Layer (Retrieval-Augmented Generation)

## Purpose

Provide grounded context for:

- Relevance classification
- Topic extraction

The model should not rely on global knowledge, but on retrieved session-specific context.

---

## D) Chunking Strategy

Chunks may include:

- Session initial notes
- Agenda-like metadata
- Nearby debate context (neighboring speeches)
- Derived session topic candidates

Chunk structure (conceptual):

Chunk {
    chunk_id
    session_id
    type
    text
    embedding
}

---

## E) Context Retrieval

Given an intervention:

- Retrieve relevant chunks
- Pass intervention text + retrieved context into LLM

Output:
- RetrievedContext[]
- With similarity scores and chunk IDs

Chunk IDs are later stored as evidence references.

---

# 5) MCP Layer (Model Context Protocol)

## Purpose

LLM must not directly:
- Write files
- Modify state
- Perform arbitrary mutations

Instead, it operates through structured tools.

Even though local, we enforce MCP-style boundaries.

---

## F) MCP Tool Server (Local)

Expose tools such as:

Read tools:
- get_session_metadata(session_id)
- get_member(member_id)
- get_intervention(intervention_id)

Write tools:
- store_intervention_analysis(intervention_id, relevance_label, topics, confidence, evidence_chunk_ids)
- append_unmatched_speaker(raw_speaker, normalized_speaker, session_id)
- write_run_summary(stats)

The tool server:
- Validates input schema
- Enforces structured outputs
- Prevents uncontrolled mutations

---

# 6) Intelligence Layer (Agent Loop)

For each intervention:

1. Retrieve context (RAG)
2. Classify relevance (relevant / neutral / non_relevant)
3. Extract topics
4. Call MCP tool to store structured result

Stored per intervention:

- relevance_label
- topics[]
- confidence (optional)
- evidence_chunk_ids[]

This enables traceability and auditability.

---

# 7) Aggregation Layer

After all interventions are processed:

Per member:
- Count per relevance label
- Categorized intervention lists
- Top 20 topics

Per party:
- Aggregate across members (based on registry snapshot)
- Top 20 topics
- Totals must equal sum of members

---

# 8) Export Layer

Responsibilities:

- Write JSON outputs strictly matching output contract
- Ensure deterministic ordering
- Avoid nondeterministic topic ranking

Outputs:
- outputs/members/interventions_index.json
- outputs/members/interventions_{member_id}.json
- outputs/parties/interventions_index.json
- outputs/parties/interventions_{party_id}.json

---

# 9) Traceability & Civic Integrity

To support credibility:

- Store evidence chunk IDs per intervention
- Store run metadata
- Optionally export low-confidence cases
- Log unmatched speakers

The system must favor explainability over opaque automation.

---

# 10) Versioning & Reproducibility

To ensure reproducibility:

- Stable ID generation
- Stable sorting
- Version stamp in output metadata (optional future addition)
- Deterministic embeddings if possible

---

# 11) Explicit Non-Goals (v0)

- No historical party tracking
- No remote database
- No live API service
- No frontend coupling inside analyzer
- No auto-retraining

---

End of architecture document.
