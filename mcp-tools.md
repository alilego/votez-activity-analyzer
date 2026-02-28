# mcp-tools.md
`votez-activity-analyzer`

This document defines the **MCP-style tool contract** used by the Analyzer Agent.

Even though the system runs locally, we enforce MCP principles:

- The LLM/agent **cannot** write files or mutate storage directly.
- The LLM/agent **can only** interact with the system through **structured tools**.
- Tools validate inputs, enforce constraints, and return deterministic results.
- All tool calls are auditable (stored in local state).

This keeps the system safe, reproducible, and easy to debug.

## See also

- `architecture.md`
- `rag-indexing.md`
- `classification-rubric.md`
- `output-contract.md`

---

## 1) Tooling model

### Roles

- **Agent (LLM)**: decides which tools to call and with what arguments.
- **Tool Server**: executes tools, validates inputs, stores results, and returns structured outputs.

### Conventions

- All tool inputs/outputs are JSON.
- All IDs are strings.
- `session_date` uses `YYYY-MM-DD`.
- All tools must return `{ "ok": true, ... }` or `{ "ok": false, "error": { ... } }`.

### Error schema

All errors follow:

```json
{
  "ok": false,
  "error": {
    "code": "STRING_ENUM",
    "message": "STRING",
    "details": { "any": "json" }
  }
}
```

Suggested `code` values:
- `VALIDATION_ERROR`
- `NOT_FOUND`
- `CONFLICT`
- `INTERNAL_ERROR`
- `RATE_LIMITED` (rare locally, but useful for future)
- `UNSUPPORTED`

---

## 2) Read tools

Read tools expose safe, deterministic access to already-ingested data.

### 2.1 `get_run_config`

Purpose:
- Give the agent the current run configuration (limits, retrieval params, etc.)

Input:
```json
{}
```

Output:
```json
{
  "ok": true,
  "config": {
    "contract_version": "v0",
    "max_topics_per_intervention": 5,
    "max_topic_length": 64,
    "constructiveness_labels": ["constructive", "neutral", "non_constructive"],
    "rag": {
      "top_k": 8,
      "min_score": 0.0
    }
  }
}
```

---

### 2.2 `get_session`

Purpose:
- Provide session metadata needed for grounding decisions.

Input:
```json
{ "session_id": "STRING" }
```

Output:
```json
{
  "ok": true,
  "session": {
    "session_id": "STRING",
    "session_date": "YYYY-MM-DD",
    "source_url": "STRING",
    "initial_notes": "STRING"
  }
}
```

---

### 2.3 `get_intervention`

Purpose:
- Fetch a canonical intervention record (what the agent classifies).

Input:
```json
{ "intervention_id": "STRING" }
```

Output:
```json
{
  "ok": true,
  "intervention": {
    "intervention_id": "STRING",
    "session_id": "STRING",
    "session_date": "YYYY-MM-DD",
    "member_id": "STRING | null",
    "raw_speaker": "STRING",
    "normalized_speaker": "STRING",
    "text": "STRING"
  }
}
```

---

### 2.4 `get_member`

Purpose:
- Fetch member info (mainly for display and party aggregation understanding).

Input:
```json
{ "member_id": "STRING" }
```

Output:
```json
{
  "ok": true,
  "member": {
    "member_id": "STRING",
    "name": "STRING",
    "chamber": "deputat | senator",
    "party_id": "STRING | null",
    "party_name": "STRING | null"
  }
}
```

Notes:
- Party values are based on the registry snapshot (“last known input”).

---

## 3) RAG tools (retrieval)

RAG is performed through tools so the agent cannot “invent context”.
The tool server owns indexing, retrieval, and scoring.

### 3.1 `retrieve_context`

Purpose:
- Retrieve the most relevant context chunks to ground classification.

Input:
```json
{
  "intervention_id": "STRING",
  "top_k": 8
}
```

Output:
```json
{
  "ok": true,
  "context": [
    {
      "chunk_id": "STRING",
      "session_id": "STRING",
      "type": "session_notes | debate_context | agenda_or_topic_candidate",
      "score": 0.0,
      "text": "STRING"
    }
  ]
}
```

Validation:
- `top_k` must be `1..50`.
- Chunks must belong to the same `session_id` as the intervention (v0 rule).

---

### 3.2 `get_chunk`

Purpose:
- Fetch a chunk by ID (useful for “evidence inspection” workflows).

Input:
```json
{ "chunk_id": "STRING" }
```

Output:
```json
{
  "ok": true,
  "chunk": {
    "chunk_id": "STRING",
    "session_id": "STRING",
    "type": "session_notes | debate_context | agenda_or_topic_candidate",
    "text": "STRING"
  }
}
```

---

## 4) Write tools (analysis results)

Write tools are the only allowed way for the agent to persist analysis.
They must be idempotent and validated.

### 4.1 `store_intervention_analysis`

Purpose:
- Store the agent’s classification + topics for an intervention.

Input:
```json
{
  "intervention_id": "STRING",
  "constructiveness_label": "constructive | neutral | non_constructive",
  "topics": ["STRING"],
  "confidence": 0.0,
  "evidence_chunk_ids": ["STRING"]
}
```

Output:
```json
{
  "ok": true,
  "stored": {
    "intervention_id": "STRING",
    "constructiveness_label": "constructive | neutral | non_constructive",
    "topics": ["STRING"],
    "confidence": 0.0,
    "evidence_chunk_ids": ["STRING"]
  }
}
```

Validation rules:
- `constructiveness_label` must be one of: `constructive`, `neutral`, `non_constructive`.
- `topics` length must be `0..max_topics_per_intervention` (from `get_run_config`).
- Each topic:
  - must be non-empty after trimming
  - length `<= max_topic_length`
- `confidence` must be in `[0.0, 1.0]`.
- `evidence_chunk_ids`:
  - may be empty ONLY if confidence is low (policy TBD); recommended non-empty.
  - all chunk IDs must exist and belong to the same session as the intervention.

Idempotency:
- Repeated calls with identical payload are no-ops returning the same stored state.
- If called with a different payload for the same `intervention_id`, behavior is:
  - either overwrite allowed (v0) OR conflict (strict mode). **Default v0: overwrite allowed with audit log.**

Audit:
- The tool server stores:
  - previous value
  - new value
  - timestamp
  - run_id

---

### 4.2 `append_unmatched_speaker`

Purpose:
- Record speaker names that could not be resolved to a member.

Input:
```json
{
  "session_id": "STRING",
  "raw_speaker": "STRING",
  "normalized_speaker": "STRING"
}
```

Output:
```json
{
  "ok": true,
  "stored": true
}
```

Idempotency:
- The same `(session_id, normalized_speaker)` should not create duplicates.

---

### 4.3 `write_run_summary`

Purpose:
- Store final run stats (useful for debugging and progress tracking).

Input:
```json
{
  "run_id": "STRING",
  "started_at": "ISO_TIMESTAMP",
  "finished_at": "ISO_TIMESTAMP",
  "stats": {
    "sessions_processed": 0,
    "interventions_total": 0,
    "interventions_classified": 0,
    "unmatched_speakers": 0
  }
}
```

Output:
```json
{
  "ok": true,
  "stored": true
}
```

---

## 5) Export tools (optional MCP boundary)

Export can be done outside the agent loop (recommended), but if you want export to be “tool-only”, define:

### 5.1 `export_outputs` (optional)

Purpose:
- Trigger export of outputs according to the output contract.

Input:
```json
{ "run_id": "STRING" }
```

Output:
```json
{
  "ok": true,
  "outputs_written": {
    "members_index": "outputs/members/interventions_index.json",
    "parties_index": "outputs/parties/interventions_index.json"
  }
}
```

Note:
- In v0, export can remain a deterministic post-processing step without involving the LLM.

---

## 6) Determinism requirements

Tools must support deterministic runs:

- `retrieve_context` returns results in deterministic order:
  1) descending score
  2) tie-breaker: `chunk_id` ascending
- `store_intervention_analysis` normalizes topics:
  - trim whitespace
  - keep order as provided (agent’s order), but remove exact duplicates
- All timestamps stored in state must not affect the final exported output ordering.

---

## 7) Minimal required tool set (v0)

To learn MCP properly while keeping scope tight, v0 requires:

Read:
- `get_run_config`
- `get_intervention`
- `get_session` (optional but recommended)

RAG:
- `retrieve_context`

Write:
- `store_intervention_analysis`
- `append_unmatched_speaker`

The rest can be added as needed.

---

## 8) Implementation

The tool contract is implemented in `scripts/mcp_server.py` as the `MCPServer` class.

All tools are dispatched through a single entry point:

```python
from mcp_server import MCPServer
with MCPServer(db_path=Path("state/state.sqlite"), run_id="run_xyz") as server:
    result = server.call("get_run_config", {})
    result = server.call("retrieve_context", {"intervention_id": "iv:abc:5"})
    result = server.call("store_intervention_analysis", {
        "intervention_id": "iv:abc:5",
        "constructiveness_label": "constructive",
        "topics": ["proces legislativ"],
        "confidence": 0.85,
        "evidence_chunk_ids": ["ch:8846:3", "ch:8846:5"],
    })
```

To exercise all tools interactively:

```bash
python3 scripts/demo_mcp.py --session-id 8846 --speech-index 10
python3 scripts/demo_mcp.py --dry-run   # skip the write step
```

---

End of MCP tools contract.
