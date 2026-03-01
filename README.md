# votez-activity-analyzer
AI-powered analysis of Romanian parliamentary activity — classifies interventions as constructive / neutral / non-constructive and extracts debate topics from stenograms.

---

## Prerequisites

### 1. Python 3.9+

```bash
python3 --version
```

### 2. Python dependencies

```bash
pip3 install -r requirements.txt
```

Installs: `sentence-transformers`, `faiss-cpu`, `openai` (used as the HTTP client for both OpenAI and Ollama).

### 3. Ollama (free local LLM — required for LLM mode)

Download from [ollama.com](https://ollama.com) or via Homebrew:

```bash
brew install ollama
```

Pull the model (one-time, ~4.7 GB) and create the 32k-context variant used by the pipeline:

```bash
ollama pull qwen2.5:7b

ollama create qwen2.5:7b-32k -f - << 'EOF'
FROM qwen2.5:7b
PARAMETER num_ctx 32768
EOF
```

> **Why `qwen2.5:7b-32k`?** Ollama's default runtime context is 4096 tokens. The `-32k`
> variant bakes in `num_ctx=32768`, giving the model enough context to receive an entire
> parliamentary session (~15–20k tokens) in a **single call** for topic extraction — no
> map-reduce splitting needed. No extra disk space is used; it references the same weights
> with a different manifest.
>
> **Legacy model** (`llama3.1:8b-8k`) still works if already set up — pass
> `--llm-model llama3.1:8b-8k` to use it. The pipeline will automatically fall back to
> map-reduce for any model with `num_ctx < 32768`.

---

## Default commands

### Run the full pipeline (LLM classification)

Step 1 — start Ollama in a separate terminal and keep it running:

```bash
ollama serve
```

Step 2 — run the pipeline:

```bash
python3 scripts/run_pipeline.py --analyzer-mode llm
```

This processes only new/changed stenograms, classifies every intervention via LLM, and exports results to `outputs/`.

> **First time?** Test on a small batch before running the full set:
> ```bash
> # Run a single stenogram end-to-end (topics + interventions) — fastest feedback loop
> python3 scripts/run_pipeline.py --analyzer-mode llm --stenogram input/stenograme/stenograma_2025-02-19_1.json
>
> # Extract topics for 3 sessions + classify 10 speeches — good end-to-end smoke test
> python3 scripts/run_pipeline.py --analyzer-mode llm --llm-sessions-limit 3 --llm-speech-limit 10
>
> # Extract topics only (no intervention classification)
> python3 scripts/run_pipeline.py --analyzer-mode llm --llm-sessions-limit 3 --llm-speech-limit 0
>
> # Classify speeches only (sessions already have LLM topics)
> python3 scripts/run_pipeline.py --analyzer-mode llm --llm-speech-limit 10
> ```
>
> `--stenogram` resets and fully reprocesses that session's topics and interventions on every run — useful when iterating on prompts.
>
> | Flag | Limits | Default |
> |------|--------|---------|
> | `--llm-sessions-limit N` | Session topic extraction (step 3b) | 0 = all |
> | `--llm-speech-limit N` | Intervention classification (step 3c) | 0 = all |

### Optimise prompts externally — without calling the local LLM

Use this two-step workflow when you want to test a prompt against a more capable external model (e.g. ChatGPT, Gemini) before committing to a full pipeline run.

**Step 1 — generate prompt files (no LLM call)**

```bash
python3 scripts/run_pipeline.py --analyzer-mode llm --build-prompts
```

This runs the full preparation logic (chunking, session header, topic grounding context) for **every session in the DB** and writes one `.txt` file per LLM call to `state/generated_prompts/`. **No LLM is called and nothing is written to the DB.** The directory is wiped and fully refreshed on every `--build-prompts` run so you always get a clean snapshot.

Prompt files are named:
```
session_topics_{date}_{session_id}_draft_{model}_{label}.txt
interventions_{date}_{session_id}_draft_{model}_{label}.txt
```

Each file contains a `=== METADATA ===` header, the full `=== SYSTEM PROMPT ===`, and the `=== USER MESSAGE ===`.

**Step 2 — send to an external model, place the response in `state/external_prompts_output/`**

Create a file with the **exact same name** as the prompt file (only the directory differs) and put the model's raw JSON response inside — just the JSON, no extra wrapper:

```
# For session topics (single-pass or reduce):
{"topics": [{"label": "...", "description": "...", "law_id": null}, ...], "session_summary": "..."}

# For interventions:
{"results": [{"speech_index": 1, "constructiveness_label": "constructive", "topics": ["..."], "confidence": 0.9, "reasoning": "..."}, ...]}
```

**Step 3 — ingest the responses and export**

```bash
python3 scripts/run_pipeline.py --analyzer-mode llm --ingest-external-outputs
```

This reads every unprocessed file from `state/external_prompts_output/`, validates and stores each result to the DB, then exports. A `.done` sidecar is created next to each ingested file so it is never double-processed.

> **Tip:** You can also target a single session directly via the sub-scripts:
> ```bash
> # Build prompts for one session only (output goes to state/generated_prompts/)
> python3 scripts/llm_session_topics.py --session-id 8856 --run-id <run_id> --build-prompts
> python3 scripts/llm_agent.py --session-id 8856 --run-id <run_id> --build-prompts
>
> # Ingest responses (reads from state/external_prompts_output/)
> python3 scripts/llm_session_topics.py --run-id <run_id> --ingest-external-outputs
> python3 scripts/llm_agent.py --run-id <run_id> --ingest-external-outputs
> ```

---

### Run baseline only (no LLM, no Ollama needed)

```bash
python3 scripts/run_pipeline.py
```

Uses keyword overlap to assign labels. Fast, deterministic, no external dependencies beyond `requirements.txt`.

### Dry run (see which files would be processed, no changes made)

```bash
python3 scripts/run_pipeline.py --dry-run
```

---

## What the pipeline does

### Baseline pass (always runs)
- Creates `state/state.sqlite` if missing and initializes schema
- Selects only new/changed stenograms from `input/stenograme/`
- Normalizes speakers and resolves them to known members
- Persists raw interventions to DB
- Assigns a deterministic `constructiveness_label` via keyword overlap
- Builds a per-session FAISS vector index (sentence-transformers embeddings)
- Exports frontend JSON artifacts to `outputs/`
- Marks processed stenograms in DB

### LLM pass (`--analyzer-mode llm`, runs after baseline — two sub-steps)

**Step 3b — Session topic extraction** (`llm_session_topics.py`):
- **Single-pass mode** (default with `qwen2.5:7b-32k`): sends the entire session to the LLM in one call for higher coverage and quality
- **Map-reduce fallback**: used automatically for small-context models (`llama3.1:8b-8k`) or sessions > 80k chars — splits into windows, extracts bullet lists per window, then merges into structured topics
- Skips sessions already processed by **any** LLM (any `topics_source LIKE 'llm_v1:%'`) by default
- Stored with `topics_source='llm_v1:{model}'` (e.g. `llm_v1:qwen2.5:7b-32k`) for auditability

**Step 3c — Intervention classification** (`llm_agent.py`):
- Sends **all speeches of a session in one call** so the model has full conversational context
- If a session is too large for the context window, speeches are split into greedy consecutive batches (never cutting a speech in half)
- Each batch includes: session date, initial notes, LLM-derived session topics (grounding context), full speeches with speaker names and indices
- LLM returns one classification object per speech: `constructiveness_label`, `topics`, `confidence`, `reasoning`
- Stores results via MCP; source stamped as `llm_agent_v1` for auditability
- Baseline labels (`constructiveness_baseline_v1`) are **never overwritten** by a re-run of the baseline — only LLM can upgrade them

Run session topic extraction alone (useful for debugging):

```bash
python3 scripts/llm_session_topics.py --session-id 8846 --run-id <run_id>
```

Force re-extraction of session topics (e.g. after switching models):

```bash
python3 scripts/run_pipeline.py --analyzer-mode llm --reprocess-session-topics
# or with a specific model:
python3 scripts/run_pipeline.py --analyzer-mode llm --llm-model mistral --reprocess-session-topics
```

---

## LLM provider options

| Provider | Flag | Cost | Requirement |
|----------|------|------|-------------|
| Ollama (default) | `--llm-provider ollama` | Free | `ollama serve` + model pulled |
| OpenAI | `--llm-provider openai` | Paid (~$5–10 for full run) | `OPENAI_API_KEY` env var |

Switch to OpenAI:

```bash
export OPENAI_API_KEY=sk-...
python3 scripts/run_pipeline.py --analyzer-mode llm --llm-provider openai
```

---

## Inspect & Debug

```bash
# Classify a single session (all its interventions) directly
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id>

# Build prompts for a single session without calling the LLM
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id> --build-prompts

# Inspect RAG retrieval for an intervention
python3 scripts/inspect_retrieval.py --session-id <id> --speech-index <n>

# Exercise all MCP tools interactively
python3 scripts/demo_mcp.py --session-id <id> --speech-index <n>
```

---

## Other commands

```bash
python3 scripts/init_db.py                                         # initialize DB only
python3 scripts/reset_state.py                                     # reset state for a clean rerun
python3 scripts/select_stenograms.py                               # show files that would be selected
python3 scripts/export_outputs.py                                  # re-export outputs without re-analyzing
python3 scripts/validate_outputs.py                                # validate exported output integrity
python3 scripts/run_pipeline.py --analyzer-cmd "<your command>"    # use a custom analyzer
```

When using `--analyzer-cmd`, these env vars are injected:
- `VOTEZ_RUN_ID`
- `VOTEZ_STENOGRAM_LIST_PATH`

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Main orchestrator — incremental, handles baseline + LLM + export |
| `scripts/analyze_interventions.py` | Baseline classifier (keyword + RAG index build) |
| `scripts/llm_session_topics.py` | LLM session topic extraction — runs before intervention classification |
| `scripts/llm_agent.py` | LLM intervention classification — the main intelligence layer |
| `scripts/rag_store.py` | Vector index build and retrieval (`sentence-transformers` + FAISS) |
| `scripts/mcp_server.py` | MCP tool server (all read, RAG, and write tools) |
| `scripts/inspect_retrieval.py` | Inspect retrieved chunks for any intervention |
| `scripts/demo_mcp.py` | Exercise all MCP tools end-to-end |

---

## SQLite tables

| Table | Contents |
|-------|----------|
| `runs` | Run metadata and status |
| `processed_stenograms` | Incremental processing state |
| `members` | Resolved deputy/senator registry |
| `interventions_raw` | All parsed interventions |
| `intervention_analysis` | Labels, topics, confidence, evidence chunk IDs, source |
| `session_chunks` | RAG chunks per session |
| `session_topics` | Derived session topics (`topics_source`: `keyword_baseline_v1` or `llm_v1:{model}`) |
| `unmatched_speakers` | Speakers that could not be resolved |
| `run_outputs` | Run summary stats |

View: `interventions_enriched` — joins all of the above for easy querying.

---

## State directories

| Directory | Contents |
|-----------|----------|
| `state/run_inputs/` | Stenogram file lists written at the start of each run |
| `state/run_outputs/` | Run summary JSON written at the end of each run |
| `state/run_prompts/` | Live prompts written during normal LLM runs — one timestamped `.txt` per call, never wiped |
| `state/generated_prompts/` | Prompts generated by `--build-prompts` — wiped and refreshed on every run, covers all sessions |
| `state/external_prompts_output/` | External model responses to ingest — place files here with names matching the corresponding `generated_prompts/` files |

---

## Docs

- `goal.md` — project goal and target outcome
- `architecture.md` — system architecture and component design
- `classification-rubric.md` — constructiveness classification rules and edge cases
- `rag-indexing.md` — RAG chunking and retrieval strategy
- `mcp-tools.md` — MCP tool contract
- `output-contract.md` — frontend JSON output schema
- `input-data.md` — input file format
