# votez-activity-analyzer
AI-powered analysis of Romanian parliamentary activity — classifies interventions as constructive / neutral / non-constructive and extracts debate topics from stenograms.

---

## Quick start

A five-minute path from a fresh clone to a fully populated `outputs/` folder. See the detailed sections further down for anything you want to understand deeper.

### 1. Install

```bash
python3 --version                     # must be 3.9+
pip3 install -r requirements.txt
brew install tesseract                # optional — needed for law-initiator OCR
brew install ollama                   # optional — needed if you want the free local LLM
ollama pull qwen3:14b                 # one-time model download for the local LLM
```

The Python dependencies cover the whole pipeline (pipeline + crawler + exports). `tesseract` and `ollama` are only required if you plan to run the law-initiator hydration or the local LLM classifier respectively. See [Prerequisites](#prerequisites) for details.

### 2. Place the input data

Two kinds of input files are expected under `input/`:

```text
input/
├── toti_deputatii.json      # Chamber of Deputies registry (used by the crawler + speaker resolution)
├── toti_senatorii.json      # Senate registry (same shape)
└── stenograme/              # one JSON file per parliamentary session (add more over time)
    ├── stenograma_2025-02-03_1.json
    ├── stenograma_2025-02-03_2.json
    └── ...
```

Each `stenograma_*.json` must follow this minimal shape:

```json
{
  "source_url": "https://www.cdep.ro/pls/steno/steno2015.stenograma?ids=8846&idl=1",
  "session_id": "8846",
  "stenograma_date": "2025-02-03",
  "initial_notes": "Şedinţa a început la ora 16.54. Lucrările au fost conduse de ...",
  "speeches": [
    { "speaker": "Domnul Vasile-Daniel Suciu", "text": "..." },
    { "speaker": "Domnul George-Nicolae Simion", "text": "..." }
  ]
}
```

Required fields: `source_url`, `session_id`, `stenograma_date` (`YYYY-MM-DD`), `speeches[].speaker`, `speeches[].text`. Optional: `initial_notes`, `speeches[].text2`, `speeches[].text3`. See [`input-data.md`](input-data.md) for the full contract including speaker-name cleaning rules.

### 3. Run the recommended sequence

**Step A — start Ollama (only if you're using the local LLM).** Keep it running in its own terminal:

```bash
ollama serve
```

**Step B — run the main pipeline (topics + intervention classification + exports).** Only new/changed stenograms are processed, so this is safe to rerun after adding files:

```bash
python3 scripts/run_pipeline.py --analyzer-mode llm
# or, for paid OpenAI classification:
# export OPENAI_API_KEY=sk-...
# python3 scripts/run_pipeline.py --analyzer-mode llm --llm-provider openai --llm-model gpt-4o-mini
```

Writes to `outputs/members/`, `outputs/parties/`, `outputs/topics/`, `outputs/session_topics/`.

**Step C — productivity metrics (word- and letter-weighted):**

```bash
python3 scripts/export_effectiveness.py
```

Writes to `outputs/productivity/`.

**Step D — crawl the CDEP deputy activity pages, hydrate law initiators, and export activity snapshots.** Single command, end-to-end:

```bash
export OPENAI_API_KEY=sk-...   # optional — enables the handwriting-signature vision fallback
python3 scripts/crawl_deputy_activity.py \
    --update-existing \
    --hydrate-law-initiators \
    --export-activity
```

Writes to `state/state.sqlite` (crawler tables) and `outputs/activity/members/` + `outputs/activity/parties/`.

### 4. Where everything lands

```text
state/state.sqlite                      # unified DB — interventions, crawler data, runs
outputs/members/                        # per-member interventions with labels & topics
outputs/parties/                        # per-party interventions
outputs/session_topics/                 # per-session derived topics
outputs/topics/                         # per-topic roll-ups
outputs/productivity/                   # word/letter productivity metrics (members + parties + total)
outputs/activity/members/               # per-member crawler activity snapshots (motions, Q&I, laws, ...)
outputs/activity/parties/               # per-party aggregations (initiated laws, majority support, ...)
```

### 5. Iterating

- Adding new stenograms? Drop them into `input/stenograme/` and rerun Step B — processed stenograms are tracked in the DB and skipped automatically.
- Tweaking the snapshot shape? `python3 scripts/crawl_deputy_activity.py --only-export-activity` rebuilds `outputs/activity/` from the current DB without recrawling.
- First-time smoke test? `python3 scripts/run_pipeline.py --analyzer-mode llm --llm-sessions-limit 3 --llm-speech-limit 10` runs a tiny slice end-to-end.

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

Installs: `sentence-transformers`, `faiss-cpu`, `openai` (used as the HTTP client for both OpenAI and Ollama), plus PDF/OCR helper libraries used by the deputy activity crawler.

### 3. Tesseract OCR (optional — required for law initiator extraction)

The deputy activity crawler can OCR `Expunerea de motive` PDFs to identify the deputies who actually authored/worked on a law initiative:

```bash
brew install tesseract
```

For best Romanian OCR, install the Romanian language data too if your Tesseract package does not include it. The crawler defaults to `ron+eng` and falls back to `eng`.

### 4. Ollama (free local LLM — required for LLM mode)

Download from [ollama.com](https://ollama.com) or via Homebrew:

```bash
brew install ollama
```

Pull the default local model used by the pipeline:

```bash
ollama pull qwen3:14b
```

> **Why `qwen3:14b`?** Step 3.1 upgrades the default local model from the old 7B baseline to a stronger 14B model. The pipeline now requests a 32k runtime context for known large-context local models, so `qwen3:14b` can still receive full-session prompts without requiring a separate wrapper model name.
>
> **Legacy model** (`llama3.1:8b-8k`) still works if already set up — pass
> `--llm-model llama3.1:8b-8k` to use it. The pipeline will automatically fall back to
> map-reduce for any model with `num_ctx < 32768`.
>
> **Optional wrappers for benchmarking:** the repo also includes `Modelfile-qwen2.5-14b-32k` and `Modelfile-qwen3-14b-32k` if you want explicit 32k Ollama aliases for repeatable local comparisons.

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

Use a specific model:

```bash
python3 scripts/run_pipeline.py --analyzer-mode llm --llm-provider ollama --llm-model qwen3:14b
python3 scripts/run_pipeline.py --analyzer-mode llm --llm-provider openai --llm-model gpt-4o-mini
```

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

### Benchmark models on the gold set

```bash
python3 scripts/benchmark_local_models.py
```

This creates an isolated DB copy per model under `state/model_benchmarks/`, reruns LLM topic extraction plus intervention classification on the gold-standard sessions only, and writes both per-model `benchmark_report.json` files and an aggregated `summary.json`. The JSON format is the same for Ollama and OpenAI runs so you can compare results side by side; benchmark summaries now include a `provider` field. `summary.json` keeps a run history under `runs`, with each entry stamped by `run_started_at`, while the top-level `results` still reflects the latest run for convenience.

If a gold session is missing from `state/state.sqlite` but its stenogram file exists in `input/stenograme/`, the benchmark now auto-imports that session into a prepared temporary source DB before evaluation so you can benchmark against the full gold set without manually rebuilding the main DB first.

Benchmark one specific local model:

```bash
python3 scripts/benchmark_local_models.py --models qwen3:14b
```

Benchmark GPT models with the same harness:

```bash
python3 scripts/benchmark_local_models.py --provider openai --models gpt-5.4-mini gpt-4o-mini --benchmark-scope limited
```

Run a more thorough OpenAI benchmark across all medium/hard gold sessions:

```bash
python3 scripts/benchmark_local_models.py --provider openai --models gpt-5.4-mini
```

Mix Ollama and OpenAI models in one run:

```bash
python3 scripts/benchmark_local_models.py --models ollama/qwen3:14b openai/gpt-5.4-mini --benchmark-scope limited
```

For a shorter smoke test:

```bash
python3 scripts/benchmark_local_models.py --models qwen3:14b qwen2.5:14b-32k --benchmark-scope limited
```

`--benchmark-scope limited` is the cheaper preset: it defaults to the first 3 gold sessions and evaluates only medium/hard gold speeches. You can still override `--session-limit` manually if you want a different cap.

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
- **Single-pass mode** (default with `qwen3:14b`): sends the entire session to the LLM in one call for higher coverage and quality
- **Map-reduce fallback**: used automatically for small-context models (`llama3.1:8b-8k`) or sessions > 80k chars — splits into windows, extracts bullet lists per window, then merges into structured topics
- Skips sessions already processed by **any** LLM (any `topics_source LIKE 'llm_v1:%'`) by default
- Stored with `topics_source='llm_v1:{model}'` (e.g. `llm_v1:qwen3:14b`) for auditability

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

OpenAI runs now default to Flex Processing (`service_tier=flex`) for lower-cost background-style workloads. To force standard processing instead, set:

```bash
export OPENAI_SERVICE_TIER=auto
```

---

## Inspect & Debug

```bash
# Classify a single session (all its interventions) directly
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id>
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id> --provider ollama --model qwen3:14b

# Choose architecture explicitly (default: `auto`, resolved from the model profile)
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id> --pipeline-architecture three_layer
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id> --pipeline-architecture one_pass
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id> --pipeline-architecture auto

# Build prompts for a single session without calling the LLM
python3 scripts/llm_agent.py --session-id <session_id> --run-id <run_id> --build-prompts

# Inspect RAG retrieval for an intervention
python3 scripts/inspect_retrieval.py --session-id <id> --speech-index <n>

# Exercise all MCP tools interactively
python3 scripts/demo_mcp.py --session-id <id> --speech-index <n>
```

---

## Productivity export (`scripts/export_effectiveness.py`)

Generate word- and letter-weighted productivity metrics from the current DB state:

```bash
python3 scripts/export_effectiveness.py
```

This reads only interventions already processed by the LLM (`relevance_source='llm_agent_v1'`) and writes separate JSON artifacts under `outputs/productivity/`:

```text
outputs/productivity/productivity_total.json
outputs/productivity/members/productivity_index.json
outputs/productivity/members/productivity_{member_id}_{name_slug}.json
outputs/productivity/parties/productivity_index.json
outputs/productivity/parties/productivity_{party_id}.json
```

Productivity is computed twice, and the generic `productivity_ratio` / `productivity_pct` fields use the letter-based result:

```text
word_productivity_pct   = constructive_word_count / total_word_count * 100
letter_productivity_pct = constructive_letter_count / total_letter_count * 100
productivity_pct        = letter_productivity_pct
counterproductiveness_pct = non_constructive_letter_count / total_letter_count * 100
```

The totals include `parliament_members`, `non_parliament_speakers`, and `all_llm_processed_speeches`. Member and party outputs include `member_id`, `name`, `party_id`, `party_name`, processed intervention counts, constructive and non-constructive counts, total/constructive/non-constructive word counts, total/constructive/non-constructive letter counts, productivity percentages, and the letter-based `counterproductiveness_pct`.

Use a different DB or output directory:

```bash
python3 scripts/export_effectiveness.py --db-path state/state.sqlite --output-dir outputs/productivity
```

---

## Other commands

```bash
python3 scripts/init_db.py                                         # initialize DB only
python3 scripts/reset_state.py                                     # reset state for a clean rerun
python3 scripts/select_stenograms.py                               # show files that would be selected
python3 scripts/export_outputs.py                                  # re-export outputs without re-analyzing
python3 scripts/export_effectiveness.py                            # export word/letter productivity metrics
python3 scripts/crawl_deputy_activity.py --limit 5                 # crawl CDEP deputy activity links into SQLite
python3 scripts/crawl_deputy_activity.py --only-export-activity    # re-export outputs/activity/ snapshots from the crawler DB
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
| `scripts/export_effectiveness.py` | Export word/letter productivity metrics by member, party, and total |
| `scripts/crawl_deputy_activity.py` | Crawl deputy profile activity pages from CDEP and store laws, decision projects, questions/interpellations, motions, and written political declarations in SQLite; also drives law-initiator OCR/vision hydration and the activity JSON export |
| `scripts/export_activity.py` | Serialize the crawler DB into per-member and per-party activity JSON snapshots under `outputs/activity/` (invoked from `crawl_deputy_activity.py` via `--export-activity`) |
| `scripts/inspect_retrieval.py` | Inspect retrieved chunks for any intervention |
| `scripts/demo_mcp.py` | Exercise all MCP tools end-to-end |

---

## Crawl deputy activity from CDEP

**Recommended usage** — one command that does the full pipeline end-to-end: crawl every deputy's activity pages, OCR/vision-hydrate law initiators, and refresh the per-member + per-party JSON snapshots under `outputs/activity/`:

```bash
export OPENAI_API_KEY=sk-...   # optional, enables the handwriting-signature vision fallback
python3 scripts/crawl_deputy_activity.py \
    --update-existing \
    --hydrate-law-initiators \
    --export-activity
```

Rerun it whenever you want a full refresh. It is incremental where it can be (stored rows are reused, OCR is skipped for already-hydrated laws) and fully idempotent for the activity export step.

For a faster iteration loop, work on a small subset first with `--limit` or `--member-id`, or just rebuild snapshots from the current DB with `--only-export-activity` (see flags below).

---

```bash
python3 scripts/crawl_deputy_activity.py
```

The crawler reads `input/toti_deputatii.json`, visits each deputy `profile_url`, follows the CDEP activity links for `Propuneri legislative iniţiate`, `Proiecte de hotarâre iniţiate`, `Întrebari şi interpelări`, `Moţiuni`, and `Declaraţii politice depuse în scris`, then writes the parsed records to `state/state.sqlite`.

Useful flags:

```bash
python3 scripts/crawl_deputy_activity.py --limit 5
python3 scripts/crawl_deputy_activity.py --member-id deputat_1
python3 scripts/crawl_deputy_activity.py --member-id 1
python3 scripts/crawl_deputy_activity.py --update-existing --limit 5
python3 scripts/crawl_deputy_activity.py --update-existing --hydrate-law-initiators --limit 5
python3 scripts/crawl_deputy_activity.py --only-hydrate-law-initiators
python3 scripts/crawl_deputy_activity.py --export-activity --limit 5
python3 scripts/crawl_deputy_activity.py --only-export-activity
python3 scripts/crawl_deputy_activity.py --dry-run --limit 1
```

For every processed deputy, the script logs the profile count, stored record count, association count, and source URL for each activity type. By default, it inserts only new crawler data: existing rows in crawler-owned tables are left unchanged, while new member associations are still added. Pass `--update-existing` when you want parsed CDEP pages to refresh existing crawler rows.

Pass `--hydrate-law-initiators` to run a second phase after the normal crawl finishes: the script reads stored laws from `dep_act_laws`, fetches each law's `Expunerea de motive` PDF through `source_url`, OCRs it locally with Tesseract, extracts the `Iniţiatori` section, and marks matching deputies in `dep_act_member_laws.is_initiator`. This is intentionally optional because scanned PDF OCR is slower than the normal crawl.

Pass `--only-hydrate-law-initiators` to skip the crawl phase and retry only stored laws that do not yet have any `is_initiator = 1` association in `dep_act_member_laws`. OCR progress logs include the `dep_act_laws.source_url` law page link for debugging no-match cases.

The crawler writes only its own tables: `dep_act_member_activity_crawl`, `dep_act_laws`, `dep_act_member_laws`, `dep_act_decision_projects`, `dep_act_member_decision_projects`, `dep_act_questions_interpellations`, `dep_act_motions`, `dep_act_member_motions`, and `dep_act_political_declarations`. It validates that targeted deputies already exist in `members`, but it never inserts or updates `members`, interventions, runs, outputs, or other pipeline tables.

Laws, decision projects, and motions can be associated with several deputies, so the entity tables are deduplicated and the `dep_act_member_*` tables store the many-to-many associations.

---

## Activity export from crawler DB

Pass `--export-activity` to the crawler to serialize the crawler DB into per-member and per-party JSON snapshots. The export runs at the end of the crawl and/or law-initiator hydration phase. Use `--only-export-activity` to skip crawling and OCR entirely and just rebuild the snapshots from the current DB — useful when iterating on the snapshot shape or after a fresh hydration run.

```bash
# crawl all deputies + export at the end
python3 scripts/crawl_deputy_activity.py --export-activity

# crawl a few, OCR their laws, then export
python3 scripts/crawl_deputy_activity.py --update-existing --hydrate-law-initiators --export-activity --limit 20

# re-export only (no crawl, no OCR) using the current DB
python3 scripts/crawl_deputy_activity.py --only-export-activity

# export into a custom root directory
python3 scripts/crawl_deputy_activity.py --only-export-activity --activity-output-dir /tmp/activity
```

Output layout (written under `outputs/activity/` by default):

```text
outputs/activity/members/activity_{member_id}_{name_slug}.json
outputs/activity/parties/activity_{party_slug}.json
```

Each **member** file contains the member's identity (`member_id`, `name`, `chamber`, `party_id`, `party_name`, `profile_url`) plus five activity blocks:

- `motions[]` — each motion's stored columns, plus `co_supporting_parties: [{party_id, party_name, members_count}]`. The subject member is excluded from his own party's count (so the number reflects "how many **other** members of that party co-supported").
- `questions_and_interpellations[]` — every Q&I row for the member, including `identifier`, `recipient`, `text`, `source_url`, and the raw listing `columns`.
- `political_declarations[]` — every written political declaration, including `title`, `full_text`, `text_url`, and the raw listing `columns`.
- `decision_projects[]` — each project, enriched with `collaborating_parties: [{party_id, party_name, members_count, members: [{member_id, name}]}]` grouping every collaborator by party.
- `laws[]` — every law the member is linked to via `dep_act_member_laws`. Each entry adds `is_initiator` (for this member), `is_adopted` (from `adopted_law_identifier`), `initiator_parties` (members with `is_initiator=1` grouped by party), and `supporter_parties` (**all** linked members grouped by party — initiators are a subset).

Each **party** file aggregates its members:

- `members_count` and `majority_threshold = min(10, ceil(members_count / 2))`.
- `laws_initiated[]` — any party member was an initiator. Includes `party_initiators: [{member_id, name}]`, full `initiator_parties` / `supporter_parties` roll-ups, and `is_adopted`.
- `laws_majority_supported_only[]` — the party had **no** initiator on this law, but ≥ `majority_threshold` of its members appear as supporters. No overlap with `laws_initiated`.
- `questions_and_interpellations[]` — all Q&I from every party member, each tagged with `asked_by: {member_id, name}`.
- `motions_majority_supported[]` — motions where ≥ `majority_threshold` of the party's members are supporters, including `all_supporting_parties` counts.

Each export run **wipes** all `activity_*.json` files in the two target folders before writing new ones, so deleted members/parties never linger. Other files in those folders are preserved.

Notes:

- The DB has no `parties` table — `members.party_id` doubles as the party's display label (e.g. `PSD`, `AUR`, `Neafiliaţi`). `party_id` and `party_name` are emitted with the same value so the shape is forward-compatible if a party mapping is added later.
- Members with `party_id IS NULL` are exported under the sentinel label `"Neafiliat (no party)"`. They still get a member file, but they are not counted toward any party file.

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
| `dep_act_member_activity_crawl` | Last CDEP crawl status, profile counts, source links, and stored counts per deputy |
| `dep_act_laws` | Deduplicated legislative proposal/law records from deputy activity pages, including adopted law identifiers such as `Lege 233/2025`, `Expunerea de motive` PDF URLs, OCR initiator text, and initiator parse errors |
| `dep_act_member_laws` | Deputy-to-law associations, including `is_initiator` when the deputy is matched in the OCR-parsed `Iniţiatori` section |
| `dep_act_decision_projects` | Deduplicated CDEP decision project records |
| `dep_act_member_decision_projects` | Deputy-to-decision-project associations |
| `dep_act_questions_interpellations` | CDEP question/interpellation records by deputy, including identifier, source link, and cleaned text |
| `dep_act_motions` | Deduplicated CDEP motion records |
| `dep_act_member_motions` | Deputy-to-motion associations |
| `dep_act_political_declarations` | Written political declarations by deputy, including title, detail URL, text URL, and full text |

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
