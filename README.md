# votez-activity-analyzer
AI-powered analysis of Romanian parliamentary activity — classifies interventions as constructive / neutral / non-constructive and extracts debate topics from stenograms.

---

## Prerequisites

### 1. Python 3.9+ and dependencies

```bash
python3 --version                     # must be 3.9+
pip3 install -r requirements.txt
```

Installs: `sentence-transformers`, `faiss-cpu`, `openai` (used as the HTTP client for both OpenAI and Ollama), plus PDF/OCR helper libraries used by the deputy activity crawler.

### 2. LLM provider — choose one (or both)

| | Local LLM (Ollama) | OpenAI API |
|---|---|---|
| **Cost** | Free | ~$5–10 for a full run |
| **Requires** | `ollama serve` running + GPU with ~10 GB VRAM | `OPENAI_API_KEY` env var |
| **Default model** | `qwen3:14b` | `gpt-5-nano` |
| **Flag** | `--llm-provider ollama` (default) | `--llm-provider openai` |

**Option A — Local LLM (Ollama, free):**

```bash
brew install ollama
ollama pull qwen3:14b                 # one-time model download (~9 GB)
```

> **Why `qwen3:14b`?** The pipeline requests a 32k runtime context for known large-context local models, so `qwen3:14b` can receive full-session prompts without requiring a separate wrapper model name.
>
> **Legacy model** (`llama3.1:8b-8k`) still works if already set up — pass
> `--llm-model llama3.1:8b-8k` to use it. The pipeline will automatically fall back to
> map-reduce for any model with `num_ctx < 32768`.
>
> **Optional wrappers for benchmarking:** the repo also includes `Modelfile-qwen2.5-14b-32k` and `Modelfile-qwen3-14b-32k` if you want explicit 32k Ollama aliases for repeatable local comparisons.

**Option B — OpenAI API (remote, paid):**

```bash
export OPENAI_API_KEY=sk-...          # set your API key
```

OpenAI runs default to Flex Processing (`service_tier=flex`) for lower-cost background-style workloads. To force standard processing instead, set `export OPENAI_SERVICE_TIER=auto`.

### 3. Stenogram scraper (optional — fetches stenograms from cdep.ro)

The [votez-scraper](../votez-scraper) project scrapes parliamentary stenograms from cdep.ro and writes JSON files that this pipeline consumes. Clone it as a sibling directory:

```bash
# from the parent directory of votez-activity-analyzer
git clone <votez-scraper-repo-url> votez-scraper
cd votez-scraper
pip3 install -r requirements.txt      # scrapy + lxml
```

The integrated `full_update.py` script (see [Two main flows](#two-main-flows)) expects the scraper at `../votez-scraper` by default. You can override this with `--scraper-dir`.

If you already have stenogram JSON files, you can skip the scraper and place them directly under `input/stenograme/`.

The script also deploys all results to `../votez-frontend/` by default:
- Analysis outputs (members, parties, topics, productivity, activity) → `data/activity_analizer/`
- Scraper registry files (deputies/senators by party, circumscription) → `lib/`

The [votez-frontend](../votez-frontend) project should be cloned as a sibling directory. Override with `--frontend-dir` or skip with `--skip-deploy`.

### 4. Input data

Two kinds of input files are expected under `input/`:

```text
input/
├── toti_deputatii.json      # Chamber of Deputies registry (already in the repo)
├── toti_senatorii.json      # Senate registry (already in the repo)
└── stenograme/              # one JSON file per parliamentary session (add more over time)
    ├── stenograma_2025-02-03_1.json
    ├── stenograma_2025-02-03_2.json
    └── ...
```

Stenogram files are produced by [votez-scraper](#3-stenogram-scraper-optional--fetches-stenograms-from-cdepro) or can be provided manually. Each `stenograma_*.json` must follow this minimal shape:

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

### 5. Tesseract OCR (optional — required for law initiator extraction)

The deputy activity crawler can OCR `Expunerea de motive` PDFs to identify the deputies who actually authored/worked on a law initiative:

```bash
brew install tesseract
```

For best Romanian OCR, install the Romanian language data too if your Tesseract package does not include it. The crawler defaults to `ron+eng` and falls back to `eng`.

---

## Two main flows

Pick the flow that matches your situation. Each one is self-contained — just copy-paste the commands in order. Make sure you have completed the [Prerequisites](#prerequisites) above first.

| Flow | When to use | What it does |
|------|-------------|--------------|
| [A. Fresh setup](#a-fresh-setup--from-zero-to-full-outputs) | First time, or after a `reset_state.py` | Processes all stenograms, crawls all deputy activity |
| [B. Incremental update](#b-incremental-update--only-new-data) | Stenograms or deputy data have been added since the last run | Processes only new/changed files, skips already-analyzed data |

---

### A. Fresh setup — from zero to full outputs

Use this after cloning the repository (or after a full state reset). Scrapes all stenograms from cdep.ro, processes them, and crawls all deputy activity.

**Single command** — scrapes stenograms, syncs them, runs the full pipeline, and crawls deputy activity:

With **local LLM** (Ollama — free, default):

```bash
ollama serve                          # in a separate terminal, keep running
python3 scripts/full_update.py \
    --update-existing-crawler \
    --adopted-law-llm-provider ollama
```

With **OpenAI API** (remote, paid). Adopted-law impact analysis defaults to OpenAI `gpt-5-mini` unless overridden:

```bash
python3 scripts/full_update.py \
    --llm-provider openai \
    --llm-model gpt-5-nano \
    --update-existing-crawler
```

The script runs 8 steps in order: scrape from cdep.ro → sync to `input/` → analysis pipeline → productivity export → deputy activity crawler → adopted-law PDF/text + impact analysis → JSON export → deploy to `votez-frontend/`.

**Skip individual steps / opt-in extras:**

| Flag | Effect |
|------|--------|
| `--only-step {1..8}` | Run **only** this step, skip all others (1=scrape 2=sync 3=pipeline 4=productivity 5=crawler 6=adopted-law-enrichment 7=export 8=deploy) |
| `--skip-scrape` | Step 1 — don't hit cdep.ro, just use existing scraper output |
| `--skip-sync` | Step 2 — don't copy files from scraper to `input/` |
| `--skip-pipeline` | Step 3 — don't run the analysis pipeline |
| `--skip-productivity` | Step 4 — don't re-export productivity metrics |
| `--skip-crawler` | Step 5 — skip the deputy activity crawl entirely |
| `--hydrate-law-initiators` | Step 5 opt-in — after crawling, download each law's *Expunerea de motive* PDF, OCR it with Tesseract, and mark initiating deputies. Slow; omitted by default. |
| `--skip-adopted-law-enrichment` | Step 6 — skip adopted-law PDF/text extraction and citizen-facing impact analysis |
| `--adopted-law-limit N` | Step 6 — process at most N adopted laws; useful for smoke tests |
| `--adopted-law-extract-only` | Step 6 — don't download PDFs; only extract text from cached `outputs/pdfs/adopted_laws/` files |
| `--force-adopted-law-extract` | Step 6 — re-extract text even when `adopted_law_text_json` already exists |
| `--skip-adopted-law-analysis` | Step 6 — hydrate adopted-law PDFs/text but skip the LLM impact analysis |
| `--force-adopted-law-analysis` | Step 6 — re-run impact analysis even when `adopted_law_analysis_json` already exists |
| `--adopted-law-llm-provider {openai,ollama}` | Step 6 — provider for adopted-law impact analysis (default: `openai`) |
| `--adopted-law-llm-model MODEL` | Step 6 — model for adopted-law impact analysis (default: `gpt-5-mini` for OpenAI) |
| `--skip-export` | Step 7 — don't re-export JSON outputs from DB to `outputs/` |
| `--skip-deploy` | Step 8 — don't copy outputs to `votez-frontend/` |

**Result:** everything lands in `outputs/`, `state/state.sqlite`, and `../votez-frontend/` (`data/activity_analizer/` + `lib/`). See [Where everything lands](#where-everything-lands) for the full layout.

<details>
<summary><b>Manual step-by-step alternative</b> (without full_update.py)</summary>

With **local LLM** for both intervention analysis and adopted-law impact analysis:

```bash
# 1. Start Ollama in a separate terminal (keep it running)
ollama serve

# 2. Scrape new stenograms (in the scraper project)
cd ../votez-scraper && python3 main_stenograme.py --scrape && cd -

# 3. Sync stenograms to input/
cp -n ../votez-scraper/output/stenograme/stenograma_*.json input/stenograme/

# 4. Main pipeline: baseline + LLM classification + export
python3 scripts/run_pipeline.py --analyzer-mode llm

# 5. Productivity metrics
python3 scripts/export_effectiveness.py

# 6. Crawl deputy activity + OCR law initiators + export activity snapshots
python3 scripts/crawl_deputy_activity.py \
    --update-existing \
    --hydrate-law-initiators \
    --export-activity

# 7. Deploy outputs to frontend
rsync -a --delete --exclude='pdfs/' outputs/ ../votez-frontend/data/activity_analizer/
```

With **OpenAI API** (remote, paid):

```bash
# 1. Scrape new stenograms (in the scraper project)
cd ../votez-scraper && python3 main_stenograme.py --scrape && cd -

# 2. Sync stenograms to input/
cp -n ../votez-scraper/output/stenograme/stenograma_*.json input/stenograme/

# 3. Main pipeline: baseline + LLM classification + export
python3 scripts/run_pipeline.py \
    --analyzer-mode llm \
    --llm-provider openai \
    --llm-model gpt-5-nano

# 4. Productivity metrics
python3 scripts/export_effectiveness.py

# 5. Crawl deputy activity + OCR law initiators + export activity snapshots
python3 scripts/crawl_deputy_activity.py \
    --update-existing \
    --hydrate-law-initiators \
    --export-activity

# 6. Deploy outputs to frontend
rsync -a --delete --exclude='pdfs/' outputs/ ../votez-frontend/data/activity_analizer/
```

</details>

---

### B. Incremental update — only new data

Use this when the repository is already set up and you want to fetch the latest stenograms and refresh everything. Each step is safe to run repeatedly — already-processed data is automatically skipped.

**Single command** — scrapes only new stenograms (the scraper is incremental), syncs them, and processes only what's new:

With **local LLM** for both intervention analysis and adopted-law impact analysis:

```bash
ollama serve                          # in a separate terminal, keep running
python3 scripts/full_update.py --adopted-law-llm-provider ollama
```

With **OpenAI API** (remote, paid). Adopted-law impact analysis defaults to OpenAI `gpt-5-mini` unless overridden:

```bash
python3 scripts/full_update.py \
    --llm-provider openai \
    --llm-model gpt-5-nano
```

**Skip individual steps / opt-in extras:**

| Flag | Effect |
|------|--------|
| `--only-step {1..8}` | Run **only** this step, skip all others (1=scrape 2=sync 3=pipeline 4=productivity 5=crawler 6=adopted-law-enrichment 7=export 8=deploy) |
| `--skip-scrape` | Step 1 — don't hit cdep.ro, just use existing scraper output |
| `--skip-sync` | Step 2 — don't copy files from scraper to `input/` |
| `--skip-pipeline` | Step 3 — don't run the analysis pipeline |
| `--skip-productivity` | Step 4 — don't re-export productivity metrics |
| `--skip-crawler` | Step 5 — skip the deputy activity crawl entirely |
| `--hydrate-law-initiators` | Step 5 opt-in — after crawling, download each law's *Expunerea de motive* PDF, OCR it with Tesseract, and mark initiating deputies. Slow; omitted by default. |
| `--skip-adopted-law-enrichment` | Step 6 — skip adopted-law PDF/text extraction and citizen-facing impact analysis |
| `--adopted-law-limit N` | Step 6 — process at most N adopted laws; useful for smoke tests |
| `--adopted-law-extract-only` | Step 6 — don't download PDFs; only extract text from cached `outputs/pdfs/adopted_laws/` files |
| `--force-adopted-law-extract` | Step 6 — re-extract text even when `adopted_law_text_json` already exists |
| `--skip-adopted-law-analysis` | Step 6 — hydrate adopted-law PDFs/text but skip the LLM impact analysis |
| `--force-adopted-law-analysis` | Step 6 — re-run impact analysis even when `adopted_law_analysis_json` already exists |
| `--adopted-law-llm-provider {openai,ollama}` | Step 6 — provider for adopted-law impact analysis (default: `openai`) |
| `--adopted-law-llm-model MODEL` | Step 6 — model for adopted-law impact analysis (default: `gpt-5-mini` for OpenAI) |
| `--skip-export` | Step 7 — don't re-export JSON outputs from DB to `outputs/` |
| `--skip-deploy` | Step 8 — don't copy outputs to `votez-frontend/` |

To restrict scraping to a specific time range:

```bash
python3 scripts/full_update.py --scrape-year 2026 --scrape-month 5
```

**What gets skipped and why:**

| Data | Tracked by | Skip condition |
|------|-----------|----------------|
| Stenograms on cdep.ro | Scraper checks latest date in `output/stenograme/` | Only fetches days after the most recent existing stenogram |
| Stenogram files (sync) | Byte-level file comparison | Identical files are not re-copied to `input/stenograme/` |
| Stenogram files (pipeline) | `processed_stenograms` table (SHA-256 hash) | Same file content already processed |
| Session topics | `session_topics.topics_source` | Any `llm_v1:*` source exists for that session |
| Intervention labels | `intervention_analysis.relevance_source` | `llm_agent_v1` row exists for that intervention |
| Deputy activity pages | `dep_act_member_activity_crawl` | Row exists (unless `--update-existing-crawler` is passed) |
| Law initiator PDFs | `outputs/pdfs/law_initiators/` cache | Cached PDF file exists on disk |
| Adopted-law PDFs | `outputs/pdfs/adopted_laws/` cache | Cached PDF file exists on disk |
| Adopted-law text | `dep_act_laws.adopted_law_text_json` | Non-empty text JSON exists unless `--force-adopted-law-extract` is passed |
| Adopted-law impact analysis | `dep_act_laws.adopted_law_analysis_json` | Non-empty analysis JSON exists unless `--force-adopted-law-analysis` is passed |

**Tip:** To check what would be processed without making any changes:

```bash
python3 scripts/full_update.py --dry-run
```

<details>
<summary><b>Manual step-by-step alternative</b> (without full_update.py)</summary>

With **local LLM** (Ollama — free, default):

```bash
# 1. Make sure Ollama is running (skip if already started)
ollama serve

# 2. Scrape only new stenograms (incremental — skips already-fetched dates)
cd ../votez-scraper && python3 main_stenograme.py --scrape && cd -

# 3. Sync new/changed files to input/
cp -n ../votez-scraper/output/stenograme/stenograma_*.json input/stenograme/

# 4. Pipeline: only processes new/changed stenograms
python3 scripts/run_pipeline.py --analyzer-mode llm

# 5. Re-export productivity metrics (picks up any new data)
python3 scripts/export_effectiveness.py

# 6. Refresh deputy activity (only new crawler data + new law initiators)
python3 scripts/crawl_deputy_activity.py \
    --hydrate-law-initiators \
    --export-activity

# 7. Deploy outputs to frontend
rsync -a --delete --exclude='pdfs/' outputs/ ../votez-frontend/data/activity_analizer/
```

With **OpenAI API** (remote, paid):

```bash
# 1. Scrape only new stenograms
cd ../votez-scraper && python3 main_stenograme.py --scrape && cd -

# 2. Sync new/changed files to input/
cp -n ../votez-scraper/output/stenograme/stenograma_*.json input/stenograme/

# 3. Pipeline: only processes new/changed stenograms
python3 scripts/run_pipeline.py \
    --analyzer-mode llm \
    --llm-provider openai \
    --llm-model gpt-5-nano

# 4. Re-export productivity metrics (picks up any new data)
python3 scripts/export_effectiveness.py

# 5. Refresh deputy activity (only new crawler data + new law initiators)
python3 scripts/crawl_deputy_activity.py \
    --hydrate-law-initiators \
    --export-activity

# 6. Deploy outputs to frontend
rsync -a --delete --exclude='pdfs/' outputs/ ../votez-frontend/data/activity_analizer/
```

</details>

---

## Detailed guide

This section explains what each pipeline step does, where outputs land, and how to iterate. For the copy-paste command flows, see [A. Fresh setup](#a-fresh-setup--from-zero-to-full-outputs) or [B. Incremental update](#b-incremental-update--only-new-data) above.

### Processing steps

**Step A — start Ollama (only if you're using the local LLM).** Keep it running in its own terminal:

```bash
ollama serve
```

**Step B — run the main pipeline (topics + intervention classification + exports).** Only new/changed stenograms are processed, so this is safe to rerun after adding files:

```bash
# Local LLM (default):
python3 scripts/run_pipeline.py --analyzer-mode llm

# OpenAI API:
python3 scripts/run_pipeline.py --analyzer-mode llm --llm-provider openai --llm-model gpt-5-nano
```

Writes to `outputs/members/`, `outputs/parties/`, `outputs/topics/`, `outputs/session_topics/`.

**Step C — productivity metrics (word- and letter-weighted):**

```bash
python3 scripts/export_effectiveness.py
```

Writes to `outputs/productivity/`.

**Step D — crawl the CDEP deputy activity pages, hydrate law initiators, and export activity snapshots.** Single command, end-to-end:

```bash
python3 scripts/crawl_deputy_activity.py \
    --update-existing \
    --hydrate-law-initiators \
    --export-activity
```

Writes to `state/state.sqlite` (crawler tables) and `outputs/activity/members/` + `outputs/activity/parties/`.
The hydrator also caches the downloaded initiator PDFs under `outputs/pdfs/law_initiators/` so reruns can extract locally without re-fetching the same law PDFs.

### Where everything lands

```text
state/state.sqlite                      # unified DB — interventions, crawler data, runs
outputs/members/                        # per-member interventions with labels & topics
outputs/parties/                        # per-party interventions
outputs/session_topics/                 # per-session derived topics
outputs/topics/                         # per-topic roll-ups
outputs/productivity/                   # word/letter productivity metrics (members + parties + total)
outputs/activity/members/               # per-member crawler activity snapshots (motions, Q&I, laws, ...)
outputs/activity/parties/               # per-party aggregations (initiated laws, majority support, ...)
outputs/activity/adopted_laws/          # one JSON per adopted law with extracted text + impact analysis
outputs/pdfs/law_initiators/            # cached law-initiator PDFs reused by OCR hydration
outputs/pdfs/adopted_laws/              # cached adopted-law "Forma adoptată" PDFs
```

### Iterating

- Adding new stenograms? Drop them into `input/stenograme/` and rerun Step B — processed stenograms are tracked in the DB and skipped automatically.
- Tweaking the snapshot shape? `python3 scripts/crawl_deputy_activity.py --only-export-activity` rebuilds `outputs/activity/` from the current DB without recrawling.
- First-time smoke test? `python3 scripts/run_pipeline.py --analyzer-mode llm --llm-sessions-limit 3 --llm-speech-limit 10` runs a tiny slice end-to-end.

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
python3 scripts/run_pipeline.py --analyzer-mode llm --llm-provider openai --llm-model gpt-5-nano
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
| `scripts/full_update.py` | End-to-end orchestrator — scrape stenograms + sync + pipeline + exports + crawler in one command |
| `scripts/run_pipeline.py` | Main orchestrator — incremental, handles baseline + LLM + export |
| `scripts/hydrate_adopted_laws.py` | Download adopted-law "Forma adoptată" PDFs, cache them, and extract structured law text into `dep_act_laws` |
| `scripts/analyze_adopted_laws.py` | Three-stage LLM pipeline that turns extracted adopted-law text into citizen-friendly impact analysis and reader summary JSON |
| `scripts/analyze_interventions.py` | Baseline classifier (keyword + RAG index build) |
| `scripts/llm_session_topics.py` | LLM session topic extraction — runs before intervention classification |
| `scripts/llm_agent.py` | LLM intervention classification — the main intelligence layer |
| `scripts/rag_store.py` | Vector index build and retrieval (`sentence-transformers` + FAISS) |
| `scripts/mcp_server.py` | MCP tool server (all read, RAG, and write tools) |
| `scripts/export_effectiveness.py` | Export word/letter productivity metrics by member, party, and total |
| `scripts/crawl_deputy_activity.py` | Crawl deputy profile activity pages from CDEP and store laws, decision projects, questions/interpellations, motions, and written political declarations in SQLite; also drives local law-initiator OCR hydration and the activity JSON export |
| `scripts/export_activity.py` | Serialize the crawler DB into per-member and per-party activity JSON snapshots under `outputs/activity/` (invoked from `crawl_deputy_activity.py` via `--export-activity`) |
| `scripts/inspect_retrieval.py` | Inspect retrieved chunks for any intervention |
| `scripts/demo_mcp.py` | Exercise all MCP tools end-to-end |

---

## Crawl deputy activity from CDEP

**Recommended usage** — one command that does the full pipeline end-to-end: crawl every deputy's activity pages, OCR-hydrate law initiators, and refresh the per-member + per-party JSON snapshots under `outputs/activity/`:

```bash
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
python3 scripts/crawl_deputy_activity.py --only-hydrate-law-initiators --hydrate-law-limit 10
python3 scripts/crawl_deputy_activity.py --only-hydrate-law-initiators
python3 scripts/crawl_deputy_activity.py --export-activity --limit 5
python3 scripts/crawl_deputy_activity.py --only-export-activity
python3 scripts/crawl_deputy_activity.py --dry-run --limit 1
```

For every processed deputy, the script logs the profile count, stored record count, association count, and source URL for each activity type. By default, it inserts only new crawler data: existing rows in crawler-owned tables are left unchanged, while new member associations are still added. Pass `--update-existing` when you want parsed CDEP pages to refresh existing crawler rows.

Pass `--hydrate-law-initiators` to run a second phase after the normal crawl finishes: the script reads stored laws from `dep_act_laws`, fetches each law's `Expunerea de motive` PDF through `source_url`, OCRs it locally with Tesseract, extracts the `Iniţiatori` section, and marks matching deputies in `dep_act_member_laws.is_initiator`. This is intentionally optional because scanned PDF OCR is slower than the normal crawl.

Before fetching anything, the hydrator looks for a cached initiator PDF in `outputs/pdfs/law_initiators/` named `initiators_<law_identifier>.pdf` (sanitized for filenames). If that file exists, extraction runs entirely locally and the law page/PDF are not fetched again. On a cache miss, the hydrator fetches the law page, picks the best initiator PDF candidate, downloads it into that cache folder, and then runs extraction from the cached local file.

For some `senat.ro` laws, the stored `source_url` resolves to the generic legislation search page rather than directly to the act details. In that case the hydrator now submits the Senat search form using the stored law number/year, reuses the resulting act page HTML, and then continues with the normal `AD.PDF` / `EM.PDF` / `Forma inițiatorului` discovery logic.

Pass `--only-hydrate-law-initiators` to skip the crawl phase and retry only stored laws that do not yet have any `is_initiator = 1` association in `dep_act_member_laws`. OCR progress logs include the `dep_act_laws.source_url` law page link for debugging no-match cases.

Pass `--hydrate-law-limit N` to cap the hydration phase to the first `N` stored laws in `law_id` order after the normal member scoping/filtering. When a limit is set, the hydrator prefers laws that do not already have a cached PDF in `outputs/pdfs/law_initiators/`, so local smoke tests do not waste their limited slots re-downloading PDFs you already have.

At the end of every hydration phase, the script also logs how many `dep_act_laws` rows in the database still do not have a cached initiator PDF in `outputs/pdfs/law_initiators/`, so you can track the remaining local download backlog even when you run with a limit.

The crawler writes only its own tables: `dep_act_member_activity_crawl`, `dep_act_laws`, `dep_act_member_laws`, `dep_act_decision_projects`, `dep_act_member_decision_projects`, `dep_act_questions_interpellations`, `dep_act_motions`, `dep_act_member_motions`, and `dep_act_political_declarations`. It validates that targeted deputies already exist in `members`, but it never inserts or updates `members`, interventions, runs, outputs, or other pipeline tables.

Laws, decision projects, and motions can be associated with several deputies, so the entity tables are deduplicated and the `dep_act_member_*` tables store the many-to-many associations.

---

## Adopted-law text and impact analysis

Step 6 of `scripts/full_update.py` enriches adopted laws in two phases:

1. `scripts/hydrate_adopted_laws.py` reads `dep_act_laws` rows with a final `adopted_law_identifier` or `law_status='adoptata_in_parlament'`, reuses or downloads the adopted-law "Forma adoptată" PDF into `outputs/pdfs/adopted_laws/`, extracts the PDF text, and stores structured text in `dep_act_laws.adopted_law_text_json`. `law_status='adoptata'` is reserved for rows that already have a final law identifier such as `Lege 233/2025`; chamber/parliament adoption before final publication is stored separately as `adoptata_in_parlament`.
2. `scripts/analyze_adopted_laws.py` reads that extracted text and runs a three-stage LLM pipeline:
   - factual extraction: affected legal acts, obligations, rights, penalties/costs, institutions, dates, target groups;
   - citizen interpretation: plain-language title, summary, affected groups, practical impact, labels and scores;
   - critic/validation: removes or flags unsupported claims and writes the final validated analysis.

The analysis deliberately uses the extracted law text as the primary source of truth, not just the title. Existing analyses are skipped by default: a row is re-analyzed only when `adopted_law_analysis_json` is empty, unless you pass `--force-adopted-law-analysis`.

Useful commands:

```bash
# Small end-to-end smoke test for adopted laws: PDF/text hydration + LLM analysis
python3 scripts/full_update.py --only-step 6 --adopted-law-limit 3

# Analyze only, assuming adopted-law text is already extracted
python3 scripts/analyze_adopted_laws.py --limit 3

# Force-refresh existing analyses
python3 scripts/analyze_adopted_laws.py --limit 3 --force

# Use a local model instead of the default OpenAI gpt-5-mini
python3 scripts/analyze_adopted_laws.py --provider ollama --model qwen3:14b --limit 3
```

Stored DB fields on `dep_act_laws` include:

- `adopted_law_pdf_filename`, `adopted_law_pdf_url`
- `adopted_law_text_json`, `adopted_law_text_extracted_at`, `adopted_law_parse_error`
- `adopted_law_analysis_json`
- `adopted_law_reader_summary` — JSON object optimized for frontend cards
- `adopted_law_analyzed_at`, `adopted_law_analysis_source`, `adopted_law_analysis_error`

After export (step 7), each adopted law gets a dedicated file:

```text
outputs/activity/adopted_laws/adopted_law_<law_slug>.json
```

Member and party activity JSONs do not embed full law text or analysis. They only include `adopted_law_details_id` and `adopted_law_details_path`, so the frontend can load the detail file when needed.

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
outputs/activity/adopted_laws/adopted_law_{law_slug}.json
```

Each **member** file contains the member's identity (`member_id`, `name`, `chamber`, `party_id`, `party_name`, `profile_url`) plus five activity blocks:

- `motions[]` — each motion's stored columns, plus `co_supporting_parties: [{party_id, party_name, members_count}]`. The subject member is excluded from his own party's count (so the number reflects "how many **other** members of that party co-supported").
- `questions_and_interpellations[]` — every Q&I row for the member, including `identifier`, `recipient`, `text`, `source_url`, and the raw listing `columns`.
- `political_declarations[]` — every written political declaration, including `title`, `full_text`, `text_url`, and the raw listing `columns`.
- `decision_projects[]` — each project, enriched with `collaborating_parties: [{party_id, party_name, members_count, members: [{member_id, name}]}]` grouping every collaborator by party.
- `laws[]` — every law the member is linked to via `dep_act_member_laws`. Each entry adds `is_initiator` (for this member), `is_adopted` (from `adopted_law_identifier`), `initiator_parties` (members with `is_initiator=1` grouped by party), and `supporter_parties` (**all** linked members grouped by party — initiators are a subset).

For adopted laws with detail data, law entries also include compact references:

- `adopted_law_details_id` — currently the same as `law_id`;
- `adopted_law_details_path` — relative path such as `adopted_laws/adopted_law_lege-34-2025.json`.

Each **party** file aggregates its members:

- `members_count` and `majority_threshold = min(10, ceil(members_count / 2))`.
- `laws_initiated[]` — any party member was an initiator. Includes `party_initiators: [{member_id, name}]`, full `initiator_parties` / `supporter_parties` roll-ups, and `is_adopted`.
- `laws_majority_supported_only[]` — the party had **no** initiator on this law, but ≥ `majority_threshold` of its members appear as supporters. No overlap with `laws_initiated`.
- `questions_and_interpellations[]` — all Q&I from every party member, each tagged with `asked_by: {member_id, name}`.
- `motions_majority_supported[]` — motions where ≥ `majority_threshold` of the party's members are supporters, including `all_supporting_parties` counts.

Each **adopted-law detail** file contains the cached PDF metadata, extracted law text JSON, full `law_analysis`, and frontend-friendly `reader_summary` JSON.

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
| `dep_act_laws` | Deduplicated legislative proposal/law records from deputy activity pages, including adopted law identifiers such as `Lege 233/2025`, cached adopted-law PDF metadata/text, citizen-facing adopted-law analysis JSON, `Expunerea de motive` PDF URLs, OCR initiator text, and parse errors |
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
