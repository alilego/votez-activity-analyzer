# votez-activity-analyzer
AI-powered analysis of parliamentary activity, classifying interventions and extracting debate topics from stenograms.

## What it does

Given stenograms of parliamentary sessions, it outputs a dataset that maps each parliament member to:
- interventions classified as relevant / neutral / non-relevant to the session topic
- top 20 debate subjects (topics) where the member intervened

## How it runs

- Runs locally on a developer machine
- Intended to be run periodically to refresh the dataset
- Produces frontend-friendly output artifacts (format/schema will evolve)

Run the default incremental pipeline:

- `python scripts/run_pipeline.py`

What this command does:
- creates SQLite DB if missing (`state/state.sqlite`)
- initializes schema if not initialized
- selects only new/changed files from `input/stenograme/`
- runs analyzer (speaker normalization + member resolution + raw intervention persistence)
- computes deterministic baseline intervention analysis (labels + topics)
- exports frontend JSON artifacts to `outputs/`
- marks successfully processed stenograms in DB state
- stores run summary in DB table `run_outputs`

Dry run (selection only):

- `python scripts/run_pipeline.py --dry-run`

## Optional Commands

- Initialize DB only: `python scripts/init_db.py`
- Reset local state for a clean rerun: `python scripts/reset_state.py`
- Show selected files: `python scripts/select_stenograms.py`
- Mark selected files manually: `python scripts/mark_processed_stenograms.py --run-id <run_id>`
- Export outputs only: `python scripts/export_outputs.py`
- Validate exported outputs: `python scripts/validate_outputs.py`
- Use custom analyzer: `python scripts/run_pipeline.py --analyzer-cmd "<your command>"`

When using a custom analyzer command, these env vars are provided:
- `VOTEZ_RUN_ID`
- `VOTEZ_STENOGRAM_LIST_PATH`

Current SQLite tables:
- `metadata`
- `runs`
- `processed_stenograms`
- `run_outputs`
- `members`
- `interventions_raw`
- `intervention_analysis` (classification scaffolding table)
- `unmatched_speakers`

## Docs

- `goal.md`
- `architecture.md`
- `input-data.md`
- `classification-rubric.md`
- `rag-indexing.md`
- `mcp-tools.md`
- `output-contract.md`