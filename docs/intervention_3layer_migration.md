# Intervention Classifier Migration: One-Pass to 3-Layer

This project moved intervention classification from a monolithic one-pass prompt to a modular 3-layer flow, while keeping downstream compatibility.

## Mapping From Old Flow

- Old: one LLM call produced rubric-like signals + final label + confidence + topics.
- New:
  - Layer A: rubric extraction (observable speech signals)
  - Layer B: final decision (label/confidence/topics)
  - Layer C: QA/normalization (targeted only when triggered)

Legacy path is still available via `--pipeline-architecture one_pass`.
Current default is `--pipeline-architecture auto`, which resolves to `three_layer` for 7B/14B local models and `one_pass` for stronger local/API profiles.

## What Was Preserved

- Romanian Parliament domain framing
- Context rules (context is interpretation-only)
- Early procedural handling
- Criteria and decision guidance
- Confidence guidance
- Topic restrictions (only from provided session topics)
- Romanian one-sentence reasoning
- Verbatim evidence quote requirement
- Final output shape compatibility (`constructiveness_label`, rubric fields, confidence, topics, reasoning, evidence quote)

## Intentional Behavior Changes

- Deterministic pre-LLM decision shortcuts are now explicit code rules.
- QA layer runs only when triggers indicate ambiguity/inconsistency.
- Per-layer schema validation + retry with repair guidance is enforced.
- Additional logging was added for:
  - raw model response snippet
  - parsed output
  - validation failures
  - deterministic shortcut decisions
  - QA trigger reasons
  - merged final payload

## New Modules

- `scripts/intervention_layers/prompts.py`
- `scripts/intervention_layers/schemas.py`
- `scripts/intervention_layers/rules.py`
- `scripts/intervention_layers/qa.py`
- `scripts/intervention_layers/orchestrator.py`

## Running

Default (profile-aware):

```bash
python3 scripts/llm_agent.py --run-id <run_id>
```

Legacy:

```bash
python3 scripts/llm_agent.py --run-id <run_id> --pipeline-architecture one_pass
```
