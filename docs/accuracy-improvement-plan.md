# Accuracy Improvement Plan

**Target:** 98% classification accuracy (constructive / neutral / non_constructive) and 95% law/amendment attribution accuracy.

**Principle:** Prefer local models → cheap APIs → expensive APIs (last resort).

---

## Current State

- **Model:** `qwen2.5:7b` (32k context) via Ollama
- **Pipeline:** 3-layer (Layer A rubric extraction → deterministic rules → Layer B decision → QA triggers → Layer C review)
- **Measured baseline accuracy:** 61.0% classification (77 speeches), 0% law attribution (on 5 processed sessions)
- **Main weaknesses:** 7B model struggles with nuanced Romanian parliamentary language, law IDs are often missed or hallucinated, no evaluation framework to measure progress

---

## Phase 1 — Foundations (measure before optimizing)

### 1.1 Build gold-standard evaluation set ✅
- [x] Select 255 speeches balanced across 18 sessions, with length variety (74 short, 101 medium, 80 long)
- [x] Include 55 speeches with detected law/amendment references
- [x] Store as `tests/gold_standard.json`
- [x] AI-generated first-pass labels for 229 speeches (8 parallel agents with full stenogram context)
- [x] 26 speeches manually labeled by human reviewer as reference examples
- [x] Human review of all 255 entries completed
- **Final distribution:** 86 constructive (33.7%), 113 neutral (44.3%), 56 non_constructive (22.0%)
- **Difficulty:** 166 easy, 59 medium, 30 hard
- **Coverage:** 42 speeches with `expected_law_ids`, 232 with `expected_topics`

### 1.2 Build evaluation harness script ✅
- [x] Create `scripts/evaluate_accuracy.py` that runs the pipeline against the gold set
- [x] Report per-label precision, recall, F1
- [x] Report law/amendment attribution accuracy (exact match and partial match)
- [x] Report confusion matrix and confidence calibration
- [x] Baseline measured (77/255 speeches evaluable from 5 processed sessions)

**Baseline results (qwen2.5:7b, three_layer pipeline):**
- Classification accuracy: **61.0%** (47/77) — gap to 98% target: **37pp**
- Law attribution: **0.0%** (0/33 exact, 0/33 partial) — gap to 95% target: **95pp**
- Per-label F1: constructive 49.0%, neutral 75.4%, non_constructive 50.0%
- Confidence calibration is inverted (high-confidence predictions are less accurate)
- Key failure patterns:
  - Committee reports misclassified as neutral (model treats formal tone as procedural)
  - Constructive political declarations with strong tone misclassified as non_constructive
  - Session moderation misclassified as constructive/non_constructive
  - No law IDs extracted or matched at all

---

## Phase 2 — Deterministic improvements (no model change needed)

### 2.1 Deterministic law-ID regex extraction
- [x] Add a function that scans all session speech text for Romanian law patterns:
  - `PL-x NNN/YYYY`, `Legea nr. NNN/YYYY`, `OUG nr. NNN/YYYY`, `HG nr. NNN/YYYY`
  - `Directiva UE ...`, `Regulamentul UE ...`
  - Generic `nr. NNN/YYYY` in legislative context
- [x] Build a per-session `{law_id: [speech_indices]}` index
- [x] Inject this structured list into topic extraction and classification prompts as pre-extracted facts
- [x] Validate LLM-returned `law_id` against the pre-extracted list (reject hallucinated IDs)
- **Expected impact:** +10-15% on law attribution accuracy

### 2.2 Pre-extract legislative agenda from session notes ✅
- [x] Parse `initial_notes` for agenda items (often numbered, with law references)
- [x] Build a structured agenda: `[{item_number, title, law_id}]`
- [x] Feed this to both topic extraction and classification prompts
- [x] Also fixed incomplete 2.1: added `law_id_index` + `_format_preextracted_law_ids` to `prompts.py`
- **Implementation:** `scripts/agenda.py` scans both `initial_notes` and session speeches for legislative item introductions (Proiect de Lege/Hotărâre, Propunere legislativă, PL-x, PHCD, OUG refs)
- **Expected impact:** +5-10% on law attribution accuracy

### 2.3 Additional deterministic shortcut rules
- [ ] Ultra-short speeches (≤10 words) that are greetings/thanks → `neutral` without LLM call
- [ ] Vote announcement patterns ("supun la vot", "cine este pentru", "votul a fost") → `neutral`
- [ ] Committee report readings (detect "raportul comisiei" + formal structure) → `constructive` candidate
- [ ] Speaker role detection: session president / secretary procedural lines → weight toward `neutral`
- [ ] **Expected impact:** +3-5% classification accuracy, reduced LLM calls by ~10-15%

### 2.4 Tighten QA trigger thresholds
- [ ] Audit current QA trigger rates — if >30% of speeches trigger Layer C, the triggers are too loose
- [ ] Raise confidence threshold from 0.65 to 0.70 for `low_confidence` trigger
- [ ] Remove `very_short_speech` trigger for speeches already handled by deterministic rules
- [ ] **Expected impact:** -20-30% LLM calls with no accuracy loss

---

## Phase 3 — Model upgrade (local)

### 3.1 Upgrade to a stronger local model
- [ ] Test `qwen2.5:14b` (needs ~10GB VRAM) — best balance of quality vs. resource use
- [ ] Test `qwen3:14b` if available — newer architecture may handle Romanian better
- [ ] Test `gemma-3:27b` (needs ~18GB VRAM) — strong multilingual performance
- [ ] Test `llama3.3:70b-q4` (needs ~40GB VRAM) — if hardware allows, best local option
- [ ] For each model: run evaluation harness, compare accuracy, latency, and resource use
- [ ] Update `Modelfile-*` and default model constants accordingly
- [ ] **Expected impact:** +5-10% classification accuracy over 7B

### 3.2 Make pipeline architecture model-aware
- [ ] For 7B-14B models: keep 3-layer pipeline (model needs guardrails)
- [ ] For 27B+ local or API models: default to `one_pass` (stronger models do better holistically)
- [ ] Add a config mapping: `{model_name: {architecture, num_ctx, chunk_chars}}`
- [ ] **Expected impact:** Better accuracy for strong models, reduced latency

### 3.3 Enhance prompts with few-shot examples
- [ ] Add 3-5 Romanian examples to each layer prompt (one clear constructive, one neutral, one non_constructive, one edge case)
- [ ] Examples should include expected `law_id` attribution
- [ ] Test with and without few-shot on the evaluation set
- [ ] **Expected impact:** +2-3% classification accuracy, especially for edge cases

---

## Phase 4 — Hybrid escalation (local + cheap API)

### 4.1 Confidence-based API escalation
- [ ] After local model classifies all speeches, identify uncertain ones (confidence < 0.75 or QA-triggered)
- [ ] Send ONLY uncertain speeches (~7-15% of total) to `gpt-4o-mini` for re-classification
- [ ] Accept API result when its confidence > local confidence
- [ ] **Expected impact:** +5-8% classification accuracy at ~80-90% cost reduction vs. full API
- [ ] **Estimated cost:** ~$0.01-0.05 per session (vs. $0.10-0.50 for full API)

### 4.2 Final escalation to strong API (if needed)
- [ ] For speeches where `gpt-4o-mini` confidence < 0.65, escalate to `gpt-4o` or `claude-3.5-sonnet`
- [ ] This should be ~1-3% of total speeches
- [ ] **Expected impact:** +1-2% to reach the 98% target
- [ ] **Estimated cost:** ~$0.01-0.02 per session

---

## Phase 5 — Continuous improvement

### 5.1 Error analysis loop
- [ ] After each pipeline run, flag low-confidence and disagreement cases
- [ ] Periodically review and add to gold-standard set
- [ ] Track accuracy trends across runs

### 5.2 Topic taxonomy refinement
- [ ] Review `config/topic_taxonomy.json` catalog coverage
- [ ] Add missing topics discovered from `new_topics` in production runs
- [ ] Improve `token_equivalents` for Romanian morphological variants

### 5.3 RAG retrieval improvements
- [ ] Evaluate whether RAG context actually helps classification (A/B test with and without)
- [ ] Consider upgrading embedding model for better Romanian support
- [ ] Test chunk overlap strategies for better context retrieval

---

## Expected Accuracy Progression

| After Phase | Classification Accuracy | Law Attribution Accuracy | Model Cost |
|-------------|------------------------|--------------------------|------------|
| Current     | **61.0%** (measured)   | **0.0%** (measured)      | Free (local 7B) |
| Phase 2     | ~87-90%                | ~75-80%                  | Free (local 7B) |
| Phase 3     | ~93-95%                | ~85-90%                  | Free (local 14B+) |
| Phase 4     | ~97-98%                | ~93-95%                  | ~$0.02-0.07/session |
| Phase 5     | 98%+                   | 95%+                     | ~$0.02-0.07/session |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-15 | Created plan | Systematic approach to reach 98%/95% accuracy targets |
| 2026-03-15 | Built gold-standard set (255 speeches) | Sampled across all 18 sessions with length/topic variety |
| 2026-03-15 | AI first-pass labeling (229 speeches) | 8 parallel classification agents with full stenogram context; human-labeled 26 used as reference; saves manual effort while human review ensures quality |
| 2026-03-15 | Human review completed | All 255 labels reviewed and corrected; minor shifts: +2 constructive, +1 neutral, -3 non_constructive vs AI first-pass |
| 2026-03-15 | Evaluation harness built | `scripts/evaluate_accuracy.py` — baseline: 61.0% classification, 0% law attribution |
| 2026-03-22 | Agenda extraction (2.2) | `scripts/agenda.py` — pre-extracts structured legislative agenda from session speeches; injected into all layer prompts and session topic extraction |
| 2026-03-22 | Fixed incomplete 2.1 | Added `law_id_index` and `_format_preextracted_law_ids` to `prompts.py` — was missing from layer prompt builders |

---

*Last updated: 2026-03-22*
