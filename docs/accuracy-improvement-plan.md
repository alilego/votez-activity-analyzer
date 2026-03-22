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

### 2.1 Deterministic law-ID regex extraction ✅
- [x] Add `scripts/law_extractor.py` with comprehensive patterns:
  - `PL-x NNN/YYYY`, `Legea nr. NNN/YYYY`, `OUG nr. NNN/YYYY`, `HG nr. NNN/YYYY`
  - `Directiva UE ...`, `Regulamentul UE ...`
  - `Ordonanța de urgență`, `Hotărârea Guvernului` (full Romanian forms)
  - Generic `nr. NNN/YYYY` in legislative context (proximity-based)
- [x] Build a per-session `SessionLawIndex` with `{law_id: [speech_indices]}` mapping
- [x] Persist law indices as JSON in `state/law_indices/` for downstream use
- [x] Inject pre-extracted law references into Layer A/B/C prompts
- [x] Add `validate_law_ids()` to reject hallucinated LLM law IDs
- [x] Replace simpler `LAW_REFERENCE_PATTERNS` in analyzer with comprehensive extractor
- **Expected impact:** +10-15% on law attribution accuracy

### 2.2 Pre-extract legislative agenda from session notes ✅
- [x] Parse `initial_notes` for numbered/bulleted/lettered agenda items with law references
- [x] Build structured agenda: `[AgendaItem(item_number, title, law_ids)]`
- [x] Persist agenda alongside law index in JSON artifact
- [x] Feed agenda to classification prompts (combined with law index)
- **Expected impact:** +5-10% on law attribution accuracy

### 2.3 Additional deterministic shortcut rules ✅
- [x] Ultra-short speeches (≤10 words) that are greetings/thanks → `neutral` (conf 0.95) without LLM call
- [x] Ultra-short procedural replies (≤3 words, e.g. "Da.", "Nu.") → `neutral` (conf 0.92)
- [x] Vote announcement patterns ("supun la vot", "cine este pentru", "votul a fost") → `neutral` (conf 0.92)
- [x] Session chair procedural lines (≤50 words) → `neutral` (conf 0.90)
- [x] Committee report readings (detect "raportul comisiei" + formal structure) → `constructive` candidate
- [x] Pre-LLM shortcuts in `rules.py` (`apply_pre_llm_shortcuts()`) — skips LLM entirely
- [x] Baseline analysis improvements in `analyze_interventions.py`
- **Expected impact:** +3-5% classification accuracy, reduced LLM calls by ~10-15%

### 2.4 Tighten QA trigger thresholds ✅
- [x] Raise confidence threshold from 0.65 to 0.70 for `low_confidence` trigger
- [x] Suppress `very_short_speech` trigger for speeches already handled by deterministic rules
- [x] Add `_is_clear_procedural_short()` helper for smart trigger suppression
- **Expected impact:** -20-30% LLM calls with no accuracy loss

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
| 2026-03-22 | Phase 2.1 — law-ID regex extraction | `scripts/law_extractor.py` with comprehensive patterns, per-session index, prompt injection, validation |
| 2026-03-22 | Phase 2.2 — agenda parsing | Parse initial_notes for structured agenda items with law references |
| 2026-03-22 | Phase 2.3 — deterministic shortcuts | Pre-LLM shortcuts for greetings, vote announcements, chair lines, committee reports; ~10-15% fewer LLM calls |
| 2026-03-22 | Phase 2.4 — QA threshold tightening | low_confidence raised 0.65→0.70; very_short_speech suppressed for procedural shorts |

---

*Last updated: 2026-03-22*
