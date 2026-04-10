# Accuracy Improvement Plan

**Target:** 98% classification accuracy (constructive / neutral / non_constructive) and 95% law/amendment attribution accuracy.

**Principle:** Prefer local models → cheap APIs → expensive APIs (last resort).

---

## Current State

- **Default local model:** `qwen3:14b` via Ollama
- **Benchmark API model tested:** `gpt-5.4-mini` via OpenAI
- **Pipeline:** model-aware defaults now resolve via shared profiles:
  - 7B-14B local models default to the guarded `three_layer` classifier
  - stronger local/API models default to `one_pass`
  - session topic extraction reuses model-specific chunk caps
- **Measured baseline accuracy:** 61.0% classification (77 speeches), 0% law attribution (original baseline on 5 processed sessions)
- **Latest benchmark signal:** `gpt-5.4-mini` reached **70.79% classification accuracy** and **21.4% exact-or-partial law attribution** on **89 medium/hard gold speeches across 17 sessions** with 100% coverage
- **Latest local benchmark signal:** `qwen3:14b` quick screen remained at **61.54% classification** and **0.0% law attribution** on 26 medium/hard speeches
- **Benchmark tooling status:** `scripts/benchmark_local_models.py` now supports Ollama and OpenAI models, mixed-provider runs, limited-cost smoke benchmarks, auto-import of missing gold sessions from `input/stenograme/`, and append-only benchmark history in `state/model_benchmarks/summary.json`
- **Main weaknesses:** non-constructive vs. neutral remains the hardest boundary, law IDs are still frequently missed, and even the stronger API benchmark is still far from the 98% / 95% target

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

### 2.3 Additional deterministic shortcut rules ✅
- [x] Ultra-short speeches (≤10 words) that are greetings/thanks → `neutral` without LLM call
- [x] Vote announcement patterns ("supun la vot", "cine este pentru", "votul a fost") → `neutral`
- [x] Committee report readings (detect "raportul comisiei" + formal structure) → `constructive` candidate
- [x] Speaker role detection: session president / secretary procedural lines → weight toward `neutral`
- [x] Ultra-short chair name-calls (≤5 words, "Domnul/Doamna X") → `neutral` without LLM call
- [x] Ultra-short floor responses (≤3 words: "Da.", "Nu.", "Prezent.") → `neutral` without LLM call
- **Implementation:** Pre-LLM shortcuts in `scripts/intervention_layers/rules.py` (`apply_pre_llm_shortcuts`) bypass the entire LLM pipeline for trivially-classifiable speeches. Post-Layer-A enhancements add committee report detection and session chair procedural bias. Wired into both three-layer and one-pass classifiers in `llm_agent.py`.
- **Data analysis:** 1,076 ultra-short speeches (35% of all 3,091), 74 vote announcements, 70 committee reports, 1,143 procedural chair speeches (37%)
- **Expected impact:** +3-5% classification accuracy, reduced LLM calls by ~10-15%

### 2.4 Tighten QA trigger thresholds ✅
- [x] Audit current QA trigger rates — `very_short_speech` alone triggered for 52.5% of all speeches (OR condition: `wc ≤ 25 OR sc ≤ 2` was far too loose)
- [x] Raise confidence threshold from 0.65 to 0.70 for `low_confidence` trigger
- [x] Tighten `very_short_speech` from OR to AND: now requires `wc ≤ 25 AND sc ≤ 2` (eliminates 38 substantive 1-2 sentence speeches from triggering)
- [x] Suppress `very_short_speech` when post-Layer-A deterministic rules already provide candidate labels (avoids redundant Layer C for trivially-classifiable speeches)
- [x] Pass `deterministic_candidates` from `apply_deterministic_rules` to `evaluate_qa_triggers` in the three-layer pipeline
- **Implementation:** Changes in `scripts/intervention_layers/qa.py` (threshold + condition + suppression parameter) and `scripts/llm_agent.py` (wiring). 4 new tests in `test_intervention_layers.py`.
- **Audit results:** OR→AND reduces `very_short_speech` triggers from 52.5% to 38.6%; combined with pre-LLM shortcuts (22.7%) and deterministic suppression, effective QA trigger rate drops well below 30%
- **Expected impact:** -20-30% LLM calls with no accuracy loss

---

## Phase 3 — Model upgrade (local)

### 3.1 Upgrade to a stronger local model
- **Why this can help accuracy:** a stronger model may better separate formal-but-substantive speeches from procedural ones, handle sharper political rhetoric without overpredicting `non_constructive`, and recover indirect law references that the 7B model misses.
- **Important:** this is a hypothesis to validate after Phase 2, not a mandatory full benchmark of every candidate model.
- [ ] First check whether the remaining errors are actually model-limited:
  - If most errors are rubric/prompt/context failures, stay in Phase 2/3.3 instead of spending hours on model swaps.
  - If most errors are nuanced semantic judgments or missed indirect legislative references, proceed with model screening.
- [x] Run a **quick screening benchmark** first on medium+hard gold speeches, reusing existing session topics:
```bash
cd /Users/alilego/Projects/votez-activity-analyzer
python3 scripts/benchmark_local_models.py --models qwen3:14b --only-hard --reuse-existing-topics --skip-missing-sessions
```
- [x] `qwen3:14b` quick screen completed on currently available gold subset:
  - Command: `python3 scripts/benchmark_local_models.py --models qwen3:14b --only-hard --reuse-existing-topics --skip-missing-sessions`
  - Scope: 26 medium/hard speeches across 5 available gold sessions (`8851`, `8856`, `8909`, `8957`, `8958`); 100% coverage on that subset
  - Result: **61.54% classification accuracy** (16/26), **0.0% law attribution** (0/4 exact, 0/4 partial)
  - Latency: ~5,994s total (~100 min) even in quick-screen mode
  - Error pattern: very high `constructive` recall (92.9%) but poor `non_constructive` recall (27.3%); strongest failure mode remains overpredicting `constructive` for partisan/off-topic speeches
  - Takeaway: this run does **not** yet show a clear Phase 3 accuracy gain; by itself it does not justify spending more time on wider local-model benchmarking
- [ ] Promote a model to full benchmark only if quick screening shows a meaningful gain (target: at least +3pp classification accuracy on medium/hard cases, or materially better constructive/non_constructive recall) with acceptable latency.
- [ ] Only then run a **full end-to-end benchmark** with per-model topic extraction for 1-2 finalists:
```bash
cd /Users/alilego/Projects/votez-activity-analyzer
python3 scripts/benchmark_local_models.py --models qwen3:14b
```
- [ ] Candidate order:
  - `qwen3:14b` — cheapest strong local baseline; already the default
  - `qwen2.5:14b` (needs ~10GB VRAM) — useful regression check vs `qwen3:14b`
  - `gemma-3:27b` (needs ~18GB VRAM) — only if 14B models plateau
  - `llama3.3:70b-q4` (needs ~40GB VRAM) — only if hardware allows and local-only is still a hard requirement
- [x] Update `Modelfile-*` and default model constants accordingly
- [x] Add shared model-profile config + isolated benchmark harness (`scripts/benchmark_local_models.py`)
- [x] Extend benchmark harness to support OpenAI models with the same output format as Ollama benchmarks
- [x] Add a cheaper benchmark preset (`--benchmark-scope limited`) for first-pass API screening on medium/hard gold speeches
- [x] Auto-import missing gold sessions from `input/stenograme/` into a prepared benchmark DB so full gold-session benchmarks do not depend on the current contents of `state/state.sqlite`
- [x] Preserve benchmark history in `state/model_benchmarks/summary.json` via timestamped appended runs (`run_started_at`)
- [x] Store full mismatch text and full prediction reasoning in benchmark reports (no JSON truncation of `text_preview` / `pred_reasoning`)
- **Current default local model:** `qwen3:14b`
- **Decision rule:** if quick screening does not show a clear win, skip deeper local benchmarking and move to prompt improvements or hybrid escalation. Current `qwen3:14b` quick-screen result falls into this bucket.
- **Current API benchmark signal:** `gpt-5.4-mini` full medium/hard benchmark:
  - Command: `python3 scripts/benchmark_local_models.py --provider openai --models gpt-5.4-mini --only-hard`
  - Scope: 89 medium/hard gold speeches across 17 sessions; 100% coverage after auto-preparing missing gold sessions
  - Result: **70.79% classification accuracy** (63/89), **14.3% exact law attribution**, **21.4% exact-or-partial law attribution**, **34.0% topic F1**
  - Latency: ~280.89s total (122.84s topics + 158.06s classification)
  - Error pattern: much better than the local 14B quick screen on non-trivial classification and topic attribution, but still weak on neutral recall (37.5% F1) and still misses most law references
  - Takeaway: stronger API models help, but current gains are still not enough to skip prompt/rubric improvements
- [ ] **Expected impact (only if the error profile is model-limited):** +5-10% classification accuracy over 7B

### 3.2 Make pipeline architecture model-aware
- [x] For 7B-14B models: keep 3-layer pipeline (model needs guardrails)
- [x] For 27B+ local or API models: default to `one_pass` (stronger models do better holistically)
- [x] Add a config mapping: `{model_name: {architecture, num_ctx, chunk_chars}}`
- **Implementation:** shared runtime config in `scripts/model_profiles.py` now resolves preferred architecture, context size, and prompt chunk caps per model; `llm_agent.py` defaults to `auto` architecture selection, `llm_session_topics.py` reuses model-specific chunk caps, `run_pipeline.py` surfaces the resolved defaults, and `benchmark_local_models.py` supports `--pipeline-architecture auto`
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
- **Current evidence:** a full-benchmark run with `gpt-5.4-mini` improves medium/hard classification over the local 14B baseline, which supports continuing Phase 4 exploration, but the current prompt/rubric stack is still the main bottleneck
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
| Phase 3     | ~93-95% if model-limited | ~85-90% if model-limited | Free (local 14B+) |
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
| 2026-03-22 | Deterministic shortcuts (2.3) | Pre-LLM shortcuts for greetings/thanks, vote announcements, name-calls, floor responses; post-Layer-A committee report detection and session chair bias; 26 tests added |
| 2026-03-22 | Tighten QA triggers (2.4) | `low_confidence` threshold 0.65→0.70; `very_short_speech` OR→AND + suppressed by deterministic candidates; audit showed 52.5% trigger rate, now well under 30% |
| 2026-03-22 | Step 3.1 scaffolding | Added shared local-model profiles, benchmark harness, `Modelfile-qwen2.5-14b-32k`, `Modelfile-qwen3-14b-32k`; switched default local model to `qwen3:14b` |
| 2026-03-28 | Phase 3 narrowed to gated screening | Avoid multi-hour end-to-end benchmarks per model; first prove a model upgrade helps on medium/hard cases, then run a full benchmark only for finalists |
| 2026-03-28 | `qwen3:14b` quick screen recorded | On 26 medium/hard speeches from 5 available gold sessions: 61.54% classification, 0% law attribution, ~100 min runtime; not enough evidence to expand local-model benchmarking yet |
| 2026-03-29 | Step 3.2 implemented | Added shared runtime model config (`architecture`, `num_ctx`, `chunk_chars`), switched intervention classifier default to profile-driven `auto`, reused chunk caps in session topic extraction, and enabled `auto` in the benchmark harness |
| 2026-04-10 | Benchmark harness generalized beyond Ollama | Added provider-aware benchmarking for OpenAI and Ollama, mixed-provider model lists, limited-cost smoke benchmark mode, and GPT-specific request handling (`max_completion_tokens` for GPT-5-family models) |
| 2026-04-10 | Full medium/hard gold benchmarking unblocked | Benchmark harness now auto-imports missing gold sessions from `input/stenograme/`, so full gold-session benchmarking no longer depends on the current main DB contents |
| 2026-04-10 | Benchmark reporting improved | `state/model_benchmarks/summary.json` now keeps append-only run history with timestamps, and `benchmark_report.json` mismatch payloads now store full `text_preview` and `pred_reasoning` instead of truncated snippets |
| 2026-04-10 | `gpt-5.4-mini` benchmark recorded | On 89 medium/hard speeches across 17 sessions: 70.79% classification, 14.3% exact law attribution, 21.4% exact-or-partial law attribution, 34.0% topic F1; clear improvement over local 14B quick screen, but still far from target |

---

*Last updated: 2026-04-10*
