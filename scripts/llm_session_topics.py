#!/usr/bin/env python3
"""
LLM-based session topic extraction — map-reduce pipeline.

For each session:

  1. Fetch all substantive chunks (>=200 chars) from session_chunks.
  2. MAP: group consecutive chunks into windows up to MAX_WINDOW_CHARS; call LLM on each
     to produce a free-form bullet list of specific subjects.
  3. REDUCE: send all window bullet lists to the LLM; it merges, deduplicates,
     and structures them into rich topic objects:
       {"label": "...", "description": "...", "law_id": "..." | null}
  4. Store result via MCP store_session_topics (source='llm_v1:{model}').

Why map-reduce?
- Coverage: every part of the session is read, not just 20 sampled chunks.
- Quality: small windows (~5 chunks) keep each map call focused and fast.
- Rich topics: law IDs are paired with descriptions so they are useful for
  grounding constructiveness classification in the intervention pass.

Supports the same --provider / --model flags as llm_agent.py.

Usage:
  python3 scripts/llm_session_topics.py \\
      --run-id run_xyz \\
      --stenogram-list-path state/run_inputs/run_xyz_stenograms.json

  # Single session:
  python3 scripts/llm_session_topics.py --session-id 8846 --run-id run_xyz

  # All sessions without LLM topics:
  python3 scripts/llm_session_topics.py --run-id run_xyz
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agenda import extract_agenda_from_session
from init_db import DEFAULT_DB_PATH, init_db
from law_ids import allowed_law_ids, extract_law_id_index_from_speeches, keep_only_allowed_law_id
from mcp_server import MCPServer
from prompt_logger import EXTERNAL_OUTPUTS_DIR, save_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_OLLAMA = "qwen2.5:7b-32k"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
EST_CHARS_PER_TOKEN = 3.8
LOG_TOKEN_USAGE_PER_CALL = False

# Single-pass mode: each chunk is capped at this length before being concatenated
# into one large prompt. This is different from MAP_CHUNK_CHARS (used in map-reduce)
# because single-pass needs to fit the ENTIRE session in one context window.
# qwen2.5:7b-32k: 32,768 tokens × 80% = ~26,200 usable tokens.
# Reserve ~600 tokens for the system prompt and ~2,048 for output → ~23,500 tokens for session text.
# 23,500 tokens × 3.5 chars/token (Romanian text is denser than English) ≈ 82,000 chars budget.
# Divide by max observed substantive chunks (142) → ~578 chars/chunk cap.
# Use 600 chars as a round number — enough for most speech summaries while fitting the budget.
SINGLE_PASS_CHUNK_CHARS = 600  # per-chunk cap in single-pass mode
# Total budget for single-pass: if capped total > this, fall back to map-reduce.
LARGE_CTX_THRESHOLD_CHARS = 82_000  # ~23k tokens — 80% of qwen2.5:7b-32k's 32k context

# Map-reduce parameters (used for small-context models or very large sessions).
# Budget: 8,192 ctx × 80% = 6,554 usable tokens × ~4 chars/token ≈ 26,000 chars.
# Subtract system prompt (~400 tokens = ~1,600 chars) and map output (~512 tokens = ~2,000 chars).
# Net window budget: ~22,000 chars ≈ 5,500 tokens — safely within 80% of the 8k ctx window.
MAX_WINDOW_CHARS = 22000
# Individual chunk text is capped so one very long speech cannot monopolise a window.
# 4,000 chars ≈ 1,000 tokens — enough to capture a speech's full argument while still
# allowing several speeches per window.
MAX_CHUNK_CHARS = 4000

MAX_TOPICS_PER_SESSION = 20
# 120 chars gives room for specific Romanian labels with law numbers and locations
# without truncating mid-word. The prompt explicitly asks for ≤10 words.
MAX_TOPIC_LABEL_LENGTH = 120
MAX_TOPIC_DESC_LENGTH = 200
# Each reduce batch stays under this char limit so it fits in the 8k ctx window.
# REDUCE_SYSTEM (~600 chars) + output (~2,000 chars) leaves ~19,400 chars for input.
MAX_REDUCE_BATCH_CHARS = 19000

MAX_RETRIES = 3
RETRY_DELAY_S = 10
# Hard timeout per LLM request. If Ollama hangs this raises httpx.ReadTimeout
# which the retry loop catches. 300s gives extra headroom for the large single-pass call.
LLM_REQUEST_TIMEOUT_S = 300
# num_ctx for large-context models (qwen2.5:7b-32k). Ollama reads this at the
# top level of the request body so it is passed via extra_body.
OLLAMA_NUM_CTX = 32768
# Fallback num_ctx for small-context models (llama3.1:8b-8k).
OLLAMA_NUM_CTX_SMALL = 8192
TOPICS_MAX_OUTPUT_TOKENS = 3072
RUN_OUTPUTS_DIR = Path("state/run_outputs")

_LLM_USAGE = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "estimated_prompt_tokens": 0,
    "estimated_completion_tokens": 0,
    "calls_with_actual_usage": 0,
}


def _reset_usage_stats() -> None:
    for key in _LLM_USAGE:
        _LLM_USAGE[key] = 0


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / EST_CHARS_PER_TOKEN)))


def _extract_usage_tokens(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None

    def pick(container, keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = None
            if isinstance(container, dict):
                value = container.get(key)
            else:
                value = getattr(container, key, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    prompt_tokens = pick(usage, ("prompt_tokens", "input_tokens", "prompt_eval_count"))
    completion_tokens = pick(usage, ("completion_tokens", "output_tokens", "eval_count"))
    return prompt_tokens, completion_tokens


def _record_llm_usage(
    response,
    system_prompt: str,
    user_prompt: str,
    output_text: str,
    call_label: str = "",
) -> None:
    prompt_tokens, completion_tokens = _extract_usage_tokens(response)
    est_prompt = _estimate_tokens_from_text(system_prompt) + _estimate_tokens_from_text(user_prompt)
    est_completion = _estimate_tokens_from_text(output_text)

    _LLM_USAGE["calls"] += 1
    _LLM_USAGE["estimated_prompt_tokens"] += est_prompt
    _LLM_USAGE["estimated_completion_tokens"] += est_completion

    if prompt_tokens is not None and completion_tokens is not None:
        _LLM_USAGE["calls_with_actual_usage"] += 1
        _LLM_USAGE["prompt_tokens"] += prompt_tokens
        _LLM_USAGE["completion_tokens"] += completion_tokens

    if LOG_TOKEN_USAGE_PER_CALL:
        label = call_label or "call"
        if prompt_tokens is not None and completion_tokens is not None:
            total = prompt_tokens + completion_tokens
            print(f"    [tokens] {label}: prompt={prompt_tokens} completion={completion_tokens} total={total}")
        else:
            total_est = est_prompt + est_completion
            print(f"    [tokens~] {label}: prompt~={est_prompt} completion~={est_completion} total~={total_est}")


def _usage_summary_payload() -> dict:
    return {
        "calls": int(_LLM_USAGE["calls"]),
        "calls_with_actual_usage": int(_LLM_USAGE["calls_with_actual_usage"]),
        "prompt_tokens": int(_LLM_USAGE["prompt_tokens"]),
        "completion_tokens": int(_LLM_USAGE["completion_tokens"]),
        "estimated_prompt_tokens": int(_LLM_USAGE["estimated_prompt_tokens"]),
        "estimated_completion_tokens": int(_LLM_USAGE["estimated_completion_tokens"]),
    }


def _safe_filename_part(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (text or ""))
    return out.strip("_") or "unknown"


def _save_failed_llm_output(
    *,
    run_id: str,
    session_id: str,
    session_date: str,
    model: str,
    stage: str,
    error: str,
    raw_text: str,
) -> Path:
    """Persist malformed or unrecoverable LLM output for post-mortem debugging."""
    RUN_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_part = _safe_filename_part(run_id or "manual")
    session_part = _safe_filename_part(str(session_id))
    model_part = _safe_filename_part(model.replace(":", "-"))
    stage_part = _safe_filename_part(stage)
    filename = f"run_{run_part}_session_topics_error_{session_part}_{ts}_{model_part}_{stage_part}.txt"
    out_path = RUN_OUTPUTS_DIR / filename
    content = (
        "=== METADATA ===\n"
        f"run_id       : {run_id or 'manual'}\n"
        f"session_id   : {session_id}\n"
        f"session_date : {session_date}\n"
        f"model        : {model}\n"
        f"stage        : {stage}\n"
        f"error        : {error}\n"
        f"saved_at     : {ts}\n\n"
        "=== RAW LLM OUTPUT ===\n"
        f"{raw_text}\n"
    )
    out_path.write_text(content, encoding="utf-8")
    return out_path

# Topic catalog config used to constrain LLM output to known canonical topics.
TOPIC_TAXONOMY_CONFIG_PATH = Path("config/topic_taxonomy.json")

_TOPIC_CATALOG_CACHE: list[dict] | None = None


def _load_topic_catalog(config_path: Path = TOPIC_TAXONOMY_CONFIG_PATH) -> list[dict]:
    """Load canonical topic catalog from config.

    Preferred source:
      config.topic_taxonomy.json -> "catalog_topics": [{id, label, description, aliases}]

    Fallback when catalog_topics is missing:
      Derive a coarse catalog from direction_rules labels.
    """
    global _TOPIC_CATALOG_CACHE
    if _TOPIC_CATALOG_CACHE is not None:
        return _TOPIC_CATALOG_CACHE

    catalog: list[dict] = []
    try:
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw = payload.get("catalog_topics", [])
                if isinstance(raw, list):
                    for item in raw:
                        if not isinstance(item, dict):
                            continue
                        label = str(item.get("label", "")).strip()
                        topic_id = str(item.get("id", "")).strip()
                        if not label or not topic_id:
                            continue
                        aliases_raw = item.get("aliases", [])
                        aliases = [str(a).strip() for a in aliases_raw if isinstance(a, str) and str(a).strip()]
                        catalog.append(
                            {
                                "id": topic_id,
                                "label": label,
                                "description": str(item.get("description", "")).strip(),
                                "aliases": aliases,
                            }
                        )

                # Fallback: build from direction rules if explicit catalog is absent.
                if not catalog:
                    rules = payload.get("direction_rules", [])
                    if isinstance(rules, list):
                        for idx, rule in enumerate(rules, 1):
                            if not isinstance(rule, dict):
                                continue
                            label = str(rule.get("label", "")).strip()
                            if not label:
                                continue
                            topic_id = f"dir_{idx:02d}"
                            catalog.append(
                                {
                                    "id": topic_id,
                                    "label": label,
                                    "description": "Direcție tematică din taxonomia locală.",
                                    "aliases": [],
                                }
                            )
    except Exception:
        catalog = []

    _TOPIC_CATALOG_CACHE = catalog
    return catalog


def _format_topic_catalog(catalog: list[dict]) -> str:
    if not catalog:
        return "Catalog topics: (none provided)"
    lines = ["Canonical topic catalog (prefer matching these):"]
    for item in catalog:
        aliases = item.get("aliases", [])
        aliases_text = f" | aliases: {', '.join(aliases[:8])}" if aliases else ""
        desc = item.get("description", "")
        desc_text = f" | desc: {desc}" if desc else ""
        lines.append(f"- {item['id']} | {item['label']}{desc_text}{aliases_text}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Prompts — map-reduce
# ---------------------------------------------------------------------------

# MAP prompt: free-form bullet list from a small window of chunks.
# Romanian so the model stays in-language and avoids translation loss.
WINDOW_SYSTEM = """Ești un analist expert al ședințelor Parlamentului României.

Citește fragmentele de stenogramă de mai jos și listează TOATE subiectele specifice menționate.

Formatul răspunsului — o listă de puncte, câte un subiect pe linie, DOAR cu informații din textul de mai jos:
- [subiect specific din text — sumă/locație/instituție dacă există]
- [proiect de lege din text — număr și scurt titlu]
- [altă temă specifică din text]

Reguli:
- Fii SPECIFIC: menționează sumele, locațiile, instituțiile, numerele de lege/proiect din text.
- Dacă apare un număr de lege sau proiect (ex. PL-x 45/2025, OUG 114/2018, nr. 360/2023), include-l.
- NU scrie generic: NU "buget", NU "educație", NU "infrastructură" singure.
- Dacă fragmentele nu conțin subiecte clare, scrie "- (fără subiecte identificate)".
- Răspunde DOAR cu lista de puncte, fără titlu sau introducere."""

# REDUCE prompt: merge all window bullet lists into structured JSON topic objects.
REDUCE_SYSTEM_BASE = """You are extracting a structured topic list from a Romanian parliamentary session.

You will receive bullet-list summaries from multiple windows covering the full session.
Your job: merge them into a deduplicated list of up to 20 rich topic objects.

You are given a canonical topic catalog. Use it as constraints:
- First try to match each topic to ONE catalog topic.
- If no catalog topic fits, emit it as a new topic with explicit reason.

Output format — respond with ONLY valid JSON, no prose, no markdown:
{
  "matched_topics": [
    {
      "catalog_topic_id": "ID_FROM_CATALOG",
      "label": "CANONICAL_LABEL_FROM_CATALOG",
      "description": "DESCRIPTION_FROM_INPUT_ONE_SENTENCE",
      "law_id": "LAW_ID_FROM_INPUT_OR_NULL",
      "confidence": 0.0
    }
  ],
  "new_topics": [
    {
      "label": "NEW_LABEL_MAX_10_WORDS",
      "description": "DESCRIPTION_FROM_INPUT_ONE_SENTENCE",
      "law_id": "LAW_ID_FROM_INPUT_OR_NULL",
      "reason_no_match": "why catalog does not fit",
      "confidence": 0.0
    }
  ],
  "session_summary": "ONE_SENTENCE_SUMMARY_FROM_INPUT"
}

Rules:
- label: MAXIMUM 10 words, in Romanian, specific to this session. If a law/bill is the topic, use its short title + identifier (e.g. "Reforma pensiilor speciale PL-x 45/2025"). Never cut a label mid-sentence.
- description: one sentence in Romanian. Include what the topic is about and all concrete details present in input (amounts, locations, institutions, percentages). Do NOT invent.
- law_id: bill/law/ordinance identifier when present, otherwise null.
- CRITICAL law_id constraint: if you include a law_id, it MUST be copied verbatim from the "Pre-extracted law IDs" list provided in the session header. Otherwise set law_id to null.
- COMBINING duplicates: merge overlapping mentions into one topic and keep all details.
- Prefer `matched_topics`; use `new_topics` only when no catalog fit exists.
- Keep total matched_topics + new_topics <= 20.
- Do NOT include "(fără subiecte identificate)" entries.
- Keep output compact to avoid truncation:
  - target 8-12 topics (maximum remains 20 only if clearly needed)
  - `description`: max 24 words
  - `session_summary`: max 35 words
  - `reason_no_match`: max 12 words
- JSON safety:
  - return exactly one JSON object and nothing else
  - all string values must be closed
  - escape internal double quotes as \\"
  - do not include raw newlines inside string values (use spaces)"""


def _build_reduce_system(catalog: list[dict]) -> str:
    return REDUCE_SYSTEM_BASE + "\n\n" + _format_topic_catalog(catalog)


def _build_window_message(session_header: str, window_chunks: list[dict], window_num: int, total_windows: int) -> str:
    """Build the user message for one map window."""
    parts = [
        f"{session_header}  [Fereastră {window_num}/{total_windows}]",
        "Fragmente:",
    ]
    for i, chunk in enumerate(window_chunks, 1):
        parts.append(f"[{i}] {chunk['text'].strip()}")
    parts.append("Listează subiectele specifice din fragmentele de mai sus, câte unul pe linie.")
    return "\n\n".join(parts)


def _build_reduce_message(session_header: str, window_results: list[str]) -> str:
    """Build the user message for the reduce step."""
    parts = [session_header, "Window summaries from across the full session:"]
    for i, prose in enumerate(window_results, 1):
        parts.append(f"--- Window {i} ---\n{prose.strip()}")
    parts.append("Merge the above into the structured JSON topic list.")
    return "\n\n".join(parts)


def _build_law_ids_context(law_id_index: dict[str, list[int]]) -> str:
    if not law_id_index:
        return ""
    lines = ["Pre-extracted law IDs from this session (use only these IDs):"]
    for law_id, speech_indices in law_id_index.items():
        indices_text = ", ".join(str(i) for i in sorted(set(speech_indices))[:12])
        if indices_text:
            lines.append(f"- {law_id} [speech_index: {indices_text}]")
        else:
            lines.append(f"- {law_id}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM calls (map-reduce)
# ---------------------------------------------------------------------------

def _build_client(provider: str, model: str):
    import openai as openai_module

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )
        print(f"Provider: OpenAI  |  Model: {model}")
        client = openai_module.OpenAI(api_key=api_key)
        client._model = model
        client._provider = "openai"
        return client

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
        base_url = f"{host}/v1"
        print(f"Provider: Ollama ({host})  |  Model: {model}")
        import urllib.request
        import urllib.error
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3)
        except urllib.error.URLError:
            raise SystemExit(
                f"Cannot reach Ollama at {host}.\n"
                "Start it with: ollama serve"
            )
        client = openai_module.OpenAI(api_key="ollama", base_url=base_url)
        client._model = model
        client._provider = "ollama"
        return client

    raise SystemExit(f"Unknown provider: {provider!r}. Choose 'openai' or 'ollama'.")


class _BuildPromptsOnly(Exception):
    """Raised inside _chat() when build_prompts_only=True to skip the LLM call."""


def _chat(
    client,
    system: str,
    user: str,
    json_mode: bool = False,
    max_tokens: int = 512,
    prompt_label: str = "",
    session_id: str = "",
    session_date: str = "",
    build_prompts_only: bool = False,
) -> str:
    """Single LLM call. Returns raw string content.

    When ``build_prompts_only=True`` the prompt is saved with a stable
    ``"draft"`` timestamp and ``_BuildPromptsOnly`` is raised instead of
    calling the LLM.  Callers must catch this exception.
    """
    if session_id:
        try:
            ts = "draft" if build_prompts_only else None
            save_prompt(
                step="session_topics",
                session_id=session_id,
                session_date=session_date,
                model=client._model,
                label=prompt_label or "call",
                system_prompt=system,
                user_message=user,
                extra_meta={"max_tokens": max_tokens, "json_mode": json_mode},
                timestamp=ts,
            )
        except Exception:
            pass  # Never let logging block the actual LLM call.

    if build_prompts_only:
        raise _BuildPromptsOnly(prompt_label or "call")

    extra_kwargs: dict = {}
    if json_mode:
        extra_kwargs["response_format"] = {"type": "json_object"}
    # For Ollama, explicitly set num_ctx so session prompts are not silently
    # truncated. Ollama reads num_ctx at the top level of the request body
    # (not nested under "options"), so extra_body merges it correctly.
    if getattr(client, "_provider", "") == "ollama":
        extra_kwargs["extra_body"] = {"num_ctx": OLLAMA_NUM_CTX}
    response = client.chat.completions.create(
        model=client._model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0 if json_mode else 0.2,
        max_tokens=max_tokens,
        timeout=LLM_REQUEST_TIMEOUT_S,
        **extra_kwargs,
    )
    content = response.choices[0].message.content or ""
    _record_llm_usage(response, system, user, content, call_label=prompt_label)
    return content


def _group_into_windows(chunks: list[dict], max_chars: int) -> list[list[dict]]:
    """Group consecutive chunks greedily up to max_chars total text.

    Each chunk's text is already capped at MAX_CHUNK_CHARS before this is called.
    Starting a new window when adding the next chunk would exceed the budget keeps
    related speeches together (better topic signal) while bounding token use.
    A chunk that is already at the cap becomes its own single-chunk window.
    """
    windows: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for chunk in chunks:
        chunk_len = len(chunk["text"])
        if current and current_chars + chunk_len > max_chars:
            windows.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += chunk_len
    if current:
        windows.append(current)
    return windows


def _batch_prose(prose_list: list[str], max_chars: int) -> list[list[str]]:
    """Group a flat list of prose strings into batches that fit within max_chars total."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for prose in prose_list:
        if current and current_chars + len(prose) > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(prose)
        current_chars += len(prose)
    if current:
        batches.append(current)
    return batches


def _to_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))


def _parse_topics_payload(raw: str) -> dict:
    """Parse and normalise LLM response.

    Supports both:
    - legacy format: {"topics": [...], "session_summary": "..."}
    - constrained format: {"matched_topics": [...], "new_topics": [...], "session_summary": "..."}

    Returns:
      {
        "topics": [...],          # flattened enriched list for DB storage
        "matched_topics": [...],  # canonical matches
        "new_topics": [...],      # proposed new topics
        "session_summary": "..."
      }
    """
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```")).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")

    matched_raw = parsed.get("matched_topics")
    new_raw = parsed.get("new_topics")
    topics_raw = parsed.get("topics")
    session_summary = str(parsed.get("session_summary", "")).strip()

    # Legacy mode: convert topics -> matched_topics (without catalog id).
    if matched_raw is None and new_raw is None:
        matched_raw = topics_raw
        new_raw = []

    if not isinstance(matched_raw, list):
        raise ValueError("'matched_topics' is not a list")
    if not isinstance(new_raw, list):
        raise ValueError("'new_topics' is not a list")

    seen: set[str] = set()
    matched_topics: list[dict] = []
    new_topics: list[dict] = []
    topics_flat: list[dict] = []

    for item in matched_raw:
        if isinstance(item, str):
            item = {"label": item}
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()[:MAX_TOPIC_LABEL_LENGTH]
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        entry = {
            "catalog_topic_id": str(item.get("catalog_topic_id", "")).strip() or None,
            "label": label,
            "description": str(item.get("description", "")).strip()[:MAX_TOPIC_DESC_LENGTH],
            "law_id": item.get("law_id") or None,
            "confidence": _to_confidence(item.get("confidence", 0.0)),
            "match_type": "catalog",
            "is_new_topic": False,
            "reason_no_match": "",
        }
        matched_topics.append(entry)
        topics_flat.append(entry.copy())
        if len(topics_flat) >= MAX_TOPICS_PER_SESSION:
            break

    if len(topics_flat) < MAX_TOPICS_PER_SESSION:
        for item in new_raw:
            if isinstance(item, str):
                item = {"label": item}
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()[:MAX_TOPIC_LABEL_LENGTH]
            if not label or label.lower() in seen:
                continue
            seen.add(label.lower())
            entry = {
                "catalog_topic_id": None,
                "label": label,
                "description": str(item.get("description", "")).strip()[:MAX_TOPIC_DESC_LENGTH],
                "law_id": item.get("law_id") or None,
                "confidence": _to_confidence(item.get("confidence", 0.0)),
                "match_type": "new_topic",
                "is_new_topic": True,
                "reason_no_match": str(item.get("reason_no_match", "")).strip()[:MAX_TOPIC_DESC_LENGTH],
            }
            new_topics.append(entry)
            topics_flat.append(entry.copy())
            if len(topics_flat) >= MAX_TOPICS_PER_SESSION:
                break

    return {
        "topics": topics_flat,
        "matched_topics": matched_topics,
        "new_topics": new_topics,
        "session_summary": session_summary,
    }


def _repair_topics_json(
    client,
    raw_text: str,
    label: str,
    session_id: str = "",
    session_date: str = "",
    run_id: str = "",
    build_prompts_only: bool = False,
) -> dict | None:
    """Attempt to repair malformed topic JSON without re-running full extraction."""
    system = """You repair malformed JSON output for Romanian parliamentary session topics.

Return ONLY valid JSON object in this schema:
{
  "matched_topics": [
    {
      "catalog_topic_id": "string or null",
      "label": "string",
      "description": "string",
      "law_id": "string or null",
      "confidence": 0.0
    }
  ],
  "new_topics": [
    {
      "label": "string",
      "description": "string",
      "law_id": "string or null",
      "reason_no_match": "string",
      "confidence": 0.0
    }
  ],
  "session_summary": "string"
}

Rules:
- Keep original content; only repair syntax/structure.
- Do NOT invent new facts.
- If a field is missing, use null/empty string/empty list appropriately.
- Output JSON only, no markdown fences."""
    user = (
        "Repair this malformed JSON into valid schema-compliant JSON.\n\n"
        "Malformed output:\n"
        f"{raw_text}"
    )
    try:
        repaired_raw = _chat(
            client=client,
            system=system,
            user=user,
            json_mode=True,
            max_tokens=3072,
            prompt_label=f"{label}_json_repair",
            session_id=session_id,
            session_date=session_date,
            build_prompts_only=build_prompts_only,
        )
        return _parse_topics_payload(repaired_raw)
    except Exception as exc:
        print(f"  JSON repair failed ({label}): {exc}")
        try:
            path = _save_failed_llm_output(
                run_id=run_id,
                session_id=session_id,
                session_date=session_date,
                model=getattr(client, "_model", ""),
                stage=f"{label}_repair_failed",
                error=str(exc),
                raw_text=raw_text,
            )
            print(f"  Saved failed raw output: {path}")
        except Exception:
            pass
        return None


def _call_reduce_once(
    client,
    session_header: str,
    prose_list: list[str],
    label: str,
    session_id: str = "",
    session_date: str = "",
    run_id: str = "",
    build_prompts_only: bool = False,
) -> dict:
    """Run one reduce LLM call over a list of prose strings. Returns parsed payload.

    In build_prompts_only mode the prompt is saved and an empty list is returned.
    """
    reduce_msg = _build_reduce_message(session_header, prose_list)
    catalog = _load_topic_catalog()
    reduce_system = _build_reduce_system(catalog)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _chat(
                client, reduce_system, reduce_msg, json_mode=True, max_tokens=TOPICS_MAX_OUTPUT_TOKENS,
                prompt_label=label.lower().replace(" ", "_"),
                session_id=session_id, session_date=session_date,
                build_prompts_only=build_prompts_only,
            )
            try:
                payload = _parse_topics_payload(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"  {label}: parse failed, attempting JSON repair: {exc}")
                try:
                    path = _save_failed_llm_output(
                        run_id=run_id,
                        session_id=session_id,
                        session_date=session_date,
                        model=getattr(client, "_model", ""),
                        stage=f"{label.lower().replace(' ', '_')}_parse_failed_raw",
                        error=str(exc),
                        raw_text=raw,
                    )
                    print(f"  Saved failed raw output: {path}")
                except Exception:
                    pass
                repaired = _repair_topics_json(
                    client=client,
                    raw_text=raw,
                    label=label.lower().replace(" ", "_"),
                    session_id=session_id,
                    session_date=session_date,
                    run_id=run_id,
                    build_prompts_only=build_prompts_only,
                )
                if repaired is None:
                    try:
                        path = _save_failed_llm_output(
                            run_id=run_id,
                            session_id=session_id,
                            session_date=session_date,
                            model=getattr(client, "_model", ""),
                            stage=f"{label.lower().replace(' ', '_')}_parse_failed",
                            error=str(exc),
                            raw_text=raw,
                        )
                        print(f"  Saved failed raw output: {path}")
                    except Exception:
                        pass
                    raise
                payload = repaired
                print(f"  {label}: JSON repair succeeded")
            topics = payload.get("topics", [])
            print(f"  {label}: {len(topics)} topics from {len(prose_list)} inputs ({len(reduce_msg)} chars)")
            return payload
        except _BuildPromptsOnly:
            print(f"  {label}: prompt saved (build-prompts mode)")
            return {"topics": [], "matched_topics": [], "new_topics": [], "session_summary": ""}
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  {label} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    return {"topics": [], "matched_topics": [], "new_topics": [], "session_summary": ""}


def _reduce(
    client,
    session_header: str,
    window_results: list[str],
    session_id: str = "",
    session_date: str = "",
    run_id: str = "",
    build_prompts_only: bool = False,
) -> dict:
    """Two-stage reduce.

    Stage 1: if all window prose fits in one batch → single reduce call (common case).
             Otherwise split into batches of MAX_REDUCE_BATCH_CHARS and reduce each
             batch to an intermediate JSON topic list serialised back to prose.
    Stage 2: reduce all intermediate lists into the final deduplicated 20 topics.
    """
    batches = _batch_prose(window_results, MAX_REDUCE_BATCH_CHARS)
    total_chars = sum(len(p) for p in window_results)
    print(f"  Reduce: {len(window_results)} window(s), {total_chars} chars → {len(batches)} batch(es)")

    if len(batches) == 1:
        payload = _call_reduce_once(
            client, session_header, batches[0], "Reduce",
            session_id=session_id, session_date=session_date,
            run_id=run_id,
            build_prompts_only=build_prompts_only,
        )
        return payload

    # Stage 1: reduce each batch to an intermediate topic list.
    intermediate_prose: list[str] = []
    for b_idx, batch in enumerate(batches, 1):
        partial_payload = _call_reduce_once(
            client, session_header, batch, f"Reduce stage-1 batch {b_idx}/{len(batches)}",
            session_id=session_id, session_date=session_date,
            run_id=run_id,
            build_prompts_only=build_prompts_only,
        )
        partial = partial_payload.get("topics", [])
        if partial:
            lines = []
            for t in partial:
                law = f" ({t['law_id']})" if t.get("law_id") else ""
                desc = f": {t['description']}" if t.get("description") else ""
                lines.append(f"- {t['label']}{law}{desc}")
            intermediate_prose.append("\n".join(lines))

    # Stage 2: merge all intermediate lists into the final result.
    final_payload = _call_reduce_once(
        client, session_header, intermediate_prose, "Reduce stage-2 final",
        session_id=session_id, session_date=session_date,
        run_id=run_id,
        build_prompts_only=build_prompts_only,
    )
    return final_payload


SINGLE_PASS_SYSTEM_BASE = """You are an expert analyst of Romanian Parliament plenary sessions.

You will receive the full transcript of one session as a numbered list of speech fragments.
Your task: read the entire text and produce a structured list of up to 20 specific topics debated.

You are given a canonical topic catalog. Use it as constraints:
- First match each topic to ONE catalog topic.
- If there is no adequate match, emit as `new_topics` with reason.

Output format — respond with ONLY valid JSON, no prose, no markdown fences:
{
  "matched_topics": [
    {
      "catalog_topic_id": "ID_FROM_CATALOG",
      "label": "CANONICAL_LABEL_FROM_CATALOG",
      "description": "ONE_SENTENCE_IN_ROMANIAN_WITH_ALL_DETAILS",
      "law_id": "IDENTIFIER_OR_NULL",
      "confidence": 0.0
    }
  ],
  "new_topics": [
    {
      "label": "NEW_LABEL_MAX_10_WORDS_IN_ROMANIAN",
      "description": "ONE_SENTENCE_IN_ROMANIAN_WITH_ALL_DETAILS",
      "law_id": "IDENTIFIER_OR_NULL",
      "reason_no_match": "why catalog does not fit",
      "confidence": 0.0
    }
  ],
  "session_summary": "ONE_SENTENCE_OVERVIEW"
}

Rules:
- label: maximum 10 words in Romanian. If a law/bill is the topic include its identifier.
- description: one sentence in Romanian with concrete details present in text.
- law_id: bill/law/ordinance identifier if present, otherwise null.
- CRITICAL law_id constraint: if you include a law_id, it MUST be copied verbatim from the "Pre-extracted law IDs" list provided in the session header. Otherwise set law_id to null.
- Prefer `matched_topics`; use `new_topics` only when no catalog fit exists.
- Keep matched_topics + new_topics <= 20.
- CRITICAL: every field MUST be grounded in the text. Do NOT invent details.
- Keep output compact to avoid truncation:
  - target 8-12 topics (maximum remains 20 only if clearly needed)
  - `description`: max 24 words
  - `session_summary`: max 35 words
  - `reason_no_match`: max 12 words
- JSON safety:
  - return exactly one JSON object and nothing else
  - all string values must be closed
  - escape internal double quotes as \\"
  - do not include raw newlines inside string values (use spaces)"""


def _build_single_pass_system(catalog: list[dict]) -> str:
    return SINGLE_PASS_SYSTEM_BASE + "\n\n" + _format_topic_catalog(catalog)


def _build_single_pass_message(session_header: str, chunks: list[dict]) -> str:
    """Build a single large user message containing all session chunks."""
    parts = [session_header, "Full session transcript (all fragments):"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['text'].strip()}")
    parts.append("Extract the structured topic list from the full transcript above.")
    return "\n\n".join(parts)


def _call_single_pass(
    client,
    session_header: str,
    chunks: list[dict],
    session_id: str = "",
    session_date: str = "",
    run_id: str = "",
    build_prompts_only: bool = False,
) -> dict:
    """Single-pass topic extraction for large-context models.

    Sends the entire session in one LLM call. Only used when the model's
    context window is large enough (signalled by OLLAMA_NUM_CTX >= 32768)
    and the session text fits within LARGE_CTX_THRESHOLD_CHARS.
    """
    user_msg = _build_single_pass_message(session_header, chunks)
    catalog = _load_topic_catalog()
    single_pass_system = _build_single_pass_system(catalog)
    total_chars = len(user_msg)
    print(f"  Single-pass: {len(chunks)} chunks, {total_chars:,} chars in one call")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _chat(
                client, single_pass_system, user_msg, json_mode=True, max_tokens=TOPICS_MAX_OUTPUT_TOKENS,
                prompt_label="single_pass",
                session_id=session_id, session_date=session_date,
                build_prompts_only=build_prompts_only,
            )
            try:
                payload = _parse_topics_payload(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"  Single-pass parse failed, attempting JSON repair: {exc}")
                try:
                    path = _save_failed_llm_output(
                        run_id=run_id,
                        session_id=session_id,
                        session_date=session_date,
                        model=getattr(client, "_model", ""),
                        stage="single_pass_parse_failed_raw",
                        error=str(exc),
                        raw_text=raw,
                    )
                    print(f"  Saved failed raw output: {path}")
                except Exception:
                    pass
                repaired = _repair_topics_json(
                    client=client,
                    raw_text=raw,
                    label="single_pass",
                    session_id=session_id,
                    session_date=session_date,
                    run_id=run_id,
                    build_prompts_only=build_prompts_only,
                )
                if repaired is None:
                    try:
                        path = _save_failed_llm_output(
                            run_id=run_id,
                            session_id=session_id,
                            session_date=session_date,
                            model=getattr(client, "_model", ""),
                            stage="single_pass_parse_failed",
                            error=str(exc),
                            raw_text=raw,
                        )
                        print(f"  Saved failed raw output: {path}")
                    except Exception:
                        pass
                    raise
                payload = repaired
                print("  Single-pass JSON repair succeeded")
            topics = payload.get("topics", [])
            print(f"  Single-pass result: {len(topics)} topics  (first 120 chars: {raw[:120]!r})")
            return payload
        except _BuildPromptsOnly:
            print(f"  Single-pass: prompt saved (build-prompts mode)")
            return {"topics": [], "matched_topics": [], "new_topics": [], "session_summary": ""}
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  Single-pass attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    return {"topics": [], "matched_topics": [], "new_topics": [], "session_summary": ""}


def _call_map_reduce(
    client,
    session_header: str,
    substantive_chunks: list[dict],
    session_id: str = "",
    session_date: str = "",
    run_id: str = "",
    build_prompts_only: bool = False,
) -> dict:
    """
    Map-reduce topic extraction.

    MAP: for each window of consecutive substantive chunks (up to MAX_WINDOW_CHARS), call the LLM to produce
         a free-form Romanian bullet list of specific subjects.
    REDUCE: send all bullet lists to the LLM; it merges, deduplicates, and structures
            them into rich {label, description, law_id} topic objects (max 20).
    """
    windows = _group_into_windows(substantive_chunks, MAX_WINDOW_CHARS)
    total_windows = len(windows)
    avg_chunks = len(substantive_chunks) / total_windows if total_windows else 0
    print(f"  Map step: {len(substantive_chunks)} chunks → {total_windows} window(s) (avg {avg_chunks:.1f} chunks/window, max {MAX_WINDOW_CHARS} chars)")

    window_results: list[str] = []
    for w_idx, window in enumerate(windows, 1):
        user_msg = _build_window_message(session_header, window, w_idx, total_windows)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prose = _chat(
                    client, WINDOW_SYSTEM, user_msg, max_tokens=512,
                    prompt_label=f"window_{w_idx}of{total_windows}",
                    session_id=session_id, session_date=session_date,
                    build_prompts_only=build_prompts_only,
                )
                window_results.append(prose)
                bullet_count = sum(
                    1 for line in prose.splitlines()
                    if line.lstrip().startswith(("-", "•", "*", "–"))
                )
                print(f"    window {w_idx}/{total_windows}: ~{bullet_count} bullets ({len(prose)} chars): {prose[:120]!r}")
                break
            except _BuildPromptsOnly:
                print(f"    window {w_idx}/{total_windows}: prompt saved (build-prompts mode)")
                break  # move to next window — no retries needed
            except Exception as exc:
                print(f"    window {w_idx} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
                else:
                    window_results.append("")  # Skip window on repeated failure.

    non_empty = [r for r in window_results if r.strip()]
    # In build_prompts_only mode non_empty will be empty, but we still call _reduce
    # so the reduce-step prompts are also generated (with a placeholder input).
    reduce_input = non_empty if non_empty else (["placeholder"] if build_prompts_only else [])
    return _reduce(client, session_header, reduce_input, session_id=session_id,
                   session_date=session_date, run_id=run_id, build_prompts_only=build_prompts_only)


# ---------------------------------------------------------------------------
# Single-session extraction
# ---------------------------------------------------------------------------

def extract_session_topics(
    server: MCPServer,
    session_id: str,
    run_id: str,
    model: str,
    client,
    provider: str,
    conn: sqlite3.Connection,
    build_prompts_only: bool = False,
) -> dict:
    """Run the map-reduce pipeline for one session and store the result via MCP.

    When ``build_prompts_only=True`` the prompt files are written but no LLM
    call is made and nothing is stored to the DB.  Returns ``{"ok": True,
    "prompts_only": True}`` in that case.
    """
    # Fetch session metadata.
    session_result = server.call("get_session", {"session_id": session_id})
    if not session_result["ok"]:
        return session_result
    session = session_result["session"]

    # Fetch all chunks ordered by position.
    all_rows = conn.execute(
        """
        SELECT chunk_id, chunk_type, chunk_index, text
        FROM session_chunks
        WHERE session_id = ?
        ORDER BY chunk_index ASC
        """,
        (session_id,),
    ).fetchall()

    # Separate the session_notes header from the speech pool.
    notes_rows = [r for r in all_rows if r["chunk_type"] == "session_notes"]

    # Keep substantive speeches only (>=200 chars).
    # Short chunks are procedural voting lines that add noise ("vă rog să votați").
    substantive_rows = [
        r for r in all_rows
        if r["chunk_type"] != "session_notes" and len(r["text"]) >= 200
    ]
    # Fall back to all non-notes if the session has very short speeches (rare).
    speech_pool = substantive_rows if len(substantive_rows) >= 5 else [
        r for r in all_rows if r["chunk_type"] != "session_notes"
    ]

    session_speeches = [
        {"speech_index": int(r["chunk_index"]), "text": r["text"]}
        for r in all_rows
        if r["chunk_type"] != "session_notes"
    ]
    law_id_index = extract_law_id_index_from_speeches(session_speeches)
    allowed_ids = allowed_law_ids(law_id_index)

    notes_text = (notes_rows[0]["text"].strip() if notes_rows else "")
    agenda = extract_agenda_from_session(notes_text, session_speeches)
    if agenda:
        print(f"  Agenda: {len(agenda)} item(s) pre-extracted")

    # Build the header that goes at the top of every window message.
    session_header = (
        f"Ședința ID: {session_id}  Data: {session.get('session_date', '')}  "
        f"({len(speech_pool)} fragmente substantive)"
    )
    if notes_text:
        session_header += f"\nNote inițiale: {notes_text[:200]}"
    if agenda:
        agenda_lines = ["Agenda legislativă (pre-extrasă din sesiune):"]
        for item in agenda:
            entry = ""
            item_num = item.get("item_number")
            if item_num is not None:
                entry += f"{item_num}. "
            title = str(item.get("title", "")).strip()
            if title:
                entry += title
            law_id = str(item.get("law_id") or "").strip()
            if law_id:
                entry += f" ({law_id})"
            if entry.strip():
                agenda_lines.append(f"- {entry.strip()}")
        if len(agenda_lines) > 1:
            session_header += "\n" + "\n".join(agenda_lines)
    law_context = _build_law_ids_context(law_id_index)
    if law_context:
        session_header += f"\n{law_context}"

    print(
        f"  session={session_id}  date={session.get('session_date', '')}  "
        f"substantive={len(speech_pool)}/{len(all_rows)} chunks"
    )

    # For large-context models (OLLAMA_NUM_CTX >= 32768) send the full session in
    # one call when the total text fits within LARGE_CTX_THRESHOLD_CHARS.
    # For small-context models or very large sessions fall back to map-reduce.
    is_large_ctx = getattr(client, "_provider", "") != "ollama" or OLLAMA_NUM_CTX >= 32768
    full_text_chars = sum(len(r["text"]) for r in speech_pool)

    if is_large_ctx:
        # Single-pass: cap each chunk at SINGLE_PASS_CHUNK_CHARS so the whole session
        # fits in the 32k context window, then check the capped total.
        single_pass_chunks = [
            {
                "chunk_id": r["chunk_id"],
                "chunk_type": r["chunk_type"],
                "text": r["text"][:SINGLE_PASS_CHUNK_CHARS] + ("…" if len(r["text"]) > SINGLE_PASS_CHUNK_CHARS else ""),
            }
            for r in speech_pool
        ]
        capped_total = sum(len(c["text"]) for c in single_pass_chunks)
        session_date = session.get("session_date", "")
        if capped_total <= LARGE_CTX_THRESHOLD_CHARS:
            print(f"  Mode: single-pass (large-ctx model, capped total {capped_total:,} chars ≤ {LARGE_CTX_THRESHOLD_CHARS:,})")
            llm_data = _call_single_pass(
                client, session_header, single_pass_chunks,
                session_id=session_id, session_date=session_date,
                run_id=run_id,
                build_prompts_only=build_prompts_only,
            )
        else:
            print(f"  Mode: map-reduce (capped total {capped_total:,} chars > {LARGE_CTX_THRESHOLD_CHARS:,}, too many chunks)")
            substantive_chunks = [
                {
                    "chunk_id": r["chunk_id"],
                    "chunk_type": r["chunk_type"],
                    "text": r["text"][:MAX_CHUNK_CHARS] + ("…" if len(r["text"]) > MAX_CHUNK_CHARS else ""),
                }
                for r in speech_pool
            ]
            llm_data = _call_map_reduce(
                client, session_header, substantive_chunks,
                session_id=session_id, session_date=session_date,
                run_id=run_id,
                build_prompts_only=build_prompts_only,
            )
    else:
        session_date = session.get("session_date", "")
        # Map-reduce: small-context model.
        print(f"  Mode: map-reduce (small-ctx model)")
        substantive_chunks = [
            {
                "chunk_id": r["chunk_id"],
                "chunk_type": r["chunk_type"],
                "text": r["text"][:MAX_CHUNK_CHARS] + ("…" if len(r["text"]) > MAX_CHUNK_CHARS else ""),
            }
            for r in speech_pool
        ]
        llm_data = _call_map_reduce(
            client, session_header, substantive_chunks,
            session_id=session_id, session_date=session_date,
            run_id=run_id,
            build_prompts_only=build_prompts_only,
        )

    if build_prompts_only:
        return {"ok": True, "prompts_only": True}

    topics = llm_data.get("topics", [])
    if allowed_ids:
        for topic in topics:
            if isinstance(topic, dict):
                topic["law_id"] = keep_only_allowed_law_id(topic.get("law_id"), allowed_ids)
    matched_topics = llm_data.get("matched_topics", [])
    new_topics = llm_data.get("new_topics", [])
    session_summary = llm_data.get("session_summary", "")
    print(f"  session_summary={session_summary!r}")
    print(f"  matched_topics={len(matched_topics)}  new_topics={len(new_topics)}")
    for t in topics:
        print(f"    - {t.get('label')} | {t.get('description', '')[:80]} | law_id={t.get('law_id')}")

    topics_source = f"llm_v1:{client._model}"
    store_result = server.call(
        "store_session_topics",
        {"session_id": session_id, "topics": topics, "topics_source": topics_source},
    )
    return store_result


# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------

def _load_all_session_ids(db_path: Path) -> list[str]:
    """Return ALL session IDs present in the DB, regardless of processing state.

    Used by --build-prompts so that every session gets a prompt file, not just
    the ones that are still pending LLM work.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM session_chunks ORDER BY session_id"
        ).fetchall()
    return [r["session_id"] for r in rows]


def _load_session_ids(
    db_path: Path,
    stenogram_list_path: Path | None,
    model: str,
    reprocess: bool = False,
) -> list[str]:
    """
    Return session IDs that need LLM topic extraction.

    Default (reprocess=False): skip any session already processed by *any* LLM
      (topics_source LIKE 'llm_v1:%'), regardless of which model was used.
      This is the safe default — don't redo expensive LLM work unless asked.

    reprocess=True: only skip sessions processed by the *same* model
      (topics_source = 'llm_v1:{model}'). Use --reprocess-session-topics to
      force re-extraction with a different or updated model.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if reprocess:
            # Skip only exact model match — reprocess sessions from other models.
            skip_condition = "st.topics_source = ?"
            skip_param: str | None = f"llm_v1:{model}"
        else:
            # Skip any LLM-processed session.
            skip_condition = "st.topics_source LIKE 'llm_v1:%'"
            skip_param = None

        paths: list[str] = []
        if stenogram_list_path:
            data = json.loads(stenogram_list_path.read_text(encoding="utf-8"))
            paths = [str(p) for p in data.get("files", [])]

        # When paths is non-empty, restrict to sessions from those stenograms.
        # When empty (no new stenograms, resuming LLM work), scan all sessions.
        if paths:
            placeholders = ",".join("?" * len(paths))
            params: list = [*paths] + ([skip_param] if skip_param is not None else [])
            rows = conn.execute(
                f"""
                SELECT DISTINCT sc.session_id
                FROM session_chunks sc
                WHERE sc.stenogram_path IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM session_topics st
                      WHERE st.session_id = sc.session_id
                        AND {skip_condition}
                  )
                ORDER BY sc.session_id
                """,
                params,
            ).fetchall()
        else:
            params = [skip_param] if skip_param is not None else []
            rows = conn.execute(
                f"""
                SELECT DISTINCT sc.session_id
                FROM session_chunks sc
                WHERE NOT EXISTS (
                    SELECT 1 FROM session_topics st
                    WHERE st.session_id = sc.session_id
                      AND {skip_condition}
                )
                ORDER BY sc.session_id
                """,
                params,
            ).fetchall()
    return [r["session_id"] for r in rows]


def run_session_topics(
    db_path: Path,
    run_id: str,
    session_ids: list[str],
    model: str,
    provider: str,
    reprocess: bool = False,
    build_prompts_only: bool = False,
) -> dict:
    _reset_usage_stats()
    client = _build_client(provider, model)

    total = len(session_ids)
    extracted = 0
    errors = 0
    error_log: list[dict] = []

    with sqlite3.connect(db_path) as raw_conn:
        raw_conn.row_factory = sqlite3.Row
        with MCPServer(db_path=db_path, run_id=run_id) as server:
            for i, session_id in enumerate(session_ids, 1):
                print(f"\n[{i}/{total}] session={session_id}")
                result = extract_session_topics(
                    server=server,
                    session_id=session_id,
                    run_id=run_id,
                    model=model,
                    client=client,
                    provider=provider,
                    conn=raw_conn,
                    build_prompts_only=build_prompts_only,
                )
                if result.get("ok"):
                    extracted += 1
                    if build_prompts_only:
                        print(f"  → prompts saved (build-prompts mode, no LLM call)")
                    else:
                        stored = result.get("stored", {})
                        print(f"  → stored: {len(stored.get('topics', []))} topics  source={stored.get('topics_source')}")
                else:
                    errors += 1
                    err_info = result.get("error", {})
                    print(f"  ✗ error: {err_info.get('code')}: {err_info.get('message')}")
                    error_log.append({"session_id": session_id, "error": err_info})

    return {
        "total": total,
        "extracted": extracted,
        "errors": errors,
        "error_log": error_log,
        "usage": _usage_summary_payload(),
    }


# ---------------------------------------------------------------------------
# External-output ingestion
# ---------------------------------------------------------------------------

def ingest_external_outputs(db_path: Path, run_id: str) -> int:
    """
    Read files from state/external_prompts_output/ whose names match
    session_topics_* and store the parsed topic JSON to the DB via MCP.

    Expected file naming (same as prompt output files):
      session_topics_{session_date}_{session_id}_{timestamp}_{model_safe}_{label}.txt

    File content: the raw JSON response from the external model — just the
    JSON object, e.g. either legacy:
      {"topics": [...], "session_summary": "..."}
    or constrained:
      {"matched_topics": [...], "new_topics": [...], "session_summary": "..."}

    Any file that has already been ingested (tracked by a .done sidecar) is
    skipped.  Returns the number of successfully ingested files.
    """
    import re as _re
    ext_dir = EXTERNAL_OUTPUTS_DIR
    if not ext_dir.exists():
        print(f"  External outputs dir not found: {ext_dir}")
        return 0

    files = sorted(ext_dir.glob("session_topics_*.txt"))
    if not files:
        print(f"  No session_topics_*.txt files in {ext_dir}")
        return 0

    ingested = 0
    init_db(db_path)

    with sqlite3.connect(db_path) as raw_conn:
        raw_conn.row_factory = sqlite3.Row
        with MCPServer(db_path=db_path, run_id=run_id) as server:
            for f in files:
                done_marker = f.with_suffix(".done")
                if done_marker.exists():
                    print(f"  [skip] {f.name}  (already ingested)")
                    continue

                # Parse filename: session_topics_{date}_{sid}_{ts}_{model}_{label}.txt
                # Fields after the step prefix are: date, sid, ts, model, label
                stem = f.stem  # strip .txt
                parts = stem.split("_")
                # parts[0] == "session", parts[1] == "topics", parts[2] == date,
                # parts[3] == session_id, parts[4] == ts, parts[5+] == model + label
                if len(parts) < 6:
                    print(f"  [skip] {f.name}  (unexpected filename format)")
                    continue
                session_date = parts[2]
                session_id = parts[3]
                # Reconstruct model from parts between ts and label.
                # Convention: model is everything between ts and the last underscore-segment(s)
                # that constitute the label.  We keep it simple: everything after ts until
                # the last segment is the model; the last segment is the label.
                model_and_label = "_".join(parts[5:])
                # Model name was sanitised with _safe_model (: → -), label is last segment.
                # We only need model for topics_source — use the full remaining string minus
                # the label suffix (last _ group).  Since model can itself contain _ we
                # reconstruct as: everything except the very last _-separated token.
                ml_parts = model_and_label.rsplit("_", 1)
                model_safe = ml_parts[0] if len(ml_parts) > 1 else model_and_label
                # Restore colons in model name (- was used as replacement).
                model_hint = model_safe  # keep the safe version for topics_source

                # utf-8-sig strips a BOM if present (common when files are
                # saved by Windows tools or copy-pasted from some editors).
                raw_text = f.read_text(encoding="utf-8-sig").strip()
                if not raw_text:
                    print(f"  [skip] {f.name}  (empty file)")
                    continue

                try:
                    payload = _parse_topics_payload(raw_text)
                    llm_data = payload.get("topics", [])
                except (json.JSONDecodeError, ValueError):
                    # Try extracting from a {"topics": [...]} wrapper.
                    try:
                        wrapper = json.loads(raw_text)
                        if isinstance(wrapper, dict):
                            payload = _parse_topics_payload(json.dumps(wrapper))
                            llm_data = payload.get("topics", [])
                        else:
                            raise ValueError("not a dict")
                    except Exception as exc:
                        print(f"  [error] {f.name}  parse failed: {exc}")
                        continue

                # Use llm_v1: prefix so the MCP validation accepts it and the
                # result is treated identically to a locally-computed topic set.
                # The model name in the filename is already sanitised (: → -);
                # restore colons so it matches the canonical llm_v1:{model} form.
                model_canonical = model_safe.replace("-", ":", 1)  # first dash only: qwen2.5-7b → qwen2.5:7b
                topics_source = f"llm_v1:{model_canonical}"
                store_result = server.call(
                    "store_session_topics",
                    {"session_id": session_id, "topics": llm_data, "topics_source": topics_source},
                )
                if store_result.get("ok"):
                    done_marker.write_text("ingested\n", encoding="utf-8")
                    ingested += 1
                    print(
                        f"  [ok] {f.name}  session={session_id}  "
                        f"{len(llm_data)} topics  source={topics_source}"
                    )
                else:
                    err = store_result.get("error", {})
                    print(f"  [error] {f.name}  store failed: {err.get('code')}: {err.get('message')}")

    return ingested


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM session topic extraction: extract session topics via LLM + MCP."
    )
    parser.add_argument("--run-id", default=os.environ.get("VOTEZ_RUN_ID"))
    parser.add_argument(
        "--stenogram-list-path",
        default=os.environ.get("VOTEZ_STENOGRAM_LIST_PATH"),
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument("--model", default="", help="Model override.")
    parser.add_argument(
        "--session-id",
        help="Extract topics for a single session (for debugging).",
    )
    parser.add_argument(
        "--reprocess-session-topics",
        action="store_true",
        default=os.environ.get("VOTEZ_REPROCESS_SESSION_TOPICS", "").lower() in ("1", "true"),
        help=(
            "Re-extract topics even for sessions already processed by any LLM. "
            "By default, any session with topics_source LIKE 'llm_v1:%%' is skipped."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N sessions (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--build-prompts",
        action="store_true",
        help=(
            "Build prompt files in state/run_prompts/ for all targeted sessions "
            "without calling the LLM.  Use this to iterate on prompts externally."
        ),
    )
    parser.add_argument(
        "--ingest-external-outputs",
        action="store_true",
        help=(
            "Ingest LLM responses from state/external_prompts_output/ "
            "(files named like the corresponding prompt files) and store them "
            "to the DB.  No LLM calls are made."
        ),
    )
    parser.add_argument(
        "--log-token-usage-per-call",
        action="store_true",
        default=os.environ.get("VOTEZ_LOG_TOKEN_USAGE_PER_CALL", "").lower() in ("1", "true", "yes"),
        help="Print prompt/completion token usage for each LLM call (or estimates if provider does not return usage).",
    )
    args = parser.parse_args()

    if not args.run_id:
        print("ERROR: --run-id or VOTEZ_RUN_ID required.")
        return 1

    model = args.model.strip()
    if not model:
        model = DEFAULT_MODEL_OLLAMA if args.provider == "ollama" else DEFAULT_MODEL_OPENAI

    db_path = Path(args.db_path)
    init_db(db_path)

    global LOG_TOKEN_USAGE_PER_CALL
    LOG_TOKEN_USAGE_PER_CALL = bool(args.log_token_usage_per_call)

    # --ingest-external-outputs: read external model responses and store to DB.
    if args.ingest_external_outputs:
        print(f"\nIngesting external session-topic outputs from {EXTERNAL_OUTPUTS_DIR} ...")
        n = ingest_external_outputs(db_path, args.run_id)
        print(f"Ingested {n} file(s).")
        return 0

    if args.session_id:
        session_ids = [args.session_id]
    elif args.build_prompts:
        # Build-prompts targets ALL sessions so the generated_prompts/ directory
        # is a complete snapshot — not just sessions with pending LLM work.
        session_ids = _load_all_session_ids(db_path)
    else:
        list_path = Path(args.stenogram_list_path) if args.stenogram_list_path else None
        session_ids = _load_session_ids(db_path, list_path, model, reprocess=args.reprocess_session_topics)

    if not session_ids:
        print("No sessions found. Nothing to do.")
        return 0

    if args.limit > 0:
        session_ids = session_ids[: args.limit]

    reprocess_note = "  (reprocess=True — overwriting existing LLM topics)" if args.reprocess_session_topics else ""
    limit_note = f"  (limit={args.limit})" if args.limit > 0 else ""
    mode_note = "  [BUILD-PROMPTS — writing to state/generated_prompts/]" if args.build_prompts else ""
    print(
        f"Session topic extraction: {len(session_ids)} session(s) "
        f"(run_id={args.run_id}){reprocess_note}{limit_note}{mode_note}"
    )

    summary = run_session_topics(
        db_path=db_path,
        run_id=args.run_id,
        session_ids=session_ids,
        model=model,
        provider=args.provider,
        reprocess=args.reprocess_session_topics,
        build_prompts_only=args.build_prompts,
    )

    if args.build_prompts:
        from prompt_logger import GENERATED_PROMPTS_DIR
        print(
            f"\nBuild-prompts finished: {summary['extracted']}/{summary['total']} sessions, "
            f"prompts written to {GENERATED_PROMPTS_DIR}/"
        )
    else:
        print(
            f"\nSession topics finished: {summary['extracted']}/{summary['total']} extracted, "
            f"{summary['errors']} error(s)."
        )
    usage = summary.get("usage", {})
    if usage:
        if int(usage.get("calls_with_actual_usage", 0)) > 0:
            print(
                "Token usage (LLM-reported): "
                f"calls={usage.get('calls', 0)}  "
                f"prompt={usage.get('prompt_tokens', 0)}  "
                f"completion={usage.get('completion_tokens', 0)}  "
                f"total={int(usage.get('prompt_tokens', 0)) + int(usage.get('completion_tokens', 0))}"
            )
        else:
            print(
                "Token usage (estimated): "
                f"calls={usage.get('calls', 0)}  "
                f"prompt~={usage.get('estimated_prompt_tokens', 0)}  "
                f"completion~={usage.get('estimated_completion_tokens', 0)}  "
                f"total~={int(usage.get('estimated_prompt_tokens', 0)) + int(usage.get('estimated_completion_tokens', 0))}"
            )
    if summary["error_log"]:
        print("Errors:")
        for entry in summary["error_log"]:
            print(f"  {entry['session_id']}: {entry['error']}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
