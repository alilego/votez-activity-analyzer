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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from init_db import DEFAULT_DB_PATH, init_db
from mcp_server import MCPServer
from prompt_logger import EXTERNAL_OUTPUTS_DIR, save_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_OLLAMA = "qwen2.5:7b-32k"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

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
REDUCE_SYSTEM = """You are extracting a structured topic list from a Romanian parliamentary session.

You will receive bullet-list summaries from multiple windows covering the full session.
Your job: merge them into a deduplicated list of up to 20 rich topic objects.

Output format — respond with ONLY valid JSON, no prose, no markdown:
{
  "topics": [
    {"label": "LABEL_FROM_INPUT_MAX_10_WORDS", "description": "DESCRIPTION_FROM_INPUT_ONE_SENTENCE", "law_id": "LAW_ID_FROM_INPUT_OR_NULL"},
    {"label": "LABEL_FROM_INPUT_MAX_10_WORDS", "description": "DESCRIPTION_FROM_INPUT_ONE_SENTENCE", "law_id": null}
  ],
  "session_summary": "ONE_SENTENCE_SUMMARY_FROM_INPUT"
}

Rules:
- label: MAXIMUM 10 words, in Romanian, specific to this session. If a law/bill is the topic, use its short title + identifier (e.g. "Reforma pensiilor speciale PL-x 45/2025"). Never cut a label mid-sentence — finish with a complete noun phrase.
- description: one sentence in Romanian. MUST include: (a) what the topic is actually about (not just its name), (b) all specific details present in the input — amounts, ALL locations mentioned, institutions, percentages. If a law/bill is mentioned, state what it regulates. ONLY use facts that appear in the input text — do NOT invent amounts, locations, or details.
- law_id: the bill/law/ordinance identifier if mentioned (e.g. "PL-x 45/2025", "OUG 114/2018", "nr. 360/2023"), otherwise null.
- COMBINING duplicates: if two entries cover the same subject but mention different locations or amounts, COMBINE them into ONE entry whose label and description list ALL locations and amounts from both. Do NOT drop any location or amount that appears in the input.
- Only drop an entry if it is truly identical (same subject, same location, same amount) to another.
- CRITICAL: every label, description and law_id you output MUST be grounded in the input text above. Do NOT copy from examples or invent details not present in the input.
- Drop generic entries like "buget", "legislativ", "procedura" unless paired with a specific amount/date/institution.
- Keep up to 20 topics. Prefer specificity over quantity.
- Do NOT include "(fără subiecte identificate)" lines."""


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
        temperature=0.2,
        max_tokens=max_tokens,
        timeout=LLM_REQUEST_TIMEOUT_S,
        **extra_kwargs,
    )
    return response.choices[0].message.content or ""


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


def _parse_topics(raw: str) -> list[dict]:
    """Parse and normalise a reduce JSON response into a list of topic dicts."""
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```")).strip()
    parsed = json.loads(raw)
    topics_raw = parsed.get("topics", [])
    if not isinstance(topics_raw, list):
        raise ValueError("'topics' is not a list")
    topics: list[dict] = []
    seen: set[str] = set()
    for item in topics_raw:
        if isinstance(item, dict) and item.get("label"):
            label = str(item["label"]).strip()[:MAX_TOPIC_LABEL_LENGTH]
            if label.lower() in seen:
                continue
            seen.add(label.lower())
            topics.append({
                "label": label,
                "description": str(item.get("description", "")).strip()[:MAX_TOPIC_DESC_LENGTH],
                "law_id": item.get("law_id") or None,
            })
        elif isinstance(item, str) and item.strip():
            label = item.strip()[:MAX_TOPIC_LABEL_LENGTH]
            if label.lower() in seen:
                continue
            seen.add(label.lower())
            topics.append({"label": label, "description": "", "law_id": None})
        if len(topics) >= MAX_TOPICS_PER_SESSION:
            break
    return topics


def _call_reduce_once(
    client,
    session_header: str,
    prose_list: list[str],
    label: str,
    session_id: str = "",
    session_date: str = "",
    build_prompts_only: bool = False,
) -> list[dict]:
    """Run one reduce LLM call over a list of prose strings. Returns parsed topic list.

    In build_prompts_only mode the prompt is saved and an empty list is returned.
    """
    reduce_msg = _build_reduce_message(session_header, prose_list)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _chat(
                client, REDUCE_SYSTEM, reduce_msg, json_mode=True, max_tokens=2048,
                prompt_label=label.lower().replace(" ", "_"),
                session_id=session_id, session_date=session_date,
                build_prompts_only=build_prompts_only,
            )
            topics = _parse_topics(raw)
            print(f"  {label}: {len(topics)} topics from {len(prose_list)} inputs ({len(reduce_msg)} chars)")
            return topics
        except _BuildPromptsOnly:
            print(f"  {label}: prompt saved (build-prompts mode)")
            return []  # no retries — prompt is saved, nothing to parse
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  {label} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    return []


def _reduce(
    client,
    session_header: str,
    window_results: list[str],
    session_id: str = "",
    session_date: str = "",
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
        topics = _call_reduce_once(
            client, session_header, batches[0], "Reduce",
            session_id=session_id, session_date=session_date,
            build_prompts_only=build_prompts_only,
        )
        return {"topics": topics, "session_summary": ""}

    # Stage 1: reduce each batch to an intermediate topic list.
    intermediate_prose: list[str] = []
    for b_idx, batch in enumerate(batches, 1):
        partial = _call_reduce_once(
            client, session_header, batch, f"Reduce stage-1 batch {b_idx}/{len(batches)}",
            session_id=session_id, session_date=session_date,
            build_prompts_only=build_prompts_only,
        )
        if partial:
            lines = []
            for t in partial:
                law = f" ({t['law_id']})" if t.get("law_id") else ""
                desc = f": {t['description']}" if t.get("description") else ""
                lines.append(f"- {t['label']}{law}{desc}")
            intermediate_prose.append("\n".join(lines))

    # Stage 2: merge all intermediate lists into the final result.
    final_topics = _call_reduce_once(
        client, session_header, intermediate_prose, "Reduce stage-2 final",
        session_id=session_id, session_date=session_date,
        build_prompts_only=build_prompts_only,
    )
    return {"topics": final_topics, "session_summary": ""}


SINGLE_PASS_SYSTEM = """You are an expert analyst of Romanian Parliament plenary sessions.

You will receive the full transcript of one session as a numbered list of speech fragments.
Your task: read the entire text and produce a structured list of up to 20 specific topics debated.

Output format — respond with ONLY valid JSON, no prose, no markdown fences:
{
  "topics": [
    {"label": "LABEL_MAX_10_WORDS_IN_ROMANIAN", "description": "ONE_SENTENCE_IN_ROMANIAN_WITH_ALL_DETAILS", "law_id": "IDENTIFIER_OR_NULL"},
    ...
  ],
  "session_summary": "ONE_SENTENCE_OVERVIEW"
}

Rules:
- label: maximum 10 words in Romanian. If a law/bill is the topic include its identifier (e.g. "Reforma pensiilor PL-x 45/2025"). Never cut a label mid-phrase.
- description: one sentence in Romanian. Must include: (a) what the topic is actually about, (b) all specific details from the text — amounts, ALL locations, institutions, percentages. If a law is mentioned, state what it regulates.
- law_id: bill/law/ordinance identifier if present (e.g. "PL-x 45/2025", "OUG 114/2018"), otherwise null.
- COMBINING: if multiple speeches cover the same subject with different locations or amounts, COMBINE into ONE entry listing ALL details.
- CRITICAL: every label, description and law_id MUST be grounded in the text. Do NOT invent details.
- Drop generic entries like "buget", "legislație", "procedură" unless paired with a specific amount/date/institution.
- Keep up to 20 topics. Prefer specificity over quantity."""


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
    build_prompts_only: bool = False,
) -> dict:
    """Single-pass topic extraction for large-context models.

    Sends the entire session in one LLM call. Only used when the model's
    context window is large enough (signalled by OLLAMA_NUM_CTX >= 32768)
    and the session text fits within LARGE_CTX_THRESHOLD_CHARS.
    """
    user_msg = _build_single_pass_message(session_header, chunks)
    total_chars = len(user_msg)
    print(f"  Single-pass: {len(chunks)} chunks, {total_chars:,} chars in one call")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _chat(
                client, SINGLE_PASS_SYSTEM, user_msg, json_mode=True, max_tokens=2048,
                prompt_label="single_pass",
                session_id=session_id, session_date=session_date,
                build_prompts_only=build_prompts_only,
            )
            topics = _parse_topics(raw)
            print(f"  Single-pass result: {len(topics)} topics  (first 120 chars: {raw[:120]!r})")
            return {"topics": topics, "session_summary": ""}
        except _BuildPromptsOnly:
            print(f"  Single-pass: prompt saved (build-prompts mode)")
            return {"topics": [], "session_summary": ""}
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  Single-pass attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    return {"topics": [], "session_summary": ""}


def _call_map_reduce(
    client,
    session_header: str,
    substantive_chunks: list[dict],
    session_id: str = "",
    session_date: str = "",
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
                   session_date=session_date, build_prompts_only=build_prompts_only)


# ---------------------------------------------------------------------------
# Single-session extraction
# ---------------------------------------------------------------------------

def extract_session_topics(
    server: MCPServer,
    session_id: str,
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
        SELECT chunk_id, chunk_type, text
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

    # Build the header that goes at the top of every window message.
    session_header = (
        f"Ședința ID: {session_id}  Data: {session.get('session_date', '')}  "
        f"({len(speech_pool)} fragmente substantive)"
    )
    notes_text = (notes_rows[0]["text"].strip() if notes_rows else "")
    if notes_text:
        session_header += f"\nNote inițiale: {notes_text[:200]}"

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
            build_prompts_only=build_prompts_only,
        )

    if build_prompts_only:
        return {"ok": True, "prompts_only": True}

    topics = llm_data.get("topics", [])
    session_summary = llm_data.get("session_summary", "")
    print(f"  session_summary={session_summary!r}")
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

    return {"total": total, "extracted": extracted, "errors": errors, "error_log": error_log}


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
    JSON object, e.g.:
      {"topics": [...], "session_summary": "..."}

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
                    llm_data = _parse_topics(raw_text)
                except (json.JSONDecodeError, ValueError):
                    # Try extracting from a {"topics": [...]} wrapper.
                    try:
                        wrapper = json.loads(raw_text)
                        if isinstance(wrapper, dict):
                            llm_data = _parse_topics(json.dumps(wrapper))
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
    args = parser.parse_args()

    if not args.run_id:
        print("ERROR: --run-id or VOTEZ_RUN_ID required.")
        return 1

    model = args.model.strip()
    if not model:
        model = DEFAULT_MODEL_OLLAMA if args.provider == "ollama" else DEFAULT_MODEL_OPENAI

    db_path = Path(args.db_path)
    init_db(db_path)

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
    if summary["error_log"]:
        print("Errors:")
        for entry in summary["error_log"]:
            print(f"  {entry['session_id']}: {entry['error']}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
