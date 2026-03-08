#!/usr/bin/env python3
"""
LLM agent for classifying parliamentary interventions — batch (full-session) mode.

Strategy
--------
Instead of one LLM call per intervention, we send ALL speeches of a session in a
single call so the model has full conversational context.  When a session is too
large to fit in the model's context window the speeches are split into greedy
batches (consecutive, never cutting a speech in the middle) and each batch is
sent as a separate call.

Per-session flow
----------------
  1. MCP get_run_config       → classification rules
  2. MCP get_session          → date, initial_notes
  3. MCP get_session_topics   → session-level topics (grounding context)
  4. Load all interventions for the session from the DB (ordered by speech_index)
  5. Split into greedy char-budget batches (BATCH_CHAR_BUDGET)
  6. For each batch:
       a. Build a single user message with ALL speeches + full stenogram context
       b. Call LLM → JSON array, one object per speech
       c. Validate each item
       d. MCP store_intervention_analysis for each item

Supported providers:
  openai  — requires OPENAI_API_KEY; model default: gpt-4o-mini
  ollama  — free local inference; model default: qwen2.5:7b-32k
            Ollama must be running: `ollama serve`

Environment variables:
  LLM_PROVIDER     (optional, default: ollama)
  OPENAI_API_KEY   (required for openai provider)
  OPENAI_MODEL     (optional, overrides --model default)
  OLLAMA_HOST      (optional, default: http://localhost:11434)

Usage:
  Called automatically by run_pipeline.py when --analyzer-mode=llm is set.
  Can also be run directly:
    python3 scripts/llm_agent.py \\
        --run-id run_xyz \\
        --stenogram-list-path state/run_inputs/run_xyz_stenograms.json \\
        --db-path state/state.sqlite

  Single session (for debugging / prompt polishing):
    python3 scripts/llm_agent.py --session-id 8856 --run-id run_xyz

  Single intervention (legacy debugging):
    python3 scripts/llm_agent.py --intervention-id iv:abc:10 --run-id run_xyz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
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

MAX_RETRIES = 3
RETRY_DELAY_S = 10
MISSING_RECOVERY_MAX_ROUNDS = 3
MISSING_RECOVERY_CHUNK_SIZE = 12

# Hard timeout per LLM batch request. Larger batches take longer to generate.
# 600s (10 min) gives headroom for a full session on a slow M1.
LLM_REQUEST_TIMEOUT_S = 600

# num_ctx for qwen2.5:7b-32k. Passed via Ollama's extra_body.
OLLAMA_NUM_CTX = 32768

# Greedy batch budget: how many chars of speech text to pack into one LLM call.
# qwen2.5:7b-32k: 32,768 tokens × 80% usable = ~26,200 tokens.
# Reserve ~1,000 tokens for system prompt + session header, ~3,000 for JSON output.
# Net speech budget: ~22,000 tokens × 3.5 chars/token (Romanian) ≈ 77,000 chars.
# Lower budget improves per-speech fidelity for local 7B models and reduces
# truncation/omission risk in long sessions.
BATCH_CHAR_BUDGET = 40_000

# ---------------------------------------------------------------------------
# System prompt — batch mode
# ---------------------------------------------------------------------------

BATCH_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților).

You will receive a numbered list of speeches from a single parliamentary session.
Classify EVERY speech and return results as a JSON array.

## Classification labels
- `constructive`: speaker genuinely advances the public good — proposes solutions, amendments, or substantive analysis aimed at better outcomes for citizens.
  Typical constructive behaviors include:
  - proposes a policy, amendment, or concrete solution
  - adds evidence, facts, or relevant reasoning
  - clarifies legal, technical, or policy consequences
  - asks substantive questions that improve debate
  - attempts compromise, refinement, or better implementation
- `neutral`: procedural / logistical / non-substantive — voting instructions, quorum calls, greetings, short interjections, chair time-keeping lines (e.g. "Vă rog, aveți cuvântul.", "Mulțumesc.").
- `non_constructive`: serves narrow interests (party, career, sponsor) or blocks debate — rhetorical attacks, filibustering, partisan positioning without substance, conspiracy claims.

## Key rule
Being on-topic is NOT sufficient for `constructive`. A speech can be fully on-topic yet `non_constructive` if it primarily serves narrow interests or blocks progress.

## Edge cases
- On-topic but purely self-serving or partisan → `non_constructive`
- Mixed content → classify by the dominant portion
- Legitimate opposition WITH evidence or concrete alternatives → `constructive`
- Opposition that is purely rhetorical or blocking → `non_constructive`
- ≤2 sentences, no policy substance, or pure chair line → `neutral`
- Formal report speeches that present analysis/recommendations to plen (e.g. committee report + approval proposal) → `constructive`
- For mixed speeches with both attacks and policy proposals: if concrete policy content is substantial, do NOT use `non_constructive` unless attacks clearly dominate.

## Topic extraction
For each speech: extract up to 5 short topics (1–4 words each, in Romanian) reflecting specific policy areas or legislative themes covered by THAT speech.
Return [] if none apply. Do NOT hallucinate topics not present in the text.
Prefer concrete noun phrases (e.g. "pensiile speciale", "buget local", "PL-x 45/2025"), not generic labels ("politică", "dezbatere", "discurs politic").
If a session topic clearly matches, reuse its wording for consistency.

## Output format
Respond with ONLY a valid JSON array — one object per speech, in the SAME ORDER as the input, no prose, no markdown fences:
[
  {
    "speech_index": <integer from input>,
    "constructiveness_label": "constructive" | "neutral" | "non_constructive",
    "confidence": 0.0-1.0,
    "topics": ["topic1", "topic2"],
    "reasoning": "o propoziție în română care explică clasificarea, referindu-se la conținut concret din discurs",
    "evidence_quote": "un citat scurt exact (6-20 cuvinte) din acel discurs"
  },
  ...
]

Return EXACTLY one object for EACH input speech_index. Do not skip any speech_index.""" 


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_session_topics(topics: list) -> str:
    if not topics:
        return ""
    lines: list[str] = []
    for t in topics:
        if isinstance(t, dict):
            label = t.get("label", "")
            desc = t.get("description", "")
            law_id = t.get("law_id")
            entry = f"- {label} ({law_id})" if law_id else f"- {label}"
            if desc:
                entry += f": {desc}"
            lines.append(entry)
        elif isinstance(t, str) and t.strip():
            lines.append(f"- {t.strip()}")
    return "\n".join(lines)


def _build_batch_message(
    session: dict,
    session_topics: list,
    speeches: list[dict],
) -> str:
    """Build the user message for one batch of speeches.

    Each speech dict must have: speech_index, raw_speaker, text.
    """
    parts: list[str] = []

    # Session header
    header = f"## Session\nDate: {session.get('session_date', '')}"
    notes = (session.get("initial_notes") or "").strip()
    if notes:
        header += f"\nInitial notes: {notes[:300]}"
    parts.append(header)

    # Session topics as grounding context
    topics_text = _format_session_topics(session_topics)
    if topics_text:
        parts.append(f"## Session topics (grounding context)\n{topics_text}")

    # Numbered speeches
    parts.append(f"## Speeches to classify ({len(speeches)} in this batch)")
    for sp in speeches:
        parts.append(
            f"[{sp['speech_index']}] Speaker: {sp['raw_speaker']}\n"
            f"{sp['text'].strip()}"
        )

    parts.append(
        "Classify every speech above. Return a JSON array with one object per speech "
        "in the same order, using the speech_index values shown."
    )
    return "\n\n".join(parts)


def _greedy_batches(speeches: list[dict], char_budget: int) -> list[list[dict]]:
    """Split speeches into greedy consecutive batches that each fit within char_budget.

    Never splits a speech across batches. A single speech larger than the budget
    gets its own batch (the model will have to do its best with a tight context).
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for sp in speeches:
        sp_chars = len(sp["text"])
        if current and current_chars + sp_chars > char_budget:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(sp)
        current_chars += sp_chars
    if current:
        batches.append(current)
    return batches


def _strip_json_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
    return content


# ---------------------------------------------------------------------------
# LLM call — batch
# ---------------------------------------------------------------------------

class _BuildPromptsOnly(Exception):
    """Raised by _call_llm_batch when build_prompts_only=True to skip the LLM call."""


def _call_llm_batch(
    client,
    provider: str,
    session: dict,
    session_topics: list,
    speeches: list[dict],
    batch_label: str = "batch",
    build_prompts_only: bool = False,
) -> list[dict]:
    """
    Call the LLM with a batch of speeches. Returns a list of raw classification
    dicts (one per speech). Raises ValueError on parse failure.

    When ``build_prompts_only=True`` the prompt is saved with a stable
    ``"draft"`` timestamp and ``_BuildPromptsOnly`` is raised instead of
    calling the LLM.
    """
    user_msg = _build_batch_message(session, session_topics, speeches)

    # json_object mode requires the response to be a single JSON object, not an array.
    # We wrap the array in an object and unwrap it after parsing.
    wrapped_system = (
        BATCH_SYSTEM_PROMPT
        + '\n\nIMPORTANT: wrap your array in a JSON object: {"results": [...]}'
    )

    # Save prompt before sending.
    try:
        ts = "draft" if build_prompts_only else None
        save_prompt(
            step="interventions",
            session_id=str(session.get("session_id", "")),
            session_date=str(session.get("session_date", "")),
            model=client._model,
            label=batch_label,
            system_prompt=wrapped_system,
            user_message=user_msg,
            extra_meta={
                "speeches": len(speeches),
                "user_chars": len(user_msg),
            },
            timestamp=ts,
        )
    except Exception:
        pass  # Never let logging block the actual LLM call.

    if build_prompts_only:
        raise _BuildPromptsOnly(batch_label)

    extra_kwargs: dict = {}
    if provider in ("openai", "ollama"):
        extra_kwargs["response_format"] = {"type": "json_object"}
    if provider == "ollama":
        extra_kwargs["extra_body"] = {"num_ctx": OLLAMA_NUM_CTX}

    response = client.chat.completions.create(
        model=client._model,
        messages=[
            {"role": "system", "content": wrapped_system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        timeout=LLM_REQUEST_TIMEOUT_S,
        **extra_kwargs,
    )
    content = _strip_json_fences(response.choices[0].message.content or "")
    parsed = json.loads(content)

    # Unwrap from {"results": [...]} if present, otherwise try to use top-level list.
    if isinstance(parsed, dict):
        results = parsed.get("results", parsed.get("speeches", []))
    elif isinstance(parsed, list):
        results = parsed
    else:
        raise ValueError(f"Unexpected LLM output type: {type(parsed)}")

    if not isinstance(results, list):
        raise ValueError(f"LLM did not return a list: {results!r}")
    return results


def _index_results_by_speech_index(raw_results: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for item in raw_results:
        idx = item.get("speech_index")
        if idx is None:
            continue
        try:
            out[int(idx)] = item
        except (TypeError, ValueError):
            continue
    return out


def _chunk_list(items: list[dict], chunk_size: int) -> list[list[dict]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _recover_missing_batch_results(
    client,
    provider: str,
    session: dict,
    session_topics: list,
    batch_label: str,
    missing_speeches: list[dict],
) -> dict[int, dict]:
    """Retry only missing speeches in small batches and merge recovered outputs."""
    recovered: dict[int, dict] = {}
    pending = list(missing_speeches)

    for round_idx in range(1, MISSING_RECOVERY_MAX_ROUNDS + 1):
        if not pending:
            break

        chunk_size = max(1, MISSING_RECOVERY_CHUNK_SIZE // (2 ** (round_idx - 1)))
        chunks = _chunk_list(pending, chunk_size)
        print(
            f"    Recovery round {round_idx}/{MISSING_RECOVERY_MAX_ROUNDS}: "
            f"{len(pending)} missing speech(es) in {len(chunks)} chunk(s) of <= {chunk_size}"
        )

        for c_idx, chunk in enumerate(chunks, 1):
            label = f"{batch_label}_recover_r{round_idx}_{c_idx}of{len(chunks)}"
            last_exc: Exception | None = None
            raw_results: list[dict] = []

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    raw_results = _call_llm_batch(
                        client=client,
                        provider=provider,
                        session=session,
                        session_topics=session_topics,
                        speeches=chunk,
                        batch_label=label,
                        build_prompts_only=False,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_S)

            if not raw_results:
                err = str(last_exc) if last_exc else "empty response"
                print(
                    f"      Recovery chunk {c_idx}/{len(chunks)} failed: {err}"
                )
                continue

            chunk_by_index = _index_results_by_speech_index(raw_results)
            if chunk_by_index:
                recovered.update(chunk_by_index)
                print(
                    f"      Recovery chunk {c_idx}/{len(chunks)}: "
                    f"{len(chunk_by_index)}/{len(chunk)} recovered"
                )
            else:
                print(
                    f"      Recovery chunk {c_idx}/{len(chunks)}: "
                    "returned no usable speech_index items"
                )

        pending = [sp for sp in pending if sp["speech_index"] not in recovered]

    return recovered


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALID_LABELS = {"constructive", "neutral", "non_constructive"}
_TOPIC_GENERIC_PHRASES = {
    "politica",
    "politici",
    "partid",
    "partide",
    "discurs politic",
    "dezbatere",
    "subiect",
    "tema",
    "probleme",
    "chestiuni",
    "declaratii politice",
}


def _strip_diacritics(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _topic_key(topic: str) -> str:
    """Build a stable comparison key for dedup/canonical matching."""
    key = _strip_diacritics(topic.casefold())
    key = re.sub(r"[^a-z0-9\s/.-]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _text_key(text: str) -> str:
    key = _strip_diacritics(text.casefold())
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text)


def _looks_non_romanian_reasoning(text: str) -> bool:
    toks = {t.casefold() for t in _word_tokens(_strip_diacritics(text))}
    if not toks:
        return True
    # Lightweight heuristic: common English markers.
    en_markers = {
        "the", "and", "without", "with", "for", "from", "of", "is",
        "are", "speech", "presentation", "rhetorical", "blocking", "debate",
    }
    return len(toks & en_markers) >= 2


def _content_tokens(text: str) -> set[str]:
    stop = {
        "si", "sau", "cu", "din", "pentru", "este", "sunt", "care", "aceasta",
        "acest", "fost", "prin", "iar", "dar", "cum", "mai", "nu", "la", "in",
        "pe", "de", "un", "o", "ai", "ale", "al", "se",
    }
    toks = []
    for t in _word_tokens(_strip_diacritics(text.casefold())):
        if len(t) < 4:
            continue
        if t in stop:
            continue
        toks.append(t)
    return set(toks)


def _reasoning_matches_speech(reasoning: str, speech_text: str) -> bool:
    r = _content_tokens(reasoning)
    s = _content_tokens(speech_text)
    if not r or not s:
        return False
    overlap = len(r & s)
    # Require at least some lexical grounding in the actual speech.
    return overlap >= 2


def _fallback_quote_from_speech(speech_text: str) -> str:
    tokens = _word_tokens(speech_text)
    if len(tokens) < 6:
        return ""
    quote = " ".join(tokens[:16]).strip()
    return quote[:160].strip()


def _quote_matches_speech(evidence_quote: str, speech_text: str) -> bool:
    q = _text_key((evidence_quote or "").strip().strip("\"“”"))
    s = _text_key(speech_text or "")
    return bool(q and len(q) >= 24 and q in s)


def _reasoning_prefix_for_label(label: str) -> str:
    if label == "constructive":
        return "Discursul este constructiv deoarece include conținut substanțial și orientat spre soluții."
    if label == "neutral":
        return "Intervenția este neutră, predominant procedurală sau fără substanță de politică publică."
    return "Discursul este non-constructiv deoarece domină atacul retoric sau blocarea dezbaterii."


def _compose_grounded_reasoning(label: str, topics: list[str], evidence_quote: str) -> str:
    base = _reasoning_prefix_for_label(label)
    if topics:
        base += f" Teme: {', '.join(topics[:2])}."
    if evidence_quote:
        base += f' Citat: "{evidence_quote}".'
    return base


def _token_variants(token: str) -> set[str]:
    variants = {token}
    suffixes = ("ului", "ilor", "elor", "ile", "ele", "lor", "le", "ul", "ii")
    for suf in suffixes:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            variants.add(token[: -len(suf)])
    return variants


def _topic_tokens(topic: str) -> set[str]:
    out: set[str] = set()
    for tok in _topic_key(topic).split():
        if len(tok) < 3:
            continue
        out.update(v for v in _token_variants(tok) if len(v) >= 3)
    return out


def _normalize_topic_text(raw: str) -> str:
    topic = str(raw).strip()
    topic = re.sub(r"^[\-\*\d\.\)\(\[\]]+\s*", "", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" ,;:.")
    return topic


def _looks_like_noise_topic(topic: str) -> bool:
    key = _topic_key(topic)
    if not key:
        return True
    if key in _TOPIC_GENERIC_PHRASES:
        return True
    # Require at least one letter; reject pure numbers/symbols.
    if not re.search(r"[a-zA-ZĂÂÎȘȚăâîșț]", topic):
        return True
    return False


def _session_topic_aliases(session_topics: list) -> list[tuple[str, str]]:
    """Return (display_label, normalized_key) pairs for fast matching."""
    out: list[tuple[str, str]] = []
    for item in session_topics:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
        else:
            label = str(item).strip()
        if not label:
            continue
        out.append((label, _topic_key(label)))
    return out


def _canonicalize_topic(topic: str, session_aliases: list[tuple[str, str]]) -> str:
    """Snap extracted topic to closest session topic label when clearly equivalent."""
    key = _topic_key(topic)
    if not key:
        return topic

    # Exact key match first.
    for label, alias_key in session_aliases:
        if key == alias_key:
            return label

    # Then conservative containment-based matching to avoid over-merging.
    for label, alias_key in session_aliases:
        if len(key) >= 6 and (key in alias_key or alias_key in key):
            return label

    # Fallback: high token overlap catches inflection variants
    # (e.g. "pensii speciale" vs "pensiile speciale").
    topic_toks = _topic_tokens(topic)
    if topic_toks:
        best_label = topic
        best_score = 0.0
        best_overlap = 0
        for label, alias_key in session_aliases:
            alias_toks = _topic_tokens(alias_key)
            if not alias_toks:
                continue
            overlap = len(topic_toks & alias_toks)
            denom = max(len(topic_toks), len(alias_toks))
            score = overlap / denom
            if score > best_score:
                best_score = score
                best_overlap = overlap
                best_label = label
        if best_score >= 0.5 and best_overlap >= 2:
            return best_label

    return topic


def _validate_one(item: dict, config: dict, session_topics: list) -> dict:
    """Validate and clean one speech result from the LLM batch output."""
    label = str(item.get("constructiveness_label", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    topics_raw = item.get("topics", [])
    if not isinstance(topics_raw, list):
        topics_raw = []
    max_topics = config["max_topics_per_intervention"]
    max_len = config["max_topic_length"]
    topics: list[str] = []
    seen: set[str] = set()
    session_aliases = _session_topic_aliases(session_topics)
    for t in topics_raw:
        t = _normalize_topic_text(t)
        if not t or len(t) > max_len or _looks_like_noise_topic(t):
            continue
        t = _canonicalize_topic(t, session_aliases)
        topic_key = _topic_key(t)
        if topic_key and topic_key not in seen:
            topics.append(t)
            seen.add(topic_key)
        if len(topics) >= max_topics:
            break

    confidence_raw = item.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(item.get("reasoning", "")).strip()
    evidence_quote = str(item.get("evidence_quote", "")).strip().strip("\"“”")

    # Encourage grounded reasoning by requiring quote overlap with speech text.
    if evidence_quote:
        quote_key = _text_key(evidence_quote)
    else:
        quote_key = ""
    # speech_text is attached dynamically by caller to avoid changing MCP schema.
    speech_text = str(item.get("_speech_text", ""))
    speech_key = _text_key(speech_text)
    has_grounding = bool(quote_key and len(quote_key) >= 24 and quote_key in speech_key)
    if not evidence_quote:
        evidence_quote = _fallback_quote_from_speech(speech_text)
        quote_key = _text_key(evidence_quote) if evidence_quote else ""
        has_grounding = bool(quote_key and len(quote_key) >= 24 and quote_key in speech_key)
    elif not has_grounding:
        # Model supplied a quote from a different speech; replace with a local quote.
        evidence_quote = _fallback_quote_from_speech(speech_text)
        quote_key = _text_key(evidence_quote) if evidence_quote else ""
        has_grounding = bool(quote_key and len(quote_key) >= 24 and quote_key in speech_key)

    if not has_grounding:
        confidence = min(confidence, 0.6)
        quote_note = "fără citat valid din discurs"
    else:
        quote_note = ""

    # Enforce Romanian and ensure reasoning is grounded in THIS speech.
    reasoning_grounded = bool(speech_text and _reasoning_matches_speech(reasoning, speech_text))
    if (
        (not reasoning)
        or _looks_non_romanian_reasoning(reasoning)
        or (speech_text and not reasoning_grounded)
    ):
        reasoning = _compose_grounded_reasoning(label, topics, evidence_quote)
        reasoning_grounded = True if evidence_quote else False
    elif evidence_quote and "Citat:" not in reasoning:
        reasoning = f'{reasoning} Citat: "{evidence_quote}".'
    if quote_note:
        reasoning = f"{reasoning} [{quote_note}]"

    return {
        "constructiveness_label": label,
        "topics": topics,
        "confidence": confidence,
        "evidence_chunk_ids": [],  # no RAG in batch mode — full context replaces it
        "reasoning": reasoning,
        "_needs_recheck": not has_grounding or not reasoning_grounded,
    }


def _recheck_single_speech(
    client,
    provider: str,
    session: dict,
    session_topics: list,
    sp: dict,
    config: dict,
    parent_batch_label: str,
) -> dict | None:
    """Reclassify one speech when initial output looks misaligned/ungrounded."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_results = _call_llm_batch(
                client=client,
                provider=provider,
                session=session,
                session_topics=session_topics,
                speeches=[sp],
                batch_label=f"{parent_batch_label}_recheck_{sp['speech_index']}",
                build_prompts_only=False,
            )
            if not raw_results:
                return None
            raw = raw_results[0]
            raw_for_validation = dict(raw)
            raw_for_validation["_speech_text"] = sp["text"]
            return _validate_one(raw_for_validation, config, session_topics)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    if last_exc:
        print(f"      Recheck failed for speech_index={sp['speech_index']}: {last_exc}")
    return None


def _realign_results_to_speeches(batch: list[dict], items: list[dict]) -> dict[int, dict]:
    """Align possibly shifted model items back to target speeches.

    Priority:
    1) exact speech_index + quote matches target speech (or no quote present)
    2) quote matches target speech (cross-index rescue)
    3) exact speech_index fallback
    """
    by_index = _index_results_by_speech_index(items)
    assigned: dict[int, dict] = {}
    used_item_ids: set[int] = set()

    # Pass 1: exact index with compatible quote.
    for sp in batch:
        idx = sp["speech_index"]
        item = by_index.get(idx)
        if item is None:
            continue
        quote = str(item.get("evidence_quote", "")).strip()
        if (not quote) or _quote_matches_speech(quote, sp["text"]):
            assigned[idx] = item
            used_item_ids.add(id(item))

    # Pass 2: quote-based rescue for unassigned speeches.
    for sp in batch:
        idx = sp["speech_index"]
        if idx in assigned:
            continue
        for item in items:
            if id(item) in used_item_ids:
                continue
            quote = str(item.get("evidence_quote", "")).strip()
            if quote and _quote_matches_speech(quote, sp["text"]):
                assigned[idx] = item
                used_item_ids.add(id(item))
                break

    # Pass 3: exact index fallback even if quote mismatches.
    for sp in batch:
        idx = sp["speech_index"]
        if idx in assigned:
            continue
        item = by_index.get(idx)
        if item is not None and id(item) not in used_item_ids:
            assigned[idx] = item
            used_item_ids.add(id(item))

    return assigned


# ---------------------------------------------------------------------------
# Session-level batch classification
# ---------------------------------------------------------------------------

def classify_session_batch(
    server: MCPServer,
    session_id: str,
    intervention_ids: list[str],
    client,
    provider: str,
    config: dict,
    db_path: Path,
    build_prompts_only: bool = False,
) -> dict:
    """
    Classify all interventions in a session using full-session batch LLM calls.

    When ``build_prompts_only=True`` prompts are saved but no LLM calls are
    made and nothing is stored to the DB.

    Returns {"classified": int, "errors": int, "error_log": list}.
    """
    # Fetch session metadata and topics once per session.
    session_result = server.call("get_session", {"session_id": session_id})
    session = session_result.get("session", {}) if session_result.get("ok") else {}
    session["session_id"] = session_id  # ensure always present for prompt logger

    topics_result = server.call("get_session_topics", {"session_id": session_id})
    session_topics: list = topics_result.get("topics", []) if topics_result.get("ok") else []

    # Build an id→intervention map for all target interventions.
    id_set = set(intervention_ids)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT intervention_id, speech_index, raw_speaker, text
            FROM interventions_raw
            WHERE session_id = ? AND member_id IS NOT NULL
            ORDER BY speech_index ASC
            """,
            (session_id,),
        ).fetchall()

    # Build speech list (all speeches in session order, not just the target subset).
    # We send the full session so the model has conversational context; we only
    # store results for the target intervention_ids.
    all_speeches = [
        {
            "intervention_id": r["intervention_id"],
            "speech_index": r["speech_index"],
            "raw_speaker": r["raw_speaker"],
            "text": r["text"],
        }
        for r in rows
    ]

    if not all_speeches:
        return {"classified": 0, "errors": 0, "error_log": []}

    # Split into greedy char-budget batches.
    batches = _greedy_batches(all_speeches, BATCH_CHAR_BUDGET)
    total_batches = len(batches)
    total_chars = sum(len(sp["text"]) for sp in all_speeches)
    print(
        f"  Session {session_id}: {len(all_speeches)} speeches, {total_chars:,} chars "
        f"→ {total_batches} batch(es)"
    )

    classified = 0
    errors = 0
    error_log: list[dict] = []

    for b_idx, batch in enumerate(batches, 1):
        batch_chars = sum(len(sp["text"]) for sp in batch)
        print(
            f"  Batch {b_idx}/{total_batches}: {len(batch)} speeches, {batch_chars:,} chars"
        )

        # Call LLM with retries.
        raw_results: list[dict] = []
        last_exc: Exception | None = None
        batch_label = f"batch_{b_idx}of{total_batches}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_results = _call_llm_batch(
                    client, provider, session, session_topics, batch,
                    batch_label=batch_label,
                    build_prompts_only=build_prompts_only,
                )
                print(f"    LLM returned {len(raw_results)} result(s)")
                break
            except _BuildPromptsOnly:
                print(f"    Batch {b_idx}/{total_batches}: prompt saved (build-prompts mode)")
                break  # no retries — prompt is already saved
            except Exception as exc:
                last_exc = exc
                print(f"    Batch {b_idx} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

        if build_prompts_only:
            continue  # skip result processing — nothing to store

        if not raw_results:
            msg = str(last_exc) if last_exc else "empty response"
            print(f"    Batch {b_idx} failed: {msg}")
            for sp in batch:
                if sp["intervention_id"] in id_set:
                    errors += 1
                    error_log.append({
                        "intervention_id": sp["intervention_id"],
                        "error": {"code": "LLM_BATCH_ERROR", "message": msg},
                    })
            continue

        # Build index: speech_index → raw result from LLM.
        result_by_index = _index_results_by_speech_index(raw_results)

        missing_speeches = [sp for sp in batch if sp["speech_index"] not in result_by_index]
        if missing_speeches:
            print(
                f"    Initial batch incomplete: {len(result_by_index)}/{len(batch)} "
                "results. Attempting targeted recovery..."
            )
            recovered = _recover_missing_batch_results(
                client=client,
                provider=provider,
                session=session,
                session_topics=session_topics,
                batch_label=batch_label,
                missing_speeches=missing_speeches,
            )
            if recovered:
                for idx, item in recovered.items():
                    result_by_index.setdefault(idx, item)
                print(
                    f"    Recovery merged: now {len(result_by_index)}/{len(batch)} "
                    "results available"
                )

        # Realign results in case the model shifted speech_index values while
        # providing quote evidence from adjacent speeches.
        aligned_by_index = _realign_results_to_speeches(batch, list(result_by_index.values()))

        # Match LLM results back to speeches in the batch.
        for sp in batch:
            iid = sp["intervention_id"]
            if iid not in id_set:
                continue  # Not a target — skip storing

            raw = aligned_by_index.get(sp["speech_index"])
            if raw is None:
                print(
                    f"    No result for speech_index={sp['speech_index']} ({iid}); "
                    "forcing single-speech recovery..."
                )
                recovered_payload = _recheck_single_speech(
                    client=client,
                    provider=provider,
                    session=session,
                    session_topics=session_topics,
                    sp=sp,
                    config=config,
                    parent_batch_label=f"{batch_label}_missing",
                )
                if recovered_payload is not None:
                    payload = recovered_payload
                    print(
                        f"      Recovered speech_index={sp['speech_index']} via single-speech retry."
                    )
                else:
                    errors += 1
                    error_log.append({
                        "intervention_id": iid,
                        "error": {
                            "code": "MISSING_IN_BATCH",
                            "message": f"speech_index {sp['speech_index']} not in LLM output",
                        },
                    })
                    continue
            else:
                try:
                    raw_for_validation = dict(raw)
                    raw_for_validation["_speech_text"] = sp["text"]
                    payload = _validate_one(raw_for_validation, config, session_topics)
                except ValueError as exc:
                    print(f"    Validation error for {iid}: {exc}")
                    errors += 1
                    error_log.append({
                        "intervention_id": iid,
                        "error": {"code": "LLM_RESPONSE_INVALID", "message": str(exc)},
                    })
                    continue

            if payload.get("_needs_recheck"):
                print(f"      Rechecking speech_index={sp['speech_index']} (ungrounded result)...")
                rechecked = _recheck_single_speech(
                    client=client,
                    provider=provider,
                    session=session,
                    session_topics=session_topics,
                    sp=sp,
                    config=config,
                    parent_batch_label=batch_label,
                )
                if rechecked is not None:
                    payload = rechecked

            print(
                f"    [{sp['speech_index']}] {sp['raw_speaker'][:40]!r}  "
                f"label={payload['constructiveness_label']}  "
                f"confidence={payload['confidence']:.2f}  "
                f"topics={payload['topics']}"
            )
            if payload["reasoning"]:
                print(f"      reasoning: {payload['reasoning']}")

            payload = {k: v for k, v in payload.items() if not k.startswith("_")}
            store_result = server.call(
                "store_intervention_analysis",
                {"intervention_id": iid, **payload},
            )
            if store_result.get("ok"):
                classified += 1
            else:
                errors += 1
                err_info = store_result.get("error", {})
                error_log.append({"intervention_id": iid, "error": err_info})

    return {"classified": classified, "errors": errors, "error_log": error_log}


# ---------------------------------------------------------------------------
# Main agent loop
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


def run_agent(
    db_path: Path,
    run_id: str,
    intervention_ids: list[str],
    model: str,
    provider: str = DEFAULT_PROVIDER,
    build_prompts_only: bool = False,
) -> dict:
    """
    Classify all given interventions grouped by session (batch mode).
    Returns a summary dict.

    When ``build_prompts_only=True`` prompts are built and saved but no LLM
    calls are made and nothing is stored to the DB.
    """
    client = _build_client(provider, model)

    # Group intervention_ids by session_id (preserving order within each session).
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" * len(intervention_ids))
        rows = conn.execute(
            f"""
            SELECT intervention_id, session_id
            FROM interventions_raw
            WHERE intervention_id IN ({placeholders})
            ORDER BY session_date, session_id, speech_index
            """,
            intervention_ids,
        ).fetchall()

    # Ordered dict: session_id → [intervention_id, ...]
    sessions: dict[str, list[str]] = {}
    for iid, sid in rows:
        sessions.setdefault(sid, []).append(iid)

    total = len(intervention_ids)
    classified = 0
    errors = 0
    error_log: list[dict] = []

    with MCPServer(db_path=db_path, run_id=run_id) as server:
        config_result = server.call("get_run_config", {})
        if not config_result["ok"]:
            raise SystemExit(f"get_run_config failed: {config_result}")
        config = config_result["config"]

        for s_idx, (session_id, s_iids) in enumerate(sessions.items(), 1):
            print(f"\n[Session {s_idx}/{len(sessions)}] session_id={session_id}  ({len(s_iids)} interventions)")
            result = classify_session_batch(
                server=server,
                session_id=session_id,
                intervention_ids=s_iids,
                client=client,
                provider=provider,
                config=config,
                db_path=db_path,
                build_prompts_only=build_prompts_only,
            )
            classified += result["classified"]
            errors += result["errors"]
            error_log.extend(result["error_log"])

    return {
        "total": total,
        "classified": classified,
        "errors": errors,
        "error_log": error_log,
    }


# ---------------------------------------------------------------------------
# External-output ingestion
# ---------------------------------------------------------------------------

def ingest_external_outputs(db_path: Path, run_id: str) -> int:
    """
    Read files from state/external_prompts_output/ whose names match
    interventions_* and store the parsed classification results to the DB.

    Expected file naming (same as prompt output files):
      interventions_{session_date}_{session_id}_{timestamp}_{model_safe}_{label}.txt

    File content: the raw JSON response from the external model, e.g.:
      {"results": [{"speech_index": 1, "constructiveness_label": "constructive",
                    "topics": [...], "confidence": 0.9, "reasoning": "..."}, ...]}

    Any file that has already been ingested (tracked by a .done sidecar) is
    skipped.  Returns the number of successfully ingested files.
    """
    ext_dir = EXTERNAL_OUTPUTS_DIR
    if not ext_dir.exists():
        print(f"  External outputs dir not found: {ext_dir}")
        return 0

    files = sorted(ext_dir.glob("interventions_*.txt"))
    if not files:
        print(f"  No interventions_*.txt files in {ext_dir}")
        return 0

    ingested = 0
    init_db(db_path)

    with sqlite3.connect(db_path) as raw_conn:
        raw_conn.row_factory = sqlite3.Row
        with MCPServer(db_path=db_path, run_id=run_id) as server:
            config_result = server.call("get_run_config", {})
            if not config_result["ok"]:
                print(f"  get_run_config failed: {config_result}")
                return 0
            config = config_result["config"]

            for f in files:
                done_marker = f.with_suffix(".done")
                if done_marker.exists():
                    print(f"  [skip] {f.name}  (already ingested)")
                    continue

                # Parse filename: interventions_{date}_{sid}_{ts}_{model}_{label}.txt
                stem = f.stem
                parts = stem.split("_")
                # parts[0] == "interventions", parts[1] == date, parts[2] == session_id,
                # parts[3] == ts, parts[4+] == model + label
                if len(parts) < 5:
                    print(f"  [skip] {f.name}  (unexpected filename format)")
                    continue
                session_id = parts[2]

                # utf-8-sig strips a BOM if present (common when files are
                # saved by Windows tools or copy-pasted from some editors).
                raw_text = f.read_text(encoding="utf-8-sig").strip()
                if not raw_text:
                    print(f"  [skip] {f.name}  (empty file)")
                    continue

                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    print(f"  [error] {f.name}  JSON parse failed: {exc}")
                    continue

                # Unwrap from {"results": [...]} if present.
                if isinstance(parsed, dict):
                    raw_results = parsed.get("results", parsed.get("speeches", []))
                elif isinstance(parsed, list):
                    raw_results = parsed
                else:
                    print(f"  [error] {f.name}  unexpected JSON structure")
                    continue

                if not isinstance(raw_results, list):
                    print(f"  [error] {f.name}  'results' is not a list")
                    continue

                # We need a speech_index → intervention_id map for this session.
                rows = raw_conn.execute(
                    """
                    SELECT intervention_id, speech_index
                    FROM interventions_raw
                    WHERE session_id = ? AND member_id IS NOT NULL
                    ORDER BY speech_index ASC
                    """,
                    (session_id,),
                ).fetchall()
                idx_to_iid: dict[int, str] = {r["speech_index"]: r["intervention_id"] for r in rows}
                idx_to_text: dict[int, str] = {r["speech_index"]: r["text"] for r in raw_conn.execute(
                    """
                    SELECT speech_index, text
                    FROM interventions_raw
                    WHERE session_id = ? AND member_id IS NOT NULL
                    """,
                    (session_id,),
                ).fetchall()}
                topics_result = server.call("get_session_topics", {"session_id": session_id})
                session_topics: list = topics_result.get("topics", []) if topics_result.get("ok") else []

                file_classified = 0
                file_errors = 0
                for item in raw_results:
                    idx = item.get("speech_index")
                    if idx is None:
                        file_errors += 1
                        continue
                    iid = idx_to_iid.get(int(idx))
                    if iid is None:
                        print(f"    [warn] speech_index={idx} not found in session {session_id}")
                        file_errors += 1
                        continue

                    try:
                        item_for_validation = dict(item)
                        item_for_validation["_speech_text"] = idx_to_text.get(int(idx), "")
                        payload = _validate_one(item_for_validation, config, session_topics)
                    except ValueError as exc:
                        print(f"    [error] speech_index={idx}: {exc}")
                        file_errors += 1
                        continue

                    store_result = server.call(
                        "store_intervention_analysis",
                        {"intervention_id": iid, **payload},
                    )
                    if store_result.get("ok"):
                        file_classified += 1
                    else:
                        err = store_result.get("error", {})
                        print(f"    [error] {iid}: {err.get('code')}: {err.get('message')}")
                        file_errors += 1

                done_marker.write_text("ingested\n", encoding="utf-8")
                ingested += 1
                print(
                    f"  [ok] {f.name}  session={session_id}  "
                    f"classified={file_classified}  errors={file_errors}"
                )

    return ingested


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _load_all_intervention_ids(db_path: Path) -> list[str]:
    """Return ALL intervention IDs for matched members, regardless of classification state.

    Used by --build-prompts so every session gets a prompt file, not just
    sessions with pending LLM work.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT intervention_id
            FROM interventions_raw
            WHERE member_id IS NOT NULL
            ORDER BY session_date, session_id, speech_index
            """
        ).fetchall()
    return [r[0] for r in rows]


def _load_intervention_ids(
    db_path: Path,
    run_id: str,
    stenogram_list_path: Path | None,
    session_id: str | None = None,
) -> list[str]:
    """Return unclassified intervention IDs for matched members in selected stenograms."""
    with sqlite3.connect(db_path) as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT ir.intervention_id
                FROM interventions_raw ir
                WHERE ir.member_id IS NOT NULL
                  AND ir.session_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM intervention_analysis ia
                      WHERE ia.intervention_id = ir.intervention_id
                        AND ia.relevance_source = 'llm_agent_v1'
                  )
                ORDER BY ir.session_date, ir.session_id, ir.speech_index
                """,
                (session_id,),
            ).fetchall()
            return [r[0] for r in rows]

        paths: list[str] = []
        if stenogram_list_path:
            data = json.loads(stenogram_list_path.read_text(encoding="utf-8"))
            paths = [str(p) for p in data.get("files", [])]

        if paths:
            placeholders = ",".join("?" * len(paths))
            rows = conn.execute(
                f"""
                SELECT ir.intervention_id
                FROM interventions_raw ir
                WHERE ir.member_id IS NOT NULL
                  AND ir.stenogram_path IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM intervention_analysis ia
                      WHERE ia.intervention_id = ir.intervention_id
                        AND ia.relevance_source = 'llm_agent_v1'
                  )
                ORDER BY ir.session_date, ir.session_id, ir.speech_index
                """,
                paths,
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ir.intervention_id
                FROM interventions_raw ir
                WHERE ir.member_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM intervention_analysis ia
                      WHERE ia.intervention_id = ir.intervention_id
                        AND ia.relevance_source = 'llm_agent_v1'
                  )
                ORDER BY ir.session_date, ir.session_id, ir.speech_index
                """
            ).fetchall()
    return [r[0] for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM agent: classify intervention constructiveness (batch / full-session mode)."
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
    parser.add_argument(
        "--model",
        default="",
        help=(
            f"Model name. Defaults to {DEFAULT_MODEL_OLLAMA!r} for ollama, "
            f"{DEFAULT_MODEL_OPENAI!r} for openai."
        ),
    )
    parser.add_argument(
        "--session-id",
        help="Classify all interventions for a single session (for debugging).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Classify at most N interventions total (0 = no limit).",
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

    if args.model.strip():
        model = args.model.strip()
    elif args.provider == "openai":
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_OPENAI)
    else:
        model = DEFAULT_MODEL_OLLAMA

    db_path = Path(args.db_path)
    init_db(db_path)

    # --ingest-external-outputs: read external model responses and store to DB.
    if args.ingest_external_outputs:
        print(f"\nIngesting external intervention outputs from {EXTERNAL_OUTPUTS_DIR} ...")
        n = ingest_external_outputs(db_path, args.run_id)
        print(f"Ingested {n} file(s).")
        return 0

    if args.build_prompts:
        # Build-prompts targets ALL interventions so generated_prompts/ is a
        # complete snapshot — not just sessions with pending LLM work.
        intervention_ids = _load_all_intervention_ids(db_path)
    else:
        list_path = Path(args.stenogram_list_path) if args.stenogram_list_path else None
        intervention_ids = _load_intervention_ids(db_path, args.run_id, list_path, session_id=args.session_id)

    if args.limit > 0:
        intervention_ids = intervention_ids[: args.limit]

    if not intervention_ids:
        no_work_msg = "No interventions found." if args.build_prompts else "No unclassified interventions found."
        print(f"{no_work_msg} Nothing to do.")
        return 0

    mode_note = "  [BUILD-PROMPTS — writing to state/generated_prompts/]" if args.build_prompts else ""
    print(
        f"LLM agent (batch mode): {len(intervention_ids)} intervention(s) "
        f"(run_id={args.run_id}){mode_note}"
    )

    summary = run_agent(
        db_path=db_path,
        run_id=args.run_id,
        intervention_ids=intervention_ids,
        model=model,
        provider=args.provider,
        build_prompts_only=args.build_prompts,
    )

    if args.build_prompts:
        from prompt_logger import GENERATED_PROMPTS_DIR
        print(
            f"\nBuild-prompts finished: {summary['total']} intervention(s) — "
            f"prompts written to {GENERATED_PROMPTS_DIR}/"
        )
    else:
        print(
            f"\nAgent finished: {summary['classified']}/{summary['total']} classified, "
            f"{summary['errors']} error(s)."
        )
    if summary["error_log"]:
        print("Errors:")
        for entry in summary["error_log"]:
            print(f"  {entry['intervention_id']}: {entry['error']}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
