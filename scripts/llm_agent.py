#!/usr/bin/env python3
"""
LLM agent for classifying parliamentary interventions in single-speech mode.

Strategy
--------
Run one LLM call per target intervention speech, while including up to
``PREVIOUS_CONTEXT_WINDOW`` previous speeches from the same session as context.
This keeps prompts grounded in local debate flow and avoids multi-item
alignment errors.

Per-session flow
----------------
  1. MCP get_run_config       → classification rules
  2. MCP get_session          → date, initial_notes
  3. MCP get_session_topics   → session-level topics (grounding context)
  4. Load all interventions for the session from the DB (ordered by speech_index)
  5. For each target intervention:
       a. Merge continuation fragments when the speaker resumes after procedural interruption(s)
       b. Build one prompt with up to 9 previous speeches as context
       c. Call LLM → JSON result for one target speech
       d. Validate output and store via MCP

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
EST_CHARS_PER_TOKEN = 3.8
LOG_TOKEN_USAGE_PER_CALL = False

MAX_RETRIES = 3
RETRY_DELAY_S = 10
PREVIOUS_CONTEXT_WINDOW = 9

# Hard timeout per LLM request. 600s gives headroom for slow local inference.
LLM_REQUEST_TIMEOUT_S = 600

# num_ctx for qwen2.5:7b-32k. Passed via Ollama's extra_body.
OLLAMA_NUM_CTX = 32768

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
            print(f"      [tokens] {label}: prompt={prompt_tokens} completion={completion_tokens} total={total}")
        else:
            total_est = est_prompt + est_completion
            print(f"      [tokens~] {label}: prompt~={est_prompt} completion~={est_completion} total~={total_est}")


def _usage_summary_payload() -> dict:
    return {
        "calls": int(_LLM_USAGE["calls"]),
        "calls_with_actual_usage": int(_LLM_USAGE["calls_with_actual_usage"]),
        "prompt_tokens": int(_LLM_USAGE["prompt_tokens"]),
        "completion_tokens": int(_LLM_USAGE["completion_tokens"]),
        "estimated_prompt_tokens": int(_LLM_USAGE["estimated_prompt_tokens"]),
        "estimated_completion_tokens": int(_LLM_USAGE["estimated_completion_tokens"]),
    }

# ---------------------------------------------------------------------------
# System prompt — intervention mode
# ---------------------------------------------------------------------------

INTERVENTION_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Your task is to evaluate whether a parliamentary intervention contributes constructively to public policy discussion.

You will receive ONE target speech from a single parliamentary session, plus up to 9 previous speeches for context. 
Evaluate and classify ONLY the target speech listed under "Speech to classify" / "Speeches to classify". Do NOT classify context speeches marked with [ctx].
Return results as JSON.

Context speeches are provided only to help understand references, replies, or implied topics.
Do NOT evaluate or classify context speeches.
Use context only to interpret the meaning of the target speech.

Being on-topic is NOT sufficient for `constructive`.

## Early filter (apply first)
If the target speech is clearly procedural (e.g. speaking order, vote instructions/announcements, greetings, chair logistics without policy substance), classify it immediately as `neutral` and skip full substantive evaluation.
When this early filter applies:
- set `constructiveness_label = neutral`
- set `procedural_content = yes`
- keep other criteria as `no` or `partial` unless explicit substantive content is clearly present

First evaluate the speech using the criteria below.

## Criteria

1. Policy proposal
Does the speaker propose a concrete policy action, amendment, or solution?
Also count as relevant here: compromise, refinement, or better implementation proposals.

2. Policy analysis
Does the speaker provide reasoning or analysis related to policy outcomes?
Also count as relevant here: evidence/facts, legal/technical/policy consequences, and substantive questions that improve debate.

3. Public interest orientation
Does the intervention focus on benefits or consequences for citizens or society (public good), not only party advantage?

4. Partisan rhetoric
Does the speech mainly attack political opponents or promote partisan messaging without substantive argument?
Also count as relevant here: personal attacks, guilt by association, slogans without substance, repeated partisan talking points without argument, obstruction without substantive justification, mockery replacing argument.

5. Legislative engagement
Does the speaker refer directly to legislative material, for example:
- a specific article of the law
- a committee report
- a legislative amendment
- a specific bill identifier (for example PL-x ...)

6. Procedural content
Is the speech mainly procedural/logistical, for example:
- voting instructions or vote announcements
- speaking order / time management
- greetings or short formal interjections
- chair interventions without substantive policy content

7. Argumentation quality
Does the speaker provide reasoning, evidence, examples, or logical explanation supporting their position?

Use this scale for criteria 1-6 (`yes` / `partial` / `no`):
- `yes`: the criterion is clearly present and supported by concrete parts of the speech.
- `partial`: the criterion is present but limited, ambiguous, or only briefly supported.
- `no`: the criterion is absent or contradicted by the speech content.

Use this scale for criterion 7 (`strong` / `weak` / `none`):
- `strong`: clear, coherent support with evidence/examples/logical explanation.
- `weak`: limited or mostly assertive support, with partial reasoning.
- `none`: no meaningful supporting reasoning/evidence.



## Decision guidance

Use the criteria fields first, then assign `constructiveness_label`:

- `constructive` if:
  - `policy_proposal = yes` OR
  - `policy_analysis = yes` OR
  - `legislative_engagement = yes`
  AND
  - `partisan_rhetoric != yes`

- `neutral` if:
  - `procedural_content = yes`
  AND
  - all other criteria are `no` or `partial`

- `non_constructive` if:
  - `partisan_rhetoric = yes`
  AND
  - `policy_proposal = no`
  AND
  - `policy_analysis = no`

Conflict resolution (when multiple rules seem to apply):
- If `partisan_rhetoric = yes` and BOTH `policy_proposal` and `policy_analysis` are `no`, classify `non_constructive` (even if other criteria are `partial`).
- If `procedural_content = yes` and ALL substantive criteria (`policy_proposal`, `policy_analysis`, `legislative_engagement`, `public_interest_orientation`) are `no` or `partial`, classify `neutral`.
- If any substantive criterion is `yes` (`policy_proposal` OR `policy_analysis` OR `legislative_engagement`) and `partisan_rhetoric != yes`, classify `constructive`.
- If both substantive content and partisan rhetoric are strong (`partisan_rhetoric = yes` plus any substantive criterion = `yes`), classify by the dominant share of content:
  - mostly substantive argumentation/proposals -> `constructive` with lower confidence
  - mostly attack/slogan/obstruction -> `non_constructive` with lower confidence

Confidence guidance:
- Clear, single-rule case: 0.80-0.95
- Mixed but still clearly one-sided: 0.65-0.79
- Balanced/ambiguous mixed case: 0.50-0.64
- Highly uncertain / insufficient evidence / strong unresolved conflict between cues: 0.30-0.49

## Classification labels
- `constructive`: speaker genuinely advances the public good through substantive policy contribution aimed at better outcomes for citizens.
  Typical constructive behaviors include:
  - proposes a policy, amendment, or concrete solution
  - adds evidence, facts, or relevant reasoning
  - clarifies legal, technical, or policy consequences
  - asks substantive questions that improve debate
  - attempts compromise, refinement, or better implementation
- `neutral`: procedural / logistical / non-substantive — voting instructions, quorum calls, greetings, short interjections, chair time-keeping lines (e.g. "Vă rog, aveți cuvântul.", "Mulțumesc.").
  Typical neutral behaviors include:
  - procedural remarks
  - vote announcements
  - chair interventions without substantive policy content
- `non_constructive`: serves narrow interests (party, career, sponsor) or blocks debate — rhetorical attacks, filibustering, partisan positioning without substance, conspiracy claims.
  Typical non-constructive behaviors include:
  - personal attacks
  - guilt by association
  - slogans without substance
  - repeated partisan talking points with no argument
  - obstruction without substantive justification
  - mockery replacing argument

## Key rule
A speech can be fully on-topic yet `non_constructive` if it primarily serves narrow interests or blocks progress.
Ideology, party, and policy direction must not affect label decisions; only the content of the speech should be considered.

## Edge cases
- On-topic but purely self-serving or partisan → `non_constructive`
- Mixed content that includes constructive behavior → `constructive` with lower confidence (adjust confidence by portion of each type of behavior)
- Legitimate opposition or criticism WITH evidence, explanations, or concrete alternatives → `constructive`
- Opposition that is purely rhetorical or blocking without evidence or concrete alternatives → `non_constructive`
- ≤2 sentences, no policy substance, or pure chair line → `neutral`
- Formal report speeches that present analysis/recommendations to plen (e.g. committee report + approval proposal) → `constructive`
- Emotional tone is not enough to classify as non-constructive; a harsh but evidence-based intervention can still be constructive

## Topic extraction
Topic selection must NOT influence `constructiveness_label`.
First determine `constructiveness_label` from criteria and decision guidance, then assign topics.
For each speech: select up to 3 topics ONLY from the provided session topics list.
If none apply, return [].
Do NOT invent new topics outside the session topics list.
You may associate a topic from context when the speech is a response to previous interventions on that topic (even if the label is implied rather than repeated verbatim).

## Output format
Respond with ONLY valid JSON — one object per target speech, in the SAME ORDER as the input, no prose, no markdown fences:
[
  {
    "speech_index": <integer from input>,
    "constructiveness_label": "constructive" | "neutral" | "non_constructive",
    "policy_proposal": "yes" | "partial" | "no",
    "policy_analysis": "yes" | "partial" | "no",
    "public_interest_orientation": "yes" | "partial" | "no",
    "partisan_rhetoric": "yes" | "partial" | "no",
    "legislative_engagement": "yes" | "partial" | "no",
    "procedural_content": "yes" | "partial" | "no",
    "argumentation_quality": "strong" | "weak" | "none",
    "confidence": 0.0-1.0,
    "topics": ["topic1", "topic2"],
    "reasoning": "o propoziție în română care explică clasificarea, referindu-se la conținut concret din discurs",
    "evidence_quote": "un citat scurt exact (6-20 cuvinte), care apare verbatim în acel discurs"
  },
  ...
]

Return EXACTLY one object for EACH input speech_index in "Speeches to classify". Do not skip any speech_index.""" 


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


def _build_intervention_message(
    session: dict,
    session_topics: list,
    speeches: list[dict],
    context_speeches: list[dict] | None = None,
) -> str:
    """Build the user message for one LLM call.

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

    if context_speeches:
        parts.append(
            f"## Previous speeches for context ({len(context_speeches)}; do NOT classify)"
        )
        for sp in context_speeches:
            parts.append(
                f"[ctx {sp['speech_index']}] Speaker: {sp['raw_speaker']}\n"
                f"{sp['text'].strip()}"
            )

    # Target speech/speeches to classify
    if len(speeches) == 1:
        parts.append("## Speech to classify (1 target speech)")
    else:
        parts.append(f"## Speeches to classify ({len(speeches)} target speeches)")
    for sp in speeches:
        parts.append(
            f"[{sp['speech_index']}] Speaker: {sp['raw_speaker']}\n"
            f"{sp['text'].strip()}"
        )

    if len(speeches) == 1:
        parts.append(
            "Classify ONLY the target speech above (not [ctx] context speeches). "
            "Return exactly one object for that speech_index."
        )
    else:
        parts.append(
            "Classify ONLY the target speeches above (not [ctx] context speeches). "
            "Return a JSON array with one object per speech in the same order, using the speech_index values shown."
        )
    return "\n\n".join(parts)


def _strip_json_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
    return content


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

class _BuildPromptsOnly(Exception):
    """Raised by _call_llm when build_prompts_only=True to skip the LLM call."""


def _call_llm(
    client,
    provider: str,
    session: dict,
    session_topics: list,
    speeches: list[dict],
    call_label: str = "call",
    context_speeches: list[dict] | None = None,
    build_prompts_only: bool = False,
) -> list[dict]:
    """
    Call the LLM for one prompt payload (typically one target speech).
    Returns a list of raw classification dicts. Raises ValueError on parse failure.

    When ``build_prompts_only=True`` the prompt is saved with a stable
    ``"draft"`` timestamp and ``_BuildPromptsOnly`` is raised instead of
    calling the LLM.
    """
    user_msg = _build_intervention_message(
        session,
        session_topics,
        speeches,
        context_speeches=context_speeches,
    )

    # json_object mode requires the response to be a single JSON object, not an array.
    # We wrap the array in an object and unwrap it after parsing.
    wrapped_system = (
        INTERVENTION_SYSTEM_PROMPT
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
            label=call_label,
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
        raise _BuildPromptsOnly(call_label)

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
    raw_content = response.choices[0].message.content or ""
    content = _strip_json_fences(raw_content)
    _record_llm_usage(response, wrapped_system, user_msg, raw_content, call_label=call_label)
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


def _is_continuation_start(text: str) -> bool:
    stripped = (text or "").lstrip()
    if not stripped:
        return False
    if stripped.startswith("...") or stripped.startswith("…"):
        return True
    first = stripped[0]
    return first.isalpha() and first.islower()


def _has_interruption_marker(text: str) -> bool:
    key = _text_key(text or "")
    return (
        "i se intrerupe microfonul" in key
        or "intrerupe microfonul" in key
        or "microfonul" in key
    )


def _is_procedural_interruption_speech(text: str) -> bool:
    key = _text_key(text or "")
    if not key:
        return False
    # Chair/facilitator interjections are usually very short procedural lines.
    if len(key) > 260:
        return False
    markers = (
        "multumesc",
        "multumim",
        "va rog",
        "se pregateste",
        "are cuvantul",
        "propuneri la ordinea de zi",
        "intram in ordinea de zi",
        "declar inchisa sedinta",
        "dezapasati",
        "domnul deputat",
        "doamna deputat",
    )
    return any(marker in key for marker in markers)


def _merge_continuation_text(all_speeches: list[dict], current_pos: int) -> tuple[str, list[int]]:
    """Merge split speeches when the same speaker continues after procedural interruption(s)."""
    current = all_speeches[current_pos]
    current_text = str(current.get("text", "")).strip()
    if not current_text:
        return "", [int(current.get("speech_index", -1))]

    merged_parts = [current_text]
    merged_indices = [int(current.get("speech_index", -1))]
    cursor = current_pos
    # Support chains like A, chair, A, chair, A ...
    while cursor >= 2:
        interruption = all_speeches[cursor - 1]
        previous = all_speeches[cursor - 2]
        if str(previous.get("raw_speaker", "")).strip() != str(current.get("raw_speaker", "")).strip():
            break
        if not _is_procedural_interruption_speech(str(interruption.get("text", ""))):
            break

        # Conservative guard: only merge when continuation is strongly signaled.
        if not (
            _is_continuation_start(merged_parts[0])
            or _has_interruption_marker(str(previous.get("text", "")))
            or _has_interruption_marker(str(interruption.get("text", "")))
        ):
            break

        merged_parts.insert(0, str(previous.get("text", "")).strip())
        merged_indices.insert(0, int(previous.get("speech_index", -1)))
        cursor -= 2

    return "\n\n".join(p for p in merged_parts if p), merged_indices


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
    """Validate and clean one speech result from LLM output."""
    label = str(item.get("constructiveness_label", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    def normalize_choice(raw, allowed: set[str], default: str) -> str:
        value = str(raw or "").strip().lower()
        return value if value in allowed else default

    policy_proposal = normalize_choice(
        item.get("policy_proposal"),
        {"yes", "partial", "no"},
        "partial",
    )
    policy_analysis = normalize_choice(
        item.get("policy_analysis"),
        {"yes", "partial", "no"},
        "partial",
    )
    public_interest_orientation = normalize_choice(
        item.get("public_interest_orientation", item.get("public_interest")),
        {"yes", "partial", "no"},
        "partial",
    )
    partisan_rhetoric = normalize_choice(
        item.get("partisan_rhetoric", item.get("rhetorical_attack")),
        {"yes", "partial", "no"},
        "partial",
    )
    legislative_engagement = normalize_choice(
        item.get("legislative_engagement"),
        {"yes", "partial", "no"},
        "partial",
    )
    procedural_content = normalize_choice(
        item.get("procedural_content"),
        {"yes", "partial", "no"},
        "partial",
    )
    arg_raw = str(item.get("argumentation_quality") or "").strip().lower()
    if not arg_raw:
        # Backward-compatible fallback when model omits the new field.
        if policy_analysis == "yes":
            arg_raw = "strong"
        elif policy_analysis == "partial":
            arg_raw = "weak"
        else:
            arg_raw = "none"
    argumentation_quality = arg_raw if arg_raw in {"strong", "weak", "none"} else "weak"

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

    # Deterministic procedural short-circuit to reduce false non-neutral labels.
    procedural_short_circuit = (
        procedural_content == "yes"
        and policy_proposal != "yes"
        and policy_analysis != "yes"
        and legislative_engagement != "yes"
        and public_interest_orientation != "yes"
        and partisan_rhetoric != "yes"
    )
    if procedural_short_circuit:
        if label != "neutral":
            label = "neutral"
            reasoning = ""
        confidence = max(confidence, 0.75)

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
        "policy_proposal": policy_proposal,
        "policy_analysis": policy_analysis,
        "public_interest_orientation": public_interest_orientation,
        "partisan_rhetoric": partisan_rhetoric,
        "legislative_engagement": legislative_engagement,
        "procedural_content": procedural_content,
        "argumentation_quality": argumentation_quality,
        "topics": topics,
        "confidence": confidence,
        "evidence_chunk_ids": [],  # no RAG here — speech + local session context are provided
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
    parent_call_label: str,
    context_speeches: list[dict] | None = None,
) -> dict | None:
    """Reclassify one speech when initial output looks misaligned/ungrounded."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_results = _call_llm(
                client=client,
                provider=provider,
                session=session,
                session_topics=session_topics,
                speeches=[sp],
                call_label=f"{parent_call_label}_recheck_{sp['speech_index']}",
                context_speeches=context_speeches,
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


def _realign_results_to_speeches(target_speeches: list[dict], items: list[dict]) -> dict[int, dict]:
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
    for sp in target_speeches:
        idx = sp["speech_index"]
        item = by_index.get(idx)
        if item is None:
            continue
        quote = str(item.get("evidence_quote", "")).strip()
        if (not quote) or _quote_matches_speech(quote, sp["text"]):
            assigned[idx] = item
            used_item_ids.add(id(item))

    # Pass 2: quote-based rescue for unassigned speeches.
    for sp in target_speeches:
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
    for sp in target_speeches:
        idx = sp["speech_index"]
        if idx in assigned:
            continue
        item = by_index.get(idx)
        if item is not None and id(item) not in used_item_ids:
            assigned[idx] = item
            used_item_ids.add(id(item))

    return assigned


# ---------------------------------------------------------------------------
# Session-level intervention classification
# ---------------------------------------------------------------------------

def classify_session_interventions(
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
    Classify all interventions in a session using one LLM call per target speech.

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

    target_speeches = [sp for sp in all_speeches if sp["intervention_id"] in id_set]
    pos_by_index = {int(sp["speech_index"]): idx for idx, sp in enumerate(all_speeches)}
    total_chars = sum(len(sp["text"]) for sp in target_speeches)
    print(
        f"  Session {session_id}: single-speech mode for {len(target_speeches)} intervention(s), "
        f"{total_chars:,} chars total, context window={PREVIOUS_CONTEXT_WINDOW}"
    )

    classified = 0
    errors = 0
    error_log: list[dict] = []

    for t_idx, sp in enumerate(target_speeches, 1):
        iid = sp["intervention_id"]
        speech_index = int(sp["speech_index"])
        current_pos = pos_by_index.get(speech_index, -1)
        sp_for_llm = dict(sp)
        continuation_indices: list[int] = [speech_index]
        if current_pos >= 0:
            merged_text, merged_indices = _merge_continuation_text(all_speeches, current_pos)
            if merged_text:
                sp_for_llm["text"] = merged_text
                continuation_indices = merged_indices
        prev_context = [
            s for s in all_speeches if s["speech_index"] < sp["speech_index"]
        ][-PREVIOUS_CONTEXT_WINDOW:]
        print(
            f"  Speech {t_idx}/{len(target_speeches)}: idx={sp['speech_index']} "
            f"context={len(prev_context)}"
        )
        if len(continuation_indices) > 1:
            print(
                "    Continuation merge: "
                f"indices={continuation_indices} speaker={sp['raw_speaker']!r}"
            )

        raw_results: list[dict] = []
        last_exc: Exception | None = None
        call_label = f"single_{t_idx}of{len(target_speeches)}_ix{sp['speech_index']}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_results = _call_llm(
                    client=client,
                    provider=provider,
                    session=session,
                    session_topics=session_topics,
                    speeches=[sp_for_llm],
                    call_label=call_label,
                    context_speeches=prev_context,
                    build_prompts_only=build_prompts_only,
                )
                print(f"    LLM returned {len(raw_results)} result(s)")
                break
            except _BuildPromptsOnly:
                print(
                    f"    Speech {t_idx}/{len(target_speeches)}: prompt saved "
                    "(build-prompts mode)"
                )
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"    Speech idx={sp['speech_index']} attempt "
                    f"{attempt}/{MAX_RETRIES} failed: {exc}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

        if build_prompts_only:
            continue

        if not raw_results:
            msg = str(last_exc) if last_exc else "empty response"
            errors += 1
            error_log.append({
                "intervention_id": iid,
                "error": {"code": "LLM_SINGLE_ERROR", "message": msg},
            })
            continue

        aligned_by_index = _realign_results_to_speeches([sp], raw_results)
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
                sp=sp_for_llm,
                config=config,
                parent_call_label=f"{call_label}_missing",
                context_speeches=prev_context,
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
                        "code": "MISSING_IN_SINGLE",
                        "message": f"speech_index {sp['speech_index']} not in LLM output",
                    },
                })
                continue
        else:
            try:
                raw_for_validation = dict(raw)
                raw_for_validation["_speech_text"] = sp_for_llm["text"]
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
                sp=sp_for_llm,
                config=config,
                parent_call_label=call_label,
                context_speeches=prev_context,
            )
            if rechecked is not None:
                payload = rechecked

        print(
            f"    [{sp['speech_index']}] {sp['raw_speaker'][:40]!r}  "
            f"label={payload['constructiveness_label']}  "
            f"proposal={payload.get('policy_proposal', '?')}  "
            f"analysis={payload.get('policy_analysis', '?')}  "
            f"public={payload.get('public_interest_orientation', '?')}  "
            f"partisan={payload.get('partisan_rhetoric', '?')}  "
            f"legislative={payload.get('legislative_engagement', '?')}  "
            f"procedural={payload.get('procedural_content', '?')}  "
            f"arg={payload.get('argumentation_quality', '?')}  "
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
    Classify all given interventions grouped by session (single-speech mode).
    Returns a summary dict.

    When ``build_prompts_only=True`` prompts are built and saved but no LLM
    calls are made and nothing is stored to the DB.
    """
    _reset_usage_stats()
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
            result = classify_session_interventions(
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
        "usage": _usage_summary_payload(),
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
        description="LLM agent: classify intervention constructiveness (single-speech mode)."
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

    if args.model.strip():
        model = args.model.strip()
    elif args.provider == "openai":
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_OPENAI)
    else:
        model = DEFAULT_MODEL_OLLAMA

    db_path = Path(args.db_path)
    init_db(db_path)

    global LOG_TOKEN_USAGE_PER_CALL
    LOG_TOKEN_USAGE_PER_CALL = bool(args.log_token_usage_per_call)

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
        f"LLM agent (single-speech mode): {len(intervention_ids)} intervention(s) "
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
            print(f"  {entry['intervention_id']}: {entry['error']}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
