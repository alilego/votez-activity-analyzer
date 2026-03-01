#!/usr/bin/env python3
"""
LLM agent loop for classifying parliamentary interventions.

For each intervention:
  1. Call MCP get_run_config   → learn the rules
  2. Call MCP get_intervention → fetch the text and session_id
  3. Call MCP get_session      → fetch initial_notes / source_url
  4. Call MCP retrieve_context → RAG: get grounded session context
  5. Call LLM                  → classify constructiveness + extract topics
  6. Call MCP store_intervention_analysis → persist the result

The LLM receives the classification rubric as a system prompt and the
intervention + context as a user message.  It must respond with a single
JSON object — no prose, no markdown fences.

Supported providers:
  openai  — requires OPENAI_API_KEY; model default: gpt-4o-mini
  ollama  — free local inference via Ollama; model default: llama3.1:8b
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

  To classify only a single intervention (for debugging):
    python3 scripts/llm_agent.py --intervention-id iv:abc:10 \\
        --run-id run_xyz --db-path state/state.sqlite

  Use Ollama explicitly:
    python3 scripts/llm_agent.py --provider ollama --model llama3.1:8b \\
        --run-id run_xyz --limit 5

  Use OpenAI explicitly:
    python3 scripts/llm_agent.py --provider openai --model gpt-4o-mini \\
        --run-id run_xyz --limit 5
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_OLLAMA = "llama3.1:8b-8k"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Retry on transient API errors
MAX_RETRIES = 3
RETRY_DELAY_S = 10
# Hard timeout per LLM request. If Ollama hangs this raises httpx.ReadTimeout
# which the retry loop catches. 180s gives headroom for slow M1 inference.
LLM_REQUEST_TIMEOUT_S = 180
# Ollama's default runtime context is 4096 tokens — too small for intervention prompts
# (~2-3k tokens with context chunks). 8192 is sufficient and faster to allocate than 16384.
OLLAMA_NUM_CTX = 8192

# ---------------------------------------------------------------------------
# System prompt (the full classification rubric, compact form)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a parliamentary debate analyst. Your task is to classify Romanian parliamentary interventions.

## Classification labels
- `constructive`: the speaker genuinely tries to advance the public good — proposes solutions, amendments, or substantive analysis aimed at better outcomes for citizens.
- `neutral`: procedural, logistical, or non-substantive (voting instructions, quorum, greetings, short interjections).
- `non_constructive`: serves narrow interests (party, career, sponsor) or blocks debate — rhetorical attacks, filibustering, partisan positioning without substance, conspiracy claims.

## Key rule
Being on-topic is NOT sufficient for `constructive`. An intervention can be fully on-topic yet `non_constructive` if it primarily serves narrow interests or blocks progress.

## Edge cases
- On-topic but self-serving → `non_constructive`
- Mixed content → classify by the dominant portion
- Legitimate opposition (with evidence or alternatives) → `constructive`
- Opposition that is purely rhetorical or blocking → `non_constructive`
- Very short with no substance → `neutral`

## Topic extraction
Extract up to 5 concise topics (1–4 words each) reflecting policy areas or legislative themes.
If none can be identified, return an empty list. Do NOT hallucinate topics.

## Evidence
You will receive retrieved context chunks. Reference the chunk_ids that most supported your decision in evidence_chunk_ids.

## Output format
Respond with ONLY a JSON object — no prose, no markdown, no explanation outside the JSON:
{
  "constructiveness_label": "constructive" | "neutral" | "non_constructive",
  "topics": ["string", ...],
  "confidence": 0.0–1.0,
  "evidence_chunk_ids": ["chunk_id", ...],
  "reasoning": "one sentence explaining the classification"
}"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _format_session_topics(topics: list) -> str:
    """Format session topics for injection into the prompt.

    Handles both the old format (list[str]) and the new map-reduce format
    (list[dict] with label/description/law_id keys).
    """
    if not topics:
        return ""
    lines: list[str] = []
    for t in topics:
        if isinstance(t, dict):
            label = t.get("label", "")
            desc = t.get("description", "")
            law_id = t.get("law_id")
            if law_id:
                lines.append(f"- {label} ({law_id}): {desc}" if desc else f"- {label} ({law_id})")
            else:
                lines.append(f"- {label}: {desc}" if desc else f"- {label}")
        elif isinstance(t, str) and t.strip():
            lines.append(f"- {t.strip()}")
    return "\n".join(lines)


def _build_user_message(
    intervention: dict,
    session: dict,
    context_chunks: list[dict],
    session_topics: list | None = None,
) -> str:
    parts: list[str] = []

    parts.append(f"## Session\nDate: {session.get('session_date', '')}")
    initial_notes = (session.get("initial_notes") or "").strip()
    if initial_notes:
        parts.append(f"Session notes: {initial_notes}")

    if session_topics:
        topics_text = _format_session_topics(session_topics)
        if topics_text:
            parts.append(f"\n## Session topics\n{topics_text}")

    parts.append(
        f"\n## Speaker\n{intervention.get('raw_speaker', '')} "
        f"(ID: {intervention.get('member_id', 'unknown')})"
    )

    parts.append(f"\n## Intervention text\n{intervention.get('text', '').strip()}")

    if context_chunks:
        parts.append("\n## Retrieved session context (for grounding)")
        for chunk in context_chunks:
            parts.append(
                f"[{chunk['chunk_id']} | {chunk['type']} | score={chunk['score']}]\n"
                f"{chunk['text'].strip()}"
            )

    return "\n\n".join(parts)


def _call_llm(model: str, user_message: str, client, provider: str) -> dict:
    """
    Call the LLM and parse the JSON response.
    Works for both openai and ollama providers.
    Raises ValueError if the response cannot be parsed.

    Why different kwargs per provider:
    - OpenAI uses response_format={"type": "json_object"} to enforce JSON output.
    - Ollama uses the same OpenAI-compatible client but its JSON mode is triggered
      differently: passing response_format still works for recent Ollama versions,
      but we also append an explicit JSON instruction to the system prompt to be safe.
    """
    extra_kwargs: dict = {}
    if provider in ("openai", "ollama"):
        extra_kwargs["response_format"] = {"type": "json_object"}
    # For Ollama, explicitly set num_ctx so large prompts are not
    # silently truncated (Ollama's default runtime context is only 4096 tokens).
    # Ollama's /v1/chat/completions endpoint reads num_ctx at the top level of the
    # request body, not nested under "options". extra_body merges into the top level.
    if provider == "ollama":
        extra_kwargs["extra_body"] = {"num_ctx": OLLAMA_NUM_CTX}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # deterministic
        timeout=LLM_REQUEST_TIMEOUT_S,
        **extra_kwargs,
    )
    content = response.choices[0].message.content

    # Ollama sometimes wraps JSON in markdown fences — strip them defensively.
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON: {content!r}") from exc


def _validate_llm_response(data: dict, config: dict, retrieved_chunk_ids: set[str]) -> dict:
    """
    Coerce and validate the LLM response against the run config rules.
    Returns a cleaned payload ready for store_intervention_analysis.
    Raises ValueError on unrecoverable issues.
    """
    valid_labels = set(config["constructiveness_labels"])
    label = str(data.get("constructiveness_label", "")).strip()
    if label not in valid_labels:
        raise ValueError(f"Invalid label from LLM: {label!r}")

    topics_raw = data.get("topics", [])
    if not isinstance(topics_raw, list):
        topics_raw = []
    max_topics = config["max_topics_per_intervention"]
    max_len = config["max_topic_length"]
    topics: list[str] = []
    seen: set[str] = set()
    for t in topics_raw:
        t = str(t).strip()
        if t and len(t) <= max_len and t not in seen:
            topics.append(t)
            seen.add(t)
        if len(topics) >= max_topics:
            break

    confidence_raw = data.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Only keep evidence chunk IDs that were actually retrieved (safety check).
    evidence_raw = data.get("evidence_chunk_ids", [])
    if not isinstance(evidence_raw, list):
        evidence_raw = []
    evidence = [str(cid) for cid in evidence_raw if str(cid) in retrieved_chunk_ids]

    return {
        "constructiveness_label": label,
        "topics": topics,
        "confidence": confidence,
        "evidence_chunk_ids": evidence,
    }


# ---------------------------------------------------------------------------
# Single-intervention classification
# ---------------------------------------------------------------------------

def classify_intervention(
    server: MCPServer,
    intervention_id: str,
    model: str,
    client,
    config: dict,
    provider: str = "ollama",
) -> dict:
    """
    Run the full agent loop for one intervention.
    Returns the result dict from store_intervention_analysis, or an error dict.
    """
    # Step 2: get_intervention
    iv_result = server.call("get_intervention", {"intervention_id": intervention_id})
    if not iv_result["ok"]:
        return iv_result
    iv = iv_result["intervention"]

    # Step 3: get_session
    session_result = server.call("get_session", {"session_id": iv["session_id"]})
    session = session_result.get("session", {}) if session_result["ok"] else {}

    # Step 3b: get_session_topics (grounding context for constructiveness classification).
    # Topics may be plain strings (keyword baseline) or rich dicts {label, description, law_id}
    # from the LLM map-reduce pipeline.  _build_user_message handles both formats.
    topics_result = server.call("get_session_topics", {"session_id": iv["session_id"]})
    session_topics: list = topics_result.get("topics", []) if topics_result.get("ok") else []

    # Step 4: retrieve_context
    ctx_result = server.call(
        "retrieve_context",
        {"intervention_id": intervention_id, "top_k": config["rag"]["top_k"]},
    )
    if not ctx_result["ok"]:
        context_chunks = []
    else:
        context_chunks = ctx_result["context"]
    retrieved_chunk_ids = {c["chunk_id"] for c in context_chunks}

    # Log what we're about to send to the LLM
    speaker = iv.get("raw_speaker", "(unknown)")
    text_preview = iv.get("text", "")[:120].replace("\n", " ")
    print(
        f"  LLM call: speaker={speaker!r}  "
        f"context_chunks={len(context_chunks)}  "
        f"text_preview={text_preview!r}..."
    )

    # Step 5: call LLM (with retries)
    user_msg = _build_user_message(iv, session, context_chunks, session_topics=session_topics)
    last_exc: Exception | None = None
    llm_data: dict | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            llm_data = _call_llm(model, user_msg, client, provider)
            break
        except Exception as exc:
            last_exc = exc
            print(f"  LLM attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    if llm_data is None:
        print(f"  LLM error (all retries exhausted): {last_exc}")
        return {"ok": False, "error": {"code": "LLM_ERROR", "message": str(last_exc)}}

    # Log raw LLM response
    label_raw = llm_data.get("constructiveness_label", "?")
    confidence_raw = llm_data.get("confidence", "?")
    topics_raw = llm_data.get("topics", [])
    reasoning = llm_data.get("reasoning", "")
    evidence_raw = llm_data.get("evidence_chunk_ids", [])
    print(
        f"  LLM response: label={label_raw}  confidence={confidence_raw}  "
        f"topics={topics_raw}  evidence={evidence_raw}"
    )
    if reasoning:
        print(f"  reasoning: {reasoning}")

    # Validate and clean LLM output
    try:
        payload = _validate_llm_response(llm_data, config, retrieved_chunk_ids)
    except ValueError as exc:
        print(f"  Validation error: {exc}")
        return {"ok": False, "error": {"code": "LLM_RESPONSE_INVALID", "message": str(exc)}}

    # Step 6: store via MCP
    store_result = server.call(
        "store_intervention_analysis",
        {"intervention_id": intervention_id, **payload},
    )
    return store_result


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def _build_client(provider: str, model: str):
    """
    Create an openai.OpenAI client configured for the given provider.

    Both openai and ollama use the same openai Python SDK.
    Ollama exposes an OpenAI-compatible REST API at localhost:11434/v1,
    so we just point base_url there and pass a dummy API key (Ollama
    doesn't require authentication, but the SDK requires a non-empty string).
    """
    import openai as openai_module

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )
        print(f"Provider: OpenAI  |  Model: {model}")
        return openai_module.OpenAI(api_key=api_key)

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
        base_url = f"{host}/v1"
        print(f"Provider: Ollama ({host})  |  Model: {model}")
        # Verify Ollama is reachable before starting the loop.
        import urllib.request
        import urllib.error
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3)
        except urllib.error.URLError:
            raise SystemExit(
                f"Cannot reach Ollama at {host}.\n"
                "Start it with: ollama serve"
            )
        return openai_module.OpenAI(api_key="ollama", base_url=base_url)

    raise SystemExit(f"Unknown provider: {provider!r}. Choose 'openai' or 'ollama'.")


def run_agent(
    db_path: Path,
    run_id: str,
    intervention_ids: list[str],
    model: str,
    provider: str = DEFAULT_PROVIDER,
) -> dict:
    """
    Classify all given interventions and return a summary dict.
    """
    client = _build_client(provider, model)

    total = len(intervention_ids)
    classified = 0
    errors = 0
    error_log: list[dict] = []

    with MCPServer(db_path=db_path, run_id=run_id) as server:
        config_result = server.call("get_run_config", {})
        if not config_result["ok"]:
            raise SystemExit(f"get_run_config failed: {config_result}")
        config = config_result["config"]

        for i, intervention_id in enumerate(intervention_ids, 1):
            print(f"\n[{i}/{total}] {intervention_id}")
            result = classify_intervention(
                server=server,
                intervention_id=intervention_id,
                model=model,
                client=client,
                config=config,
                provider=provider,
            )
            if result.get("ok"):
                classified += 1
                stored = result.get("stored", {})
                label = stored.get("constructiveness_label", "?")
                confidence = stored.get("confidence", 0.0)
                topics = stored.get("topics", [])
                print(
                    f"  → stored: label={label}  confidence={confidence:.2f}  "
                    f"topics={topics}"
                )
            else:
                errors += 1
                err_info = result.get("error", {})
                print(
                    f"  ✗ error: {err_info.get('code')}: {err_info.get('message')}"
                )
                error_log.append({"intervention_id": intervention_id, "error": err_info})

    return {
        "total": total,
        "classified": classified,
        "errors": errors,
        "error_log": error_log,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _load_intervention_ids(
    db_path: Path,
    run_id: str,
    stenogram_list_path: Path | None,
    session_id: str | None = None,
) -> list[str]:
    """
    Return all intervention IDs for matched members in the selected stenograms.
    Skips interventions already classified by the LLM in this run.
    If session_id is given, restrict to that session regardless of stenogram list.
    """
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

        # When paths is non-empty, restrict to interventions from those stenograms.
        # When empty (no new stenograms, resuming LLM work), scan all interventions.
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
        description="LLM agent: classify intervention constructiveness via LLM + MCP."
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
            f"{DEFAULT_MODEL_OPENAI!r} for openai. "
            "Also reads OPENAI_MODEL env var for openai provider."
        ),
    )
    parser.add_argument(
        "--session-id",
        help="Classify all interventions for a single session. Skips the stenogram list.",
    )
    parser.add_argument(
        "--intervention-id",
        help="Classify a single intervention (for debugging). Skips the stenogram list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Classify at most N interventions (0 = no limit). Useful for testing.",
    )
    args = parser.parse_args()

    if not args.run_id:
        print("ERROR: --run-id or VOTEZ_RUN_ID required.")
        return 1

    # Resolve model default based on provider
    if args.model.strip():
        model = args.model.strip()
    elif args.provider == "openai":
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_OPENAI)
    else:
        model = DEFAULT_MODEL_OLLAMA

    db_path = Path(args.db_path)
    init_db(db_path)

    if args.intervention_id:
        intervention_ids = [args.intervention_id]
    else:
        list_path = Path(args.stenogram_list_path) if args.stenogram_list_path else None
        intervention_ids = _load_intervention_ids(db_path, args.run_id, list_path, session_id=args.session_id)

    if args.limit > 0:
        intervention_ids = intervention_ids[: args.limit]

    if not intervention_ids:
        print("No unclassified interventions found. Nothing to do.")
        return 0

    print(
        f"LLM agent: classifying {len(intervention_ids)} intervention(s) "
        f"(run_id={args.run_id})"
    )

    summary = run_agent(
        db_path=db_path,
        run_id=args.run_id,
        intervention_ids=intervention_ids,
        model=model,
        provider=args.provider,
    )

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
