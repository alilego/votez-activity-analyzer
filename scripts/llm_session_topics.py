#!/usr/bin/env python3
"""
LLM-based session topic extraction.

For each session that has only keyword-baseline topics (topics_source='keyword_baseline_v1'),
or no topics at all:

  1. Fetch session initial_notes via MCP get_session
  2. Fetch the first N substantial chunks from session_chunks (early speeches)
  3. Call LLM: "What are the main legislative/policy topics of this session?"
  4. Store result via MCP store_session_topics (source='llm_v1')

The LLM-derived topics are then used by the intervention LLM pass as grounding context,
replacing the keyword-taxonomy topics that the baseline builds.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_OLLAMA = "llama3.1:8b-8k"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# How many chunks to sample for topic extraction.
# Chunks are sampled evenly across the full session (not just the first N)
# so that topics from all agenda items are represented.
# 20 evenly-spaced chunks ≈ 2,700 tokens — fits any model comfortably.
SAMPLE_CHUNKS = 20

MAX_TOPICS_PER_SESSION = 20
MAX_TOPIC_LENGTH = 64

MAX_RETRIES = 3
RETRY_DELAY_S = 10
# Hard timeout per LLM request. If Ollama hangs this raises httpx.ReadTimeout
# which the retry loop catches. 180s gives headroom for slow M1 inference.
LLM_REQUEST_TIMEOUT_S = 180
# Ollama's default runtime context is 4096 tokens — too small for session prompts
# (~5k tokens). 8192 is sufficient and allocates much faster than 16384 on M1.
OLLAMA_NUM_CTX = 8192

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Two-step prompts
#
# Why two steps?
# Small local models (llama3.1:8b) produce much better content when allowed to
# reason freely in prose first. When forced directly into JSON they fall back to
# generic terms. Step 1 extracts rich prose; Step 2 distils it into structured JSON.
# ---------------------------------------------------------------------------

# Step 1: free-form analysis in Romanian — model describes the session content.
STEP1_SYSTEM = """Ești un analist expert al ședințelor Parlamentului României.

Citește fragmentele de stenogramă și listează TOATE subiectele specifice menționate.

Formatul răspunsului — o listă cu linii separate, fiecare linie = un subiect specific:
- buget de stat 2025
- Autostrada A7 sector Siret-Pașcani
- dotare școli gimnaziale
- ambulanțe județ Sibiu
- Astra Film Festival
- etc.

Reguli:
- Fii SPECIFIC: menționează sumele, locațiile, instituțiile, numerele de lege din text.
- NU scrie generic: NU "buget", NU "educație", NU "infrastructură" — scrie exact ce apare în text.
- Listează cel puțin 10 subiecte dacă textul le conține.
- Răspunde DOAR cu lista de puncte, fără introducere."""

# Step 2: extract structured JSON from the prose analysis.
STEP2_SYSTEM = """You extract a structured topic list from a parliamentary session analysis.

Given a plain-text analysis of a Romanian parliamentary session, extract 5–20 specific topics.

Rules:
- Topics must be specific to this session (e.g. "buget de stat 2025", "PL-x 45/2025", "spital Suceava").
- Do NOT use generic terms like "legislativ", "politica", "economie", "procedura".
- Each topic: 2–6 words, in Romanian.
- Deduplicate similar topics.

Respond with ONLY valid JSON, nothing else:
{"topics": ["topic1", "topic2", ...], "reasoning": "one sentence summary"}"""


def _build_session_message(session: dict, chunks: list[dict]) -> str:
    """Build the user message for step 1 (free-form analysis)."""
    parts: list[str] = []
    parts.append(
        f"Ședința ID: {session.get('session_id', '')}  "
        f"Data: {session.get('session_date', '')}"
    )
    initial_notes = (session.get("initial_notes") or "").strip()
    if initial_notes:
        parts.append(f"Note inițiale:\n{initial_notes}")

    if chunks:
        # Number each fragment so the model treats each as separate — reduces tendency to summarize.
        frag_parts: list[str] = ["Fragmente din stenogramă:"]
        for i, chunk in enumerate(chunks, 1):
            frag_parts.append(f"[Fragment {i}]\n{chunk['text'].strip()}")
        parts.append("\n\n".join(frag_parts))

    parts.append("Listează TOATE subiectele specifice din fragmentele de mai sus, câte unul pe linie.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM calls (two-step)
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


def _chat(client, system: str, user: str, json_mode: bool = False) -> str:
    """Single LLM call. Returns raw string content."""
    extra_kwargs: dict = {}
    if json_mode:
        extra_kwargs["response_format"] = {"type": "json_object"}
    # For Ollama, explicitly set num_ctx so large session prompts are not
    # silently truncated (Ollama's default runtime context is only 4096 tokens).
    # Ollama's /v1/chat/completions endpoint reads num_ctx at the top level of the
    # request body, not nested under "options". extra_body merges into the top level.
    if getattr(client, "_provider", "") == "ollama":
        extra_kwargs["extra_body"] = {"num_ctx": OLLAMA_NUM_CTX}
    response = client.chat.completions.create(
        model=client._model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        timeout=LLM_REQUEST_TIMEOUT_S,
        **extra_kwargs,
    )
    return response.choices[0].message.content or ""


def _call_llm_two_step(session_message: str, client, provider: str) -> dict:
    """
    Two-step extraction:
      Step 1 — free-form prose analysis (no JSON constraint, model reasons freely).
      Step 2 — extract structured JSON from the prose.

    Why two steps? llama3.1:8b produces rich, accurate content in free-form Romanian
    but defaults to generic terms when forced into JSON directly. Separating reasoning
    from formatting produces much better topic quality.
    """
    # Step 1: free-form analysis.
    prose = _chat(client, STEP1_SYSTEM, session_message, json_mode=False)
    print(f"  Step 1 prose ({len(prose)} chars):")
    for line in prose.strip().splitlines()[:30]:
        print(f"    {line}")

    # Step 2: extract JSON from the prose.
    extraction_user = (
        f"Analiza ședinței:\n{prose}\n\n"
        "Extrage lista de subiecte specifice ca JSON."
    )
    raw = _chat(client, STEP2_SYSTEM, extraction_user, json_mode=True)

    # Strip markdown fences defensively.
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.startswith("```")).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Step 2 returned non-JSON: {raw!r}") from exc


def _validate_topics(data: dict) -> list[str]:
    topics_raw = data.get("topics", [])
    if not isinstance(topics_raw, list):
        return []
    topics: list[str] = []
    seen: set[str] = set()
    for t in topics_raw:
        t = str(t).strip()
        if t and len(t) <= MAX_TOPIC_LENGTH and t not in seen:
            topics.append(t)
            seen.add(t)
        if len(topics) >= MAX_TOPICS_PER_SESSION:
            break
    return topics


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
) -> dict:
    # Fetch session metadata.
    session_result = server.call("get_session", {"session_id": session_id})
    if not session_result["ok"]:
        return session_result
    session = session_result["session"]

    # Fetch all chunks for this session, then sample evenly across the full timeline.
    # This ensures topics from all agenda items are represented, not just the opening.
    all_rows = conn.execute(
        """
        SELECT chunk_id, chunk_type, text
        FROM session_chunks
        WHERE session_id = ?
        ORDER BY chunk_index ASC
        """,
        (session_id,),
    ).fetchall()

    # Always include session_notes (first chunk if present).
    notes_rows = [r for r in all_rows if r["chunk_type"] == "session_notes"]

    # Filter to substantive speeches only (>=200 chars) before sampling.
    # Short chunks are mostly voting procedural lines ("Marginal 37 - vot, vă rog")
    # which add noise and no topical signal.
    substantive_rows = [
        r for r in all_rows
        if r["chunk_type"] != "session_notes" and len(r["text"]) >= 200
    ]
    # Fall back to all non-notes if too few substantive chunks.
    speech_pool = substantive_rows if len(substantive_rows) >= 5 else [
        r for r in all_rows if r["chunk_type"] != "session_notes"
    ]

    n_speeches = SAMPLE_CHUNKS - len(notes_rows)
    if len(speech_pool) <= n_speeches:
        sampled_speeches = speech_pool
    else:
        step = len(speech_pool) / n_speeches
        sampled_speeches = [speech_pool[int(i * step)] for i in range(n_speeches)]

    sampled_rows = notes_rows + sampled_speeches
    chunks = [{"chunk_id": r["chunk_id"], "chunk_type": r["chunk_type"], "text": r["text"]} for r in sampled_rows]

    initial_notes = session.get("initial_notes", "")
    text_preview = initial_notes[:80].replace("\n", " ") if initial_notes else "(no notes)"
    print(
        f"  LLM call: session={session_id}  date={session.get('session_date', '')}  "
        f"sampled={len(chunks)}/{len(all_rows)} chunks  notes_preview={text_preview!r}..."
    )

    session_msg = _build_session_message(session, chunks)
    last_exc: Exception | None = None
    llm_data: dict | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            llm_data = _call_llm_two_step(session_msg, client, provider)
            break
        except Exception as exc:
            last_exc = exc
            print(f"  LLM attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    if llm_data is None:
        return {"ok": False, "error": {"code": "LLM_ERROR", "message": str(last_exc)}}

    topics = _validate_topics(llm_data)
    reasoning = llm_data.get("reasoning", "")
    print(f"  LLM response: {len(topics)} topics  reasoning={reasoning!r}")
    print(f"  topics={topics}")

    topics_source = f"llm_v1:{client._model}"
    store_result = server.call(
        "store_session_topics",
        {"session_id": session_id, "topics": topics, "topics_source": topics_source},
    )
    return store_result


# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------

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
                )
                if result.get("ok"):
                    extracted += 1
                    stored = result.get("stored", {})
                    print(f"  → stored: {len(stored.get('topics', []))} topics  source={stored.get('topics_source')}")
                else:
                    errors += 1
                    err_info = result.get("error", {})
                    print(f"  ✗ error: {err_info.get('code')}: {err_info.get('message')}")
                    error_log.append({"session_id": session_id, "error": err_info})

    return {"total": total, "extracted": extracted, "errors": errors, "error_log": error_log}


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
    args = parser.parse_args()

    if not args.run_id:
        print("ERROR: --run-id or VOTEZ_RUN_ID required.")
        return 1

    model = args.model.strip()
    if not model:
        model = DEFAULT_MODEL_OLLAMA if args.provider == "ollama" else DEFAULT_MODEL_OPENAI

    db_path = Path(args.db_path)
    init_db(db_path)

    if args.session_id:
        session_ids = [args.session_id]
    else:
        list_path = Path(args.stenogram_list_path) if args.stenogram_list_path else None
        session_ids = _load_session_ids(db_path, list_path, model, reprocess=args.reprocess_session_topics)

    if not session_ids:
        print("No sessions need LLM topic extraction. Nothing to do.")
        return 0

    if args.limit > 0:
        session_ids = session_ids[: args.limit]

    reprocess_note = "  (reprocess=True — overwriting existing LLM topics)" if args.reprocess_session_topics else ""
    limit_note = f"  (limit={args.limit})" if args.limit > 0 else ""
    print(f"Session topic extraction: {len(session_ids)} session(s) (run_id={args.run_id}){reprocess_note}{limit_note}")

    summary = run_session_topics(
        db_path=db_path,
        run_id=args.run_id,
        session_ids=session_ids,
        model=model,
        provider=args.provider,
        reprocess=args.reprocess_session_topics,
    )

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
