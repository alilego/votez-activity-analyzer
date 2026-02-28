#!/usr/bin/env python3
"""
Inspect RAG retrieval results for a specific intervention.

Prints the intervention text and the top-k retrieved context chunks with their
scores and retrieval reason (session_notes / neighbor / similarity).

Usage examples:
  # By session ID and speech index:
  python3 scripts/inspect_retrieval.py --session-id 8846 --speech-index 10

  # By intervention ID (stored in DB):
  python3 scripts/inspect_retrieval.py --intervention-id iv:abc123:10

  # Override top-k (default 8):
  python3 scripts/inspect_retrieval.py --session-id 8846 --speech-index 10 --top-k 5

  # Show full chunk text (default is truncated to 300 chars):
  python3 scripts/inspect_retrieval.py --session-id 8846 --speech-index 10 --full-text
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

import rag_store
from init_db import DEFAULT_DB_PATH


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _resolve_by_db(db_path: Path, intervention_id: str) -> tuple[str, str, int | None]:
    """Return (session_id, intervention_text, speech_index) from DB."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT session_id, text, speech_index FROM interventions_raw WHERE intervention_id = ?",
            (intervention_id,),
        ).fetchone()
    if row is None:
        raise SystemExit(f"Intervention not found in DB: {intervention_id}")
    return row[0], row[1], row[2]


def _resolve_by_session_speech(
    db_path: Path, session_id: str, speech_index: int
) -> tuple[str, str, int]:
    """Return (session_id, intervention_text, speech_index) from DB."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT text FROM interventions_raw WHERE session_id = ? AND speech_index = ?",
            (session_id, speech_index),
        ).fetchone()
    if row is None:
        raise SystemExit(
            f"No intervention found for session_id={session_id} speech_index={speech_index}. "
            "Make sure the pipeline has been run and that speech has a matched member."
        )
    return session_id, row[0], speech_index


def _print_divider(char: str = "-", width: int = 80) -> None:
    print(char * width)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect RAG retrieval for a specific intervention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--intervention-id", help="Intervention ID (from DB)")
    group.add_argument("--session-id", help="Session ID (use with --speech-index)")
    parser.add_argument(
        "--speech-index",
        type=int,
        help="Speech index within the session (required when using --session-id)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=rag_store.DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {rag_store.DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Show full chunk text instead of truncated preview",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    if args.session_id and args.speech_index is None:
        parser.error("--speech-index is required when using --session-id")

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}. Run the pipeline first.")

    # Resolve intervention.
    if args.intervention_id:
        session_id, text, speech_index = _resolve_by_db(db_path, args.intervention_id)
    else:
        session_id, text, speech_index = _resolve_by_session_speech(
            db_path, args.session_id, args.speech_index
        )

    preview_len = None if args.full_text else 300

    _print_divider("=")
    print(f"INTERVENTION  session={session_id}  speech_index={speech_index}")
    _print_divider("=")
    print(_truncate(text, preview_len) if preview_len else text.strip())
    print()

    # Check index exists.
    meta_path = rag_store._meta_path(session_id)
    if not meta_path.exists():
        print(
            f"No RAG index found for session {session_id}.\n"
            "Run the pipeline first (python3 scripts/run_pipeline.py --dry-run rebuilds indexes).\n"
            "Or run: python3 scripts/run_pipeline.py"
        )
        return 1

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    total_chunks = len(meta.get("chunks", []))

    chunks = rag_store.retrieve_chunks(
        session_id=session_id,
        intervention_text=text,
        intervention_speech_index=speech_index,
        top_k=args.top_k,
    )

    print(f"Retrieved {len(chunks)} chunk(s) from {total_chunks} total in session  [top_k={args.top_k}]")
    _print_divider()

    reason_labels = {
        "session_notes": "SESSION_NOTES",
        "neighbor":      "NEIGHBOR     ",
        "similarity":    "SIMILARITY   ",
    }

    for i, chunk in enumerate(chunks, 1):
        label = reason_labels.get(chunk.reason, chunk.reason.upper())
        neighbor_info = ""
        if chunk.source_speech_index is not None:
            delta = ""
            if speech_index is not None and chunk.reason == "neighbor":
                d = chunk.source_speech_index - speech_index
                delta = f"  Δ={d:+d}"
            neighbor_info = f"  speech_idx={chunk.source_speech_index}{delta}"
        score_str = f"score={chunk.score:.4f}" if chunk.reason == "similarity" else "score=pinned"
        print(
            f"[{i:02d}] {label}  {score_str}  chunk_id={chunk.chunk_id}{neighbor_info}"
        )
        chunk_text = chunk.text
        if preview_len:
            chunk_text = _truncate(chunk_text, preview_len)
        print(f"     {chunk_text}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
