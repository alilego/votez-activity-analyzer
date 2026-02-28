#!/usr/bin/env python3
"""
Demo: exercise all MCP tools for one intervention without an LLM.

This script simulates what the LLM agent loop will do for a single
intervention: read config → read session → read intervention → retrieve
context → store a classification → verify it was stored.

It is NOT a test suite — it is a human-readable walkthrough of the full
MCP tool call sequence so you can verify the server works before wiring
in an LLM.

Usage:
    python3 scripts/demo_mcp.py
    python3 scripts/demo_mcp.py --session-id 8846 --speech-index 10
    python3 scripts/demo_mcp.py --intervention-id iv:abc123:10
    python3 scripts/demo_mcp.py --dry-run   # skip the write step
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from init_db import DEFAULT_DB_PATH
from mcp_server import MCPServer


def _pp(label: str, data: dict) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _resolve_intervention_id(db_path: Path, session_id: str, speech_index: int) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT intervention_id FROM interventions_raw "
            "WHERE session_id = ? AND speech_index = ?",
            (session_id, speech_index),
        ).fetchone()
    if row is None:
        raise SystemExit(
            f"No intervention found for session_id={session_id} "
            f"speech_index={speech_index}. Run the pipeline first."
        )
    return row[0]


def _pick_any_intervention(db_path: Path) -> str:
    """Return the first classified intervention with a member_id."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT ir.intervention_id
            FROM interventions_raw ir
            WHERE ir.member_id IS NOT NULL
            ORDER BY ir.session_date, ir.speech_index
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise SystemExit("No classified interventions in DB. Run the pipeline first.")
    return row[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo all MCP tools for one intervention.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--intervention-id", help="Intervention ID to use")
    group.add_argument("--session-id", help="Session ID (use with --speech-index)")
    parser.add_argument("--speech-index", type=int, help="Speech index within session")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the write step (store_intervention_analysis)",
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}. Run the pipeline first.")

    # Resolve intervention ID.
    if args.intervention_id:
        intervention_id = args.intervention_id
    elif args.session_id:
        if args.speech_index is None:
            raise SystemExit("--speech-index required with --session-id")
        intervention_id = _resolve_intervention_id(db_path, args.session_id, args.speech_index)
    else:
        intervention_id = _pick_any_intervention(db_path)
        print(f"No intervention specified — using: {intervention_id}")

    # Use the run that produced this intervention so the FK constraint is satisfied.
    with sqlite3.connect(db_path) as _c:
        _row = _c.execute(
            "SELECT run_id FROM interventions_raw WHERE intervention_id = ?",
            (intervention_id,),
        ).fetchone()
    run_id = _row[0] if _row else "demo_run"

    print(f"\n{'═' * 72}")
    print("  MCP Tool Server Demo")
    print(f"  intervention_id = {intervention_id}")
    print(f"{'═' * 72}")

    with MCPServer(db_path=db_path, run_id=run_id) as server:

        # ── Tool 1: get_run_config ──────────────────────────────────────────
        result = server.call("get_run_config", {})
        _pp("1. get_run_config  →  learn the rules before classifying", result)
        assert result["ok"], f"get_run_config failed: {result}"
        config = result["config"]

        # ── Tool 2: get_intervention ───────────────────────────────────────
        result = server.call("get_intervention", {"intervention_id": intervention_id})
        _pp("2. get_intervention  →  fetch what needs to be classified", result)
        assert result["ok"], f"get_intervention failed: {result}"
        iv = result["intervention"]

        # ── Tool 3: get_session ────────────────────────────────────────────
        result = server.call("get_session", {"session_id": iv["session_id"]})
        _pp("3. get_session  →  understand the session context", result)
        assert result["ok"], f"get_session failed: {result}"

        # ── Tool 4: get_member ─────────────────────────────────────────────
        if iv["member_id"]:
            result = server.call("get_member", {"member_id": iv["member_id"]})
            _pp("4. get_member  →  who is speaking?", result)
            assert result["ok"], f"get_member failed: {result}"
        else:
            print("\n  (skipping get_member — no member_id resolved for this intervention)")

        # ── Tool 5: retrieve_context ───────────────────────────────────────
        result = server.call(
            "retrieve_context",
            {"intervention_id": intervention_id, "top_k": config["rag"]["top_k"]},
        )
        _pp("5. retrieve_context  →  RAG: ground the classification in session context", result)
        assert result["ok"], f"retrieve_context failed: {result}"
        context = result["context"]
        evidence_chunk_ids = [c["chunk_id"] for c in context[:3]]  # use top 3 as evidence

        # ── Tool 6: get_chunk (evidence inspection) ────────────────────────
        if evidence_chunk_ids:
            result = server.call("get_chunk", {"chunk_id": evidence_chunk_ids[0]})
            _pp("6. get_chunk  →  inspect a specific evidence chunk", result)
            assert result["ok"], f"get_chunk failed: {result}"

        # ── Tool 7: store_intervention_analysis ────────────────────────────
        if not args.dry_run:
            write_payload = {
                "intervention_id": intervention_id,
                "constructiveness_label": "neutral",   # demo value — LLM would decide this
                "topics": ["proces legislativ"],        # demo value — LLM would extract these
                "confidence": 0.72,                    # demo value
                "evidence_chunk_ids": evidence_chunk_ids,
            }
            print(f"\n{'─' * 72}")
            print("  7. store_intervention_analysis  →  persist the classification")
            print(f"{'─' * 72}")
            print("  Input:")
            print(json.dumps(write_payload, ensure_ascii=False, indent=2))
            result = server.call("store_intervention_analysis", write_payload)
            _pp("  Output:", result)
            assert result["ok"], f"store_intervention_analysis failed: {result}"

            # Idempotency check: same call again should return the same result.
            result2 = server.call("store_intervention_analysis", write_payload)
            assert result2["ok"], f"Idempotency check failed: {result2}"
            print("\n  ✓ Idempotency check passed (second identical call succeeded)")

            # Validation checks: demonstrate error handling.
            print(f"\n{'─' * 72}")
            print("  8. Validation demo  →  server rejects invalid inputs")
            print(f"{'─' * 72}")

            bad_label = server.call("store_intervention_analysis", {
                **write_payload,
                "constructiveness_label": "definitely_maybe",
            })
            print(f"\n  Bad label → {json.dumps(bad_label, ensure_ascii=False)}")
            assert not bad_label["ok"]

            bad_confidence = server.call("store_intervention_analysis", {
                **write_payload,
                "confidence": 1.5,
            })
            print(f"  Bad confidence → {json.dumps(bad_confidence, ensure_ascii=False)}")
            assert not bad_confidence["ok"]

            bad_topic_length = server.call("store_intervention_analysis", {
                **write_payload,
                "topics": ["x" * 100],
            })
            print(f"  Topic too long → {json.dumps(bad_topic_length, ensure_ascii=False)}")
            assert not bad_topic_length["ok"]

            bad_chunk = server.call("store_intervention_analysis", {
                **write_payload,
                "evidence_chunk_ids": ["ch:nonexistent:999"],
            })
            print(f"  Bad chunk ID → {json.dumps(bad_chunk, ensure_ascii=False)}")
            assert not bad_chunk["ok"]

            print("\n  ✓ All validation checks correctly rejected bad inputs")
        else:
            print("\n  (--dry-run: skipping store_intervention_analysis)")

    print(f"\n{'═' * 72}")
    print("  Demo complete — all tools exercised successfully.")
    print(f"{'═' * 72}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
