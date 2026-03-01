#!/usr/bin/env python3
"""
Minimal orchestrator for incremental runs.

Flow:
1) Select new/changed stenograms
2) Execute analyzer command  (speaker resolution, chunking, RAG index, baseline labels)
3) Optionally run LLM agent  (LLM classification via MCP, when --analyzer-mode=llm)
4) Export frontend artifacts
5) Mark files as processed only if all steps succeed
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from init_db import init_db
from mark_processed_stenograms import mark_candidates
from select_stenograms import DEFAULT_DB_PATH, DEFAULT_INPUT_DIR, select_candidates


def _default_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _insert_running_run(conn: sqlite3.Connection, run_id: str) -> None:
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO runs (
            run_id, started_at, status,
            sessions_processed, interventions_total,
            interventions_classified, unmatched_speakers
        )
        VALUES (?, ?, 'running', 0, 0, 0, 0)
        ON CONFLICT(run_id) DO UPDATE SET
            started_at = excluded.started_at,
            status = 'running'
        """,
        (run_id, now),
    )


def _finish_run(conn: sqlite3.Connection, run_id: str, status: str, sessions_processed: int) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?,
            finished_at = ?,
            sessions_processed = ?
        WHERE run_id = ?
        """,
        (status, _utc_now_iso(), sessions_processed, run_id),
    )


def _write_candidate_file(run_id: str, candidates: list[str]) -> Path:
    run_inputs_dir = Path("state/run_inputs")
    run_inputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_inputs_dir / f"{run_id}_stenograms.json"
    out_path.write_text(json.dumps({"run_id": run_id, "files": candidates}, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run incremental analyzer pipeline.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Path to stenogram directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--run-id",
        default=_default_run_id(),
        help="Run identifier used in state metadata",
    )
    parser.add_argument(
        "--analyzer-cmd",
        default="",
        help=(
            "Analyzer command to execute. If omitted, defaults to "
            "'python scripts/analyze_interventions.py'. "
            "The command receives env vars: VOTEZ_RUN_ID and VOTEZ_STENOGRAM_LIST_PATH."
        ),
    )
    parser.add_argument(
        "--analyzer-mode",
        choices=["baseline", "llm"],
        default="baseline",
        help=(
            "baseline (default): keyword/RAG baseline only. "
            "llm: run baseline first, then classify via LLM agent."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider for llm mode (default: ollama).",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Model name for LLM mode (default: llama3.1:8b for ollama, gpt-4o-mini for openai).",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=0,
        help="Classify at most N interventions in LLM mode (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected files and exit.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip output export step (not recommended).",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    input_dir = Path(args.input_dir)
    repo_root = Path.cwd()
    run_id = args.run_id

    print(f"\n{'='*60}")
    print(f"Pipeline start  run_id={run_id}")
    print(f"  DB:        {db_path}")
    print(f"  Input dir: {input_dir}")
    print(f"  Mode:      {args.analyzer_mode}")
    if args.analyzer_mode == "llm":
        provider = args.llm_provider
        model_hint = args.llm_model or ("llama3.1:8b" if provider == "ollama" else "gpt-4o-mini")
        print(f"  Provider:  {provider}  model={model_hint}")
        print(f"  LLM steps: 3b) session topics  3c) intervention classification")
    print(f"{'='*60}\n")

    # Always bootstrap DB on pipeline start:
    # - creates file if missing
    # - creates schema if missing/uninitialized
    print("Step 1/4  Initializing database...")
    init_db(db_path)
    print(f"  DB ready: {db_path}")

    print("\nStep 2/4  Selecting stenograms...")
    try:
        candidates = select_candidates(db_path=db_path, input_dir=input_dir, repo_root=repo_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not candidates:
        print("  No new/changed stenograms found. Nothing to process.")
        return 0

    print(f"  {len(candidates)} stenogram(s) selected:")
    for c in candidates:
        print(f"    {c.path}  [{c.reason}]")

    if args.dry_run:
        print("\nDry run: exiting without processing.")
        return 0

    candidate_paths = [c.path for c in candidates]
    candidate_file = _write_candidate_file(run_id, candidate_paths)

    env = os.environ.copy()
    env["VOTEZ_RUN_ID"] = run_id
    env["VOTEZ_STENOGRAM_LIST_PATH"] = str(candidate_file)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        _insert_running_run(conn, run_id)
        conn.commit()

    print(f"\nStep 3/4  Running baseline analyzer ({len(candidates)} file(s))...")
    if args.analyzer_cmd.strip():
        proc = subprocess.run(args.analyzer_cmd, shell=True, env=env)
    else:
        default_cmd = [sys.executable, "scripts/analyze_interventions.py"]
        proc = subprocess.run(default_cmd, env=env)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        if proc.returncode != 0:
            _finish_run(conn, run_id, "failed", 0)
            conn.commit()
            print(f"\nBaseline analyzer failed (exit code {proc.returncode}). Nothing was marked processed.")
            return proc.returncode

        # Mark stenograms as processed right after baseline succeeds.
        # This ensures that if LLM steps are interrupted, re-running won't redo the baseline.
        # LLM steps have their own per-item skip logic (topics_source, relevance_source).
        marked = mark_candidates(conn, candidates, run_id)
        conn.commit()
        print(f"  Stenograms marked as processed: {marked}")

    # Optional LLM passes (run outside conn context to avoid lock contention).
    if args.analyzer_mode == "llm":
        # Step 3b: LLM session topic extraction (runs before intervention classification
        # so that LLM-derived session topics are available as grounding context).
        print(f"\nStep 3b/4  Running LLM session topic extraction...")
        topics_cmd = [
            sys.executable, "scripts/llm_session_topics.py",
            "--provider", args.llm_provider,
        ]
        if args.llm_model.strip():
            topics_cmd += ["--model", args.llm_model.strip()]
        topics_proc = subprocess.run(topics_cmd, env=env)
        if topics_proc.returncode != 0:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                _finish_run(conn, run_id, "failed", 0)
                conn.commit()
            print(f"\nLLM session topic extraction failed (exit code {topics_proc.returncode}). Nothing was marked processed.")
            return topics_proc.returncode

        # Step 3c: LLM intervention classification.
        llm_cmd = [sys.executable, "scripts/llm_agent.py", "--provider", args.llm_provider]
        if args.llm_model.strip():
            llm_cmd += ["--model", args.llm_model.strip()]
        if args.llm_limit > 0:
            llm_cmd += ["--limit", str(args.llm_limit)]
            print(f"\nStep 3c/4  Running LLM intervention classification (limit={args.llm_limit})...")
        else:
            print(f"\nStep 3c/4  Running LLM intervention classification (all interventions)...")
        llm_proc = subprocess.run(llm_cmd, env=env)
        if llm_proc.returncode != 0:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                _finish_run(conn, run_id, "failed", 0)
                conn.commit()
            print(f"\nLLM intervention classification failed (exit code {llm_proc.returncode}). Nothing was marked processed.")
            return llm_proc.returncode

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        if not args.skip_export:
            step_label = "Step 4/4" if args.analyzer_mode == "baseline" else "Step 4/4"
            print(f"\n{step_label}  Exporting frontend JSON artifacts...")
            export_cmd = [sys.executable, "scripts/export_outputs.py", "--db-path", str(db_path)]
            export_proc = subprocess.run(export_cmd, env=env)
            if export_proc.returncode != 0:
                _finish_run(conn, run_id, "failed", 0)
                conn.commit()
                print(f"\nExport failed (exit code {export_proc.returncode}). Nothing was marked processed.")
                return export_proc.returncode
        else:
            print("\nStep 4/4  Export skipped (--skip-export).")

        _finish_run(conn, run_id, "completed", len(candidates))
        conn.commit()

    print(f"\n{'='*60}")
    print(f"Pipeline complete  run_id={run_id}")
    print(f"  Stenograms processed: {len(candidates)}")
    print(f"  Outputs: {'skipped' if args.skip_export else 'outputs/'}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
