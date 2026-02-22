#!/usr/bin/env python3
"""
Minimal orchestrator for incremental runs.

Flow:
1) Select new/changed stenograms
2) Execute analyzer command
3) Export frontend artifacts
4) Mark files as processed only if analyzer + export succeed
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

    # Always bootstrap DB on pipeline start:
    # - creates file if missing
    # - creates schema if missing/uninitialized
    init_db(db_path)

    try:
        candidates = select_candidates(db_path=db_path, input_dir=input_dir, repo_root=repo_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not candidates:
        print("No new/changed stenograms found. Nothing to process.")
        return 0

    if args.dry_run:
        print(f"Dry run: {len(candidates)} file(s) selected.")
        for item in candidates:
            print(f"- {item.path} [{item.reason}]")
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

    print(f"Running analyzer for {len(candidates)} file(s) (run_id={run_id})...")
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
            print(f"Analyzer failed with exit code {proc.returncode}. Nothing was marked processed.")
            return proc.returncode

        if not args.skip_export:
            export_cmd = [sys.executable, "scripts/export_outputs.py", "--db-path", str(db_path)]
            export_proc = subprocess.run(export_cmd, env=env)
            if export_proc.returncode != 0:
                _finish_run(conn, run_id, "failed", 0)
                conn.commit()
                print(
                    f"Export failed with exit code {export_proc.returncode}. "
                    "Nothing was marked processed."
                )
                return export_proc.returncode

        marked = mark_candidates(conn, candidates, run_id)
        _finish_run(conn, run_id, "completed", marked)
        conn.commit()

    print(
        f"Run completed. Marked {marked} stenogram(s) as processed (run_id={run_id}). "
        f"{'Export skipped.' if args.skip_export else 'Outputs exported to outputs/.'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
