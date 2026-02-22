#!/usr/bin/env python3
"""
Mark current new/changed stenograms as processed in local SQLite state.

This script reuses the same selection logic as `select_stenograms.py` and
upserts entries into `processed_stenograms`.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from select_stenograms import (
    DEFAULT_DB_PATH,
    DEFAULT_INPUT_DIR,
    StenogramCandidate,
    select_candidates,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_run_id() -> str:
    return f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _ensure_run_exists(conn: sqlite3.Connection, run_id: str) -> None:
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO runs (
            run_id, started_at, finished_at, status,
            sessions_processed, interventions_total,
            interventions_classified, unmatched_speakers
        )
        VALUES (?, ?, ?, 'completed', 0, 0, 0, 0)
        ON CONFLICT(run_id) DO NOTHING
        """,
        (run_id, now, now),
    )


def mark_candidates(
    conn: sqlite3.Connection, candidates: list[StenogramCandidate], run_id: str
) -> int:
    if not candidates:
        return 0
    for item in candidates:
        conn.execute(
            """
            INSERT INTO processed_stenograms (
                stenogram_path,
                content_hash,
                file_mtime_ns,
                last_processed_run_id,
                last_processed_at
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(stenogram_path) DO UPDATE SET
                content_hash = excluded.content_hash,
                file_mtime_ns = excluded.file_mtime_ns,
                last_processed_run_id = excluded.last_processed_run_id,
                last_processed_at = CURRENT_TIMESTAMP
            """,
            (
                item.path,
                item.content_hash,
                item.file_mtime_ns,
                run_id,
            ),
        )
    return len(candidates)


def mark_processed(db_path: Path, input_dir: Path, repo_root: Path, run_id: str) -> int:
    candidates = select_candidates(db_path=db_path, input_dir=input_dir, repo_root=repo_root)
    if not candidates:
        print("No new/changed stenograms to mark.")
        return 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        _ensure_run_exists(conn, run_id)
        marked = mark_candidates(conn, candidates, run_id)
        conn.commit()

    print(f"Marked {marked} stenogram(s) as processed (run_id={run_id}).")
    for item in candidates:
        print(f"- {item.path} [{item.reason}]")
    return marked


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mark new/changed stenograms as processed in SQLite state."
    )
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
    args = parser.parse_args()

    db_path = Path(args.db_path)
    input_dir = Path(args.input_dir)
    repo_root = Path.cwd()

    if not db_path.exists():
        print(f"ERROR: State DB not found at '{db_path}'. Run: python scripts/init_db.py")
        return 1

    try:
        mark_processed(
            db_path=db_path,
            input_dir=input_dir,
            repo_root=repo_root,
            run_id=args.run_id,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
