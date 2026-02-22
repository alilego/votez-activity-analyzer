#!/usr/bin/env python3
"""
Reset local analyzer state for a clean rerun.

What it clears:
- SQLite state tables (data rows only, keeps schema)
- JSON artifacts under state/run_inputs and state/run_outputs
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from init_db import DEFAULT_DB_PATH, init_db


TABLES_IN_DELETE_ORDER = [
    "unmatched_speakers",
    "session_topics",
    "session_chunks",
    "intervention_analysis",
    "interventions_raw",
    "processed_stenograms",
    "run_outputs",
    "members",
    "runs",
    "metadata",
]


def _clear_db(db_path: Path) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute("PRAGMA foreign_keys = OFF;")
        for table in TABLES_IN_DELETE_ORDER:
            conn.execute(f"DELETE FROM {table}")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.commit()


def _delete_json_files(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    count = 0
    for file_path in dir_path.glob("*.json"):
        file_path.unlink(missing_ok=True)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset local analyzer state.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    _clear_db(db_path)
    # Recreate schema metadata after wipe so tooling can inspect version.
    init_db(db_path)

    deleted_inputs = _delete_json_files(Path("state/run_inputs"))
    deleted_outputs = _delete_json_files(Path("state/run_outputs"))

    print("State reset complete.")
    print(f"- DB cleared: {db_path}")
    print(f"- Deleted run_inputs JSON files: {deleted_inputs}")
    print(f"- Deleted run_outputs JSON files: {deleted_outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
