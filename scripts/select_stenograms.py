#!/usr/bin/env python3
"""
Select only new/changed stenogram files for incremental runs.

Compares files in input/stenograme against the local SQLite state table
`processed_stenograms` and returns the files that should be processed now.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path("state/state.sqlite")
DEFAULT_INPUT_DIR = Path("input/stenograme")


@dataclass(frozen=True)
class StenogramCandidate:
    path: str
    content_hash: str
    file_mtime_ns: int
    reason: str  # "new" | "changed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_processed_map(conn: sqlite3.Connection) -> dict[str, tuple[str, int]]:
    rows = conn.execute(
        """
        SELECT stenogram_path, content_hash, file_mtime_ns
        FROM processed_stenograms
        """
    ).fetchall()
    return {row[0]: (row[1], int(row[2])) for row in rows}


def _require_state_tables(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='processed_stenograms'
        """
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Missing table 'processed_stenograms'. "
            "Run: python scripts/init_db.py"
        )


def select_candidates(db_path: Path, input_dir: Path, repo_root: Path) -> list[StenogramCandidate]:
    if not db_path.exists():
        raise RuntimeError(f"State DB not found at '{db_path}'. Run: python scripts/init_db.py")
    if not input_dir.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        _require_state_tables(conn)
        processed = _load_processed_map(conn)

    candidates: list[StenogramCandidate] = []
    for file_path in sorted(input_dir.glob("*.json")):
        stat = file_path.stat()
        rel_path = file_path.resolve().relative_to(repo_root.resolve()).as_posix()
        file_hash = _sha256_file(file_path)

        prev = processed.get(rel_path)
        if prev is None:
            candidates.append(
                StenogramCandidate(
                    path=rel_path,
                    content_hash=file_hash,
                    file_mtime_ns=stat.st_mtime_ns,
                    reason="new",
                )
            )
            continue

        prev_hash, _prev_mtime_ns = prev
        if prev_hash != file_hash:
            candidates.append(
                StenogramCandidate(
                    path=rel_path,
                    content_hash=file_hash,
                    file_mtime_ns=stat.st_mtime_ns,
                    reason="changed",
                )
            )
    return candidates


def _print_text(candidates: list[StenogramCandidate]) -> None:
    if not candidates:
        print("No new/changed stenograms found.")
        return
    print(f"Found {len(candidates)} new/changed stenogram(s):")
    for item in candidates:
        print(f"- {item.path} [{item.reason}]")


def _print_json(candidates: list[StenogramCandidate]) -> None:
    payload = {
        "count": len(candidates),
        "candidates": [
            {
                "path": c.path,
                "reason": c.reason,
                "content_hash": c.content_hash,
                "file_mtime_ns": c.file_mtime_ns,
            }
            for c in candidates
        ],
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select new/changed stenograms for incremental processing."
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
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    input_dir = Path(args.input_dir)
    repo_root = Path.cwd()

    try:
        candidates = select_candidates(db_path=db_path, input_dir=input_dir, repo_root=repo_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.format == "json":
        _print_json(candidates)
    else:
        _print_text(candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
