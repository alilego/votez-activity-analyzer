#!/usr/bin/env python3
"""
Export frontend aggregate files that are not part of the analyzer subfolders.

These files are imported directly by the frontend:

* data/activity_analizer/adopted_law_reader_summaries.json
* data/activity_analizer/analysis_intervals.json
* data/evolutia_partidelor_camera_deputatilor.json
* data/evolutia_partidelor_senat.json

The party-evolution export preserves historical snapshots from an existing
output/frontend file and appends or replaces only the current DB snapshot.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from init_db import DEFAULT_DB_PATH, init_db


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_FRONTEND_DATA_DIR = Path("../votez-frontend/data")

ADOPTED_LAW_READER_SUMMARIES = "adopted_law_reader_summaries.json"
ANALYSIS_INTERVALS = "analysis_intervals.json"
CAMERA_EVOLUTION = "evolutia_partidelor_camera_deputatilor.json"
SENAT_EVOLUTION = "evolutia_partidelor_senat.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_json_object(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _date_parts(date_text: str) -> tuple[int, int, int]:
    date = datetime.fromisoformat(date_text[:10])
    return date.year, date.month, date.day


def _snapshot_date_from_db(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(substr(updated_at, 1, 10)) FROM members").fetchone()
    value = str(row[0]) if row and row[0] else ""
    return value or datetime.now(timezone.utc).date().isoformat()


def _party_snapshot(conn: sqlite3.Connection, *, chamber: str, date_text: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT COALESCE(party_id, 'Neafiliaţi') AS party_id, COUNT(*) AS seats
        FROM members
        WHERE chamber = ?
        GROUP BY COALESCE(party_id, 'Neafiliaţi')
        ORDER BY seats DESC, party_id
        """,
        (chamber,),
    ).fetchall()
    total = sum(int(row[1]) for row in rows)
    year, month, day = _date_parts(date_text)
    return {
        "year": year,
        "month": month,
        "day": day,
        "total_seats": total,
        "parties": [
            {
                "name": str(party_id),
                "seats": int(seats),
                "percentage": round((int(seats) / total) * 100, 2) if total else 0.0,
            }
            for party_id, seats in rows
        ],
    }


def _snapshot_key(snapshot: dict[str, Any]) -> tuple[int, int, int]:
    return int(snapshot["year"]), int(snapshot["month"]), int(snapshot["day"])


def _load_history(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        data = _read_json(path)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def _upsert_snapshot(history: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    target = _snapshot_key(snapshot)
    filtered = [item for item in history if _snapshot_key(item) != target]
    filtered.append(snapshot)
    return sorted(filtered, key=_snapshot_key)


def _first_date(history: list[dict[str, Any]]) -> str | None:
    if not history:
        return None
    first = min(history, key=_snapshot_key)
    return f"{int(first['year']):04d}-{int(first['month']):02d}-{int(first['day']):02d}"


def _last_date(history: list[dict[str, Any]]) -> str | None:
    if not history:
        return None
    last = max(history, key=_snapshot_key)
    return f"{int(last['year']):04d}-{int(last['month']):02d}-{int(last['day']):02d}"


def export_reader_summaries(conn: sqlite3.Connection, output_path: Path) -> int:
    rows = conn.execute(
        """
        SELECT law_id, source_url, identifier, adopted_law_identifier,
               adopted_law_reader_summary
        FROM dep_act_laws
        WHERE adopted_law_reader_summary IS NOT NULL
          AND adopted_law_reader_summary <> ''
        ORDER BY law_id
        """
    ).fetchall()
    out: dict[str, Any] = {}
    for law_id, source_url, identifier, adopted_law_identifier, summary_raw in rows:
        summary = _safe_json_object(summary_raw)
        if not summary:
            continue
        item = dict(summary)
        item["law_id"] = str(law_id)
        item["source_url"] = source_url
        item["identifier"] = identifier
        item["adopted_law_identifier"] = adopted_law_identifier
        out[str(law_id)] = item
    _write_json(output_path, out)
    return len(out)


def _date_range(conn: sqlite3.Connection, sql: str) -> dict[str, str | None]:
    row = conn.execute(sql).fetchone()
    return {
        "startDate": str(row[0])[:10] if row and row[0] else None,
        "endDate": str(row[1])[:10] if row and row[1] else None,
    }


def export_analysis_intervals(
    conn: sqlite3.Connection,
    output_path: Path,
    *,
    existing_path: Path,
    camera_history: list[dict[str, Any]],
    senat_history: list[dict[str, Any]],
) -> dict[str, Any]:
    existing = _read_json(output_path)
    if not isinstance(existing, dict):
        existing = _read_json(existing_path)
    intervals: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    starts = [date for date in (_first_date(camera_history), _first_date(senat_history)) if date]
    ends = [date for date in (_last_date(camera_history), _last_date(senat_history)) if date]
    intervals["partyEvolution"] = {
        "startDate": min(starts) if starts else None,
        "endDate": max(ends) if ends else None,
    }

    interventions_range = _date_range(
        conn,
        "SELECT MIN(substr(session_date, 1, 10)), MAX(substr(session_date, 1, 10)) FROM interventions_raw",
    )
    if interventions_range["startDate"] and interventions_range["endDate"]:
        intervals["regional"] = interventions_range
        intervals["constructiveInterventions"] = interventions_range

    laws_range = _date_range(
        conn,
        "SELECT MIN(substr(vote_date, 1, 10)), MAX(substr(vote_date, 1, 10)) FROM dep_act_laws_votes",
    )
    if laws_range["startDate"] and laws_range["endDate"]:
        intervals["laws"] = laws_range

    intervals["generatedAt"] = _utc_now_iso()
    _write_json(output_path, intervals)
    return intervals


def export_frontend_auxiliary(
    *,
    db_path: Path,
    output_dir: Path,
    frontend_data_dir: Path,
) -> dict[str, Any]:
    init_db(db_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend_activity_dir = frontend_data_dir / "activity_analizer"

    with sqlite3.connect(db_path) as conn:
        reader_count = export_reader_summaries(
            conn,
            output_dir / ADOPTED_LAW_READER_SUMMARIES,
        )
        snapshot_date = _snapshot_date_from_db(conn)
        camera_history = _load_history(
            output_dir / CAMERA_EVOLUTION,
            frontend_data_dir / CAMERA_EVOLUTION,
        )
        senat_history = _load_history(
            output_dir / SENAT_EVOLUTION,
            frontend_data_dir / SENAT_EVOLUTION,
        )
        camera_history = _upsert_snapshot(
            camera_history,
            _party_snapshot(conn, chamber="deputat", date_text=snapshot_date),
        )
        senat_history = _upsert_snapshot(
            senat_history,
            _party_snapshot(conn, chamber="senator", date_text=snapshot_date),
        )
        _write_json(output_dir / CAMERA_EVOLUTION, camera_history)
        _write_json(output_dir / SENAT_EVOLUTION, senat_history)
        intervals = export_analysis_intervals(
            conn,
            output_dir / ANALYSIS_INTERVALS,
            existing_path=frontend_activity_dir / ANALYSIS_INTERVALS,
            camera_history=camera_history,
            senat_history=senat_history,
        )

    return {
        "reader_summaries": reader_count,
        "camera_snapshots": len(camera_history),
        "senat_snapshots": len(senat_history),
        "party_evolution_end": intervals.get("partyEvolution", {}).get("endDate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export auxiliary frontend aggregate JSON files.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--frontend-data-dir", default=str(DEFAULT_FRONTEND_DATA_DIR))
    args = parser.parse_args()

    result = export_frontend_auxiliary(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
        frontend_data_dir=Path(args.frontend_data_dir),
    )
    print(
        "  Wrote auxiliary frontend exports: "
        f"{result['reader_summaries']} reader summaries, "
        f"{result['camera_snapshots']} camera snapshot(s), "
        f"{result['senat_snapshots']} senate snapshot(s), "
        f"party evolution through {result['party_evolution_end']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
