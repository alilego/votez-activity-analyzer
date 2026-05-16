#!/usr/bin/env python3
"""
Export dep_act_laws plus aggregated electronic adoption votes for the frontend.

Each law includes a nested ``details`` object and ``voting_sessions``: votes on
the same calendar day are grouped into one session, with per-party counts of
distinct deputies who cast a YES on a final adoption-style ballot.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from init_db import DEFAULT_DB_PATH, init_db

DEFAULT_OUTPUT_FILE = Path("outputs/laws_votes/laws_votes.json")

# Final chamber ballots in favor of adoption (see crawl_deputy_activity._classify_law_vote_type).
ADOPTION_YES_VOTE_TYPES = ("final_adoption", "final_vote")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _maybe_json(raw: str | None) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _law_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(dep_act_laws)").fetchall()
    return {str(r[1]) for r in rows}


def _pick(available: set[str], *names: str) -> list[str]:
    return [n for n in names if n in available]


def export_laws_votes_json(
    *,
    db_path: Path,
    output_path: Path,
) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cols = _law_columns(conn)
        required = {"law_id", "source_url", "title", "details_text", "columns_json"}
        missing = required - cols
        if missing:
            raise RuntimeError(f"dep_act_laws missing columns: {sorted(missing)}")

        select_parts = _pick(
            cols,
            "law_id",
            "source_url",
            "identifier",
            "adopted_law_identifier",
            "title",
            "law_status",
            "details_text",
            "columns_json",
            "adopted_law_pdf_url",
            "adopted_law_text_json",
            "adopted_law_analysis_json",
            "adopted_law_reader_summary",
        )
        sql = f"SELECT {', '.join(select_parts)} FROM dep_act_laws ORDER BY law_id"
        law_rows = conn.execute(sql).fetchall()
        col_index = {name: i for i, name in enumerate(select_parts)}

        vote_agg: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        placeholders = ",".join("?" * len(ADOPTION_YES_VOTE_TYPES))
        vote_sql = f"""
            SELECT
                v.law_id,
                substr(v.vote_date, 1, 10) AS session_day,
                COALESCE(m.party_id, 'unknown') AS party_id,
                COUNT(DISTINCT v.member_normalized_name) AS yes_members
            FROM dep_act_laws_votes v
            LEFT JOIN members m ON m.normalized_name = v.member_normalized_name
            WHERE v.vote = 'YES'
              AND v.vote_type IN ({placeholders})
            GROUP BY v.law_id, session_day, party_id
        """
        for law_id, day, party_id, cnt in conn.execute(vote_sql, ADOPTION_YES_VOTE_TYPES):
            vote_agg[str(law_id)][str(day)][str(party_id)] = int(cnt)

        def idx(name: str) -> int | None:
            return col_index.get(name)

        laws_out: list[dict[str, Any]] = []
        for row in law_rows:
            lid = str(row[idx("law_id")])
            details_text = row[idx("details_text")]
            columns_raw = row[idx("columns_json")]
            laws_out.append(
                {
                    "law_id": lid,
                    "source_url": row[idx("source_url")],
                    "identifier": row[idx("identifier")] if idx("identifier") is not None else None,
                    "adopted_law_identifier": (
                        row[idx("adopted_law_identifier")]
                        if idx("adopted_law_identifier") is not None
                        else None
                    ),
                    "title": row[idx("title")],
                    "details": {
                        "details_text": details_text,
                        "columns_json": _maybe_json(columns_raw) if columns_raw is not None else None,
                        "adopted_law_pdf_url": (
                            row[idx("adopted_law_pdf_url")]
                            if idx("adopted_law_pdf_url") is not None
                            else None
                        ),
                        "adopted_law_text": (
                            _maybe_json(row[idx("adopted_law_text_json")])
                            if idx("adopted_law_text_json") is not None
                            else None
                        ),
                        "adopted_law_analysis": (
                            _maybe_json(row[idx("adopted_law_analysis_json")])
                            if idx("adopted_law_analysis_json") is not None
                            else None
                        ),
                        "adopted_law_summary": (
                            row[idx("adopted_law_reader_summary")]
                            if idx("adopted_law_reader_summary") is not None
                            else None
                        ),
                    },
                    "status": row[idx("law_status")] if idx("law_status") is not None else None,
                    "voting_sessions": _sessions_for_law(vote_agg.get(lid, {})),
                }
            )

        payload = {
            "generated_at": _utc_now_iso(),
            "laws": laws_out,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return len(laws_out)
    finally:
        conn.close()


def _sessions_for_law(by_day: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for day in sorted(by_day.keys()):
        parties = by_day[day]
        breakdown = [
            {"party_id": pid, "member_count": parties[pid]}
            for pid in sorted(parties.keys(), key=lambda p: (-parties[p], p))
        ]
        sessions.append(
            {
                "vote_date": day,
                "parties": breakdown,
            }
        )
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description="Export laws + adoption vote aggregates to JSON.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_FILE})",
    )
    args = parser.parse_args()
    n = export_laws_votes_json(db_path=Path(args.db_path), output_path=Path(args.output))
    print(f"  Wrote {n} law(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
