#!/usr/bin/env python3
"""
Update only the interventions_enriched view in the state DB.
Does not create or modify tables. Use after changing the view definition in init_db.py.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("state/state.sqlite")

VIEW_SQL = """
DROP VIEW IF EXISTS interventions_enriched;
CREATE VIEW interventions_enriched AS
SELECT
    iv.intervention_id,
    iv.run_id,
    iv.session_id,
    iv.session_date,
    iv.stenogram_path,
    iv.speech_index,
    iv.raw_speaker,
    iv.normalized_speaker,
    iv.member_id,
    m.bills_authored_total,
    m.amendments_added_total,
    m.name AS member_name,
    m.party_id,
    iv.text,
    ia.relevance_label,
    ia.relevance_source,
    ia.reasoning,
    ia.topics_json,
    ia.layer_a_json,
    ia.confidence,
    st.topics_json AS session_topics_json,
    ia.evidence_chunk_ids_json,
    ia.analysis_version,
    iv.created_at AS raw_created_at,
    iv.updated_at AS raw_updated_at,
    ia.created_at AS analysis_created_at,
    ia.updated_at AS analysis_updated_at
FROM interventions_raw iv
LEFT JOIN intervention_analysis ia
    ON ia.intervention_id = iv.intervention_id
LEFT JOIN session_topics st
    ON st.session_id = iv.session_id
LEFT JOIN members m
    ON m.member_id = iv.member_id;
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update interventions_enriched view only (no full DB init)."
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(VIEW_SQL)
    print(f"Updated view interventions_enriched in {db_path}")


if __name__ == "__main__":
    main()
