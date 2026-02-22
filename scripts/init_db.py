#!/usr/bin/env python3
"""
Initialize local SQLite state for votez-activity-analyzer.

This script is safe to run multiple times. It creates the database and tables
if they do not exist yet.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("state/state.sqlite")


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL, -- running | completed | failed
            sessions_processed INTEGER NOT NULL DEFAULT 0,
            interventions_total INTEGER NOT NULL DEFAULT 0,
            interventions_classified INTEGER NOT NULL DEFAULT 0,
            unmatched_speakers INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS processed_stenograms (
            stenogram_path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            last_processed_run_id TEXT NOT NULL,
            last_processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (last_processed_run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS run_outputs (
            run_id TEXT PRIMARY KEY,
            sessions_count INTEGER NOT NULL,
            speeches_total INTEGER NOT NULL,
            summary_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS members (
            member_id TEXT PRIMARY KEY,
            source_member_id TEXT NOT NULL,
            chamber TEXT NOT NULL, -- deputat | senator
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            party_id TEXT,
            profile_url TEXT,
            circumscriptie TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_members_chamber_source
            ON members(chamber, source_member_id);
        CREATE INDEX IF NOT EXISTS idx_members_normalized_name
            ON members(normalized_name);

        CREATE TABLE IF NOT EXISTS interventions_raw (
            intervention_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            session_date TEXT,
            stenogram_path TEXT NOT NULL,
            speech_index INTEGER NOT NULL,
            raw_speaker TEXT NOT NULL,
            normalized_speaker TEXT NOT NULL,
            member_id TEXT,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            UNIQUE(stenogram_path, speech_index)
        );

        CREATE INDEX IF NOT EXISTS idx_interventions_raw_run_id
            ON interventions_raw(run_id);
        CREATE INDEX IF NOT EXISTS idx_interventions_raw_session_id
            ON interventions_raw(session_id);
        CREATE INDEX IF NOT EXISTS idx_interventions_raw_member_id
            ON interventions_raw(member_id);

        CREATE TABLE IF NOT EXISTS intervention_analysis (
            intervention_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            relevance_label TEXT NOT NULL, -- relevant | neutral | non_relevant | unknown
            relevance_source TEXT NOT NULL DEFAULT 'session_topics_primary',
            topics_json TEXT NOT NULL, -- JSON array of strings
            confidence REAL, -- nullable in scaffolding stage
            evidence_chunk_ids_json TEXT NOT NULL, -- JSON array of strings
            analysis_version TEXT NOT NULL DEFAULT 'scaffold_v1',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (intervention_id) REFERENCES interventions_raw(intervention_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_intervention_analysis_run_id
            ON intervention_analysis(run_id);
        CREATE INDEX IF NOT EXISTS idx_intervention_analysis_label
            ON intervention_analysis(relevance_label);

        CREATE TABLE IF NOT EXISTS session_topics (
            session_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            stenogram_path TEXT NOT NULL,
            topics_json TEXT NOT NULL, -- JSON array of session-level topics
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_session_topics_run_id
            ON session_topics(run_id);

        CREATE TABLE IF NOT EXISTS session_chunks (
            chunk_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            stenogram_path TEXT NOT NULL,
            chunk_type TEXT NOT NULL, -- session_notes | speech
            chunk_index INTEGER NOT NULL,
            source_speech_index INTEGER,
            text TEXT NOT NULL,
            tokens_json TEXT NOT NULL, -- normalized token list for deterministic retrieval
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_session_chunks_session_id
            ON session_chunks(session_id);
        CREATE INDEX IF NOT EXISTS idx_session_chunks_run_id
            ON session_chunks(run_id);

        CREATE TABLE IF NOT EXISTS unmatched_speakers (
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            stenogram_path TEXT NOT NULL,
            raw_speaker TEXT NOT NULL,
            normalized_speaker TEXT NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            PRIMARY KEY (run_id, session_id, normalized_speaker)
        );

        CREATE INDEX IF NOT EXISTS idx_processed_stenograms_hash
            ON processed_stenograms(content_hash);

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
            m.name AS member_name,
            m.party_id,
            iv.text,
            iv.text_hash,
            ia.relevance_label,
            ia.relevance_source,
            ia.topics_json,
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
    )
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', '6')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )
    try:
        conn.execute(
            "ALTER TABLE intervention_analysis ADD COLUMN relevance_source TEXT NOT NULL DEFAULT 'session_topics_primary'"
        )
    except sqlite3.OperationalError:
        # Column already exists.
        pass


def init_db(db_path: Path) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _create_schema(conn)
        conn.commit()
    return db_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize SQLite state database.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    created_path = init_db(db_path)
    print(f"SQLite state initialized at: {created_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
