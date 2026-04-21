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

LEGACY_ACTIVITY_TABLE_RENAMES = (
    ("member_activity_crawl", "dep_act_member_activity_crawl"),
    ("laws", "dep_act_laws"),
    ("member_laws", "dep_act_member_laws"),
    ("decision_projects", "dep_act_decision_projects"),
    ("member_decision_projects", "dep_act_member_decision_projects"),
    ("questions_interpellations", "dep_act_questions_interpellations"),
    ("motions", "dep_act_motions"),
    ("member_motions", "dep_act_member_motions"),
    ("political_declarations", "dep_act_political_declarations"),
)

ACTIVITY_JOIN_TABLE_FOREIGN_KEYS = (
    (
        "dep_act_member_laws",
        {"members", "dep_act_laws"},
        """
        CREATE TABLE {table} (
            member_id TEXT NOT NULL,
            law_id TEXT NOT NULL,
            is_initiator INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (law_id) REFERENCES dep_act_laws(law_id),
            PRIMARY KEY (member_id, law_id)
        )
        """,
        ("member_id", "law_id", "is_initiator", "created_at"),
    ),
    (
        "dep_act_member_decision_projects",
        {"members", "dep_act_decision_projects"},
        """
        CREATE TABLE {table} (
            member_id TEXT NOT NULL,
            decision_project_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (decision_project_id) REFERENCES dep_act_decision_projects(decision_project_id),
            PRIMARY KEY (member_id, decision_project_id)
        )
        """,
        ("member_id", "decision_project_id", "created_at"),
    ),
    (
        "dep_act_member_motions",
        {"members", "dep_act_motions"},
        """
        CREATE TABLE {table} (
            member_id TEXT NOT NULL,
            motion_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (motion_id) REFERENCES dep_act_motions(motion_id),
            PRIMARY KEY (member_id, motion_id)
        )
        """,
        ("member_id", "motion_id", "created_at"),
    ),
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _rename_legacy_activity_tables(conn: sqlite3.Connection) -> None:
    for old_name, new_name in LEGACY_ACTIVITY_TABLE_RENAMES:
        if not _table_exists(conn, old_name) or _table_exists(conn, new_name):
            continue
        conn.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")


def _repair_activity_join_foreign_keys(conn: sqlite3.Connection) -> None:
    for table, expected_targets, create_sql, columns in ACTIVITY_JOIN_TABLE_FOREIGN_KEYS:
        if not _table_exists(conn, table):
            continue
        fk_targets = {
            row[2]
            for row in conn.execute(
                f"PRAGMA foreign_key_list({_quote_identifier(table)})"
            )
        }
        if fk_targets == expected_targets:
            continue
        temp_table = f"__rebuild_{table}"
        quoted_temp = _quote_identifier(temp_table)
        quoted_table = _quote_identifier(table)
        existing_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({quoted_table})")
        }
        copy_columns = [column for column in columns if column in existing_columns]
        column_sql = ", ".join(_quote_identifier(column) for column in copy_columns)
        conn.execute(f"DROP TABLE IF EXISTS {quoted_temp}")
        conn.execute(create_sql.format(table=quoted_temp))
        if copy_columns:
            conn.execute(
                f"INSERT INTO {quoted_temp} ({column_sql}) "
                f"SELECT {column_sql} FROM {quoted_table}"
            )
        conn.execute(f"DROP TABLE {quoted_table}")
        conn.execute(f"ALTER TABLE {quoted_temp} RENAME TO {quoted_table}")


def _create_schema(conn: sqlite3.Connection) -> None:
    _rename_legacy_activity_tables(conn)
    _repair_activity_join_foreign_keys(conn)
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
            bills_authored_total INTEGER NOT NULL DEFAULT 0,
            amendments_added_total INTEGER NOT NULL DEFAULT 0,
            profile_url TEXT,
            circumscriptie TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_members_chamber_source
            ON members(chamber, source_member_id);
        CREATE INDEX IF NOT EXISTS idx_members_normalized_name
            ON members(normalized_name);

        CREATE TABLE IF NOT EXISTS dep_act_member_activity_crawl (
            member_id TEXT PRIMARY KEY,
            profile_url TEXT NOT NULL,
            legislative_proposals_url TEXT,
            legislative_proposals_count INTEGER,
            promulgated_laws_count INTEGER,
            decision_projects_url TEXT,
            decision_projects_count INTEGER,
            questions_url TEXT,
            questions_count INTEGER,
            motions_url TEXT,
            motions_count INTEGER,
            political_declarations_url TEXT,
            political_declarations_count INTEGER,
            legislative_proposals_stored INTEGER NOT NULL DEFAULT 0,
            decision_projects_stored INTEGER NOT NULL DEFAULT 0,
            questions_stored INTEGER NOT NULL DEFAULT 0,
            motions_stored INTEGER NOT NULL DEFAULT 0,
            political_declarations_stored INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            crawled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id)
        );

        CREATE TABLE IF NOT EXISTS dep_act_laws (
            law_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL UNIQUE,
            identifier TEXT,
            adopted_law_identifier TEXT,
            motive_pdf_url TEXT,
            initiators_text TEXT,
            initiators_extracted_at TEXT,
            initiators_parse_error TEXT,
            initiators_source TEXT,
            title TEXT NOT NULL,
            details_text TEXT NOT NULL,
            columns_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dep_act_member_laws (
            member_id TEXT NOT NULL,
            law_id TEXT NOT NULL,
            is_initiator INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (law_id) REFERENCES dep_act_laws(law_id),
            PRIMARY KEY (member_id, law_id)
        );

        CREATE TABLE IF NOT EXISTS dep_act_decision_projects (
            decision_project_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL UNIQUE,
            identifier TEXT,
            title TEXT NOT NULL,
            details_text TEXT NOT NULL,
            columns_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dep_act_member_decision_projects (
            member_id TEXT NOT NULL,
            decision_project_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (decision_project_id) REFERENCES dep_act_decision_projects(decision_project_id),
            PRIMARY KEY (member_id, decision_project_id)
        );

        CREATE TABLE IF NOT EXISTS dep_act_questions_interpellations (
            question_id TEXT PRIMARY KEY,
            member_id TEXT NOT NULL,
            source_url TEXT NOT NULL UNIQUE,
            identifier TEXT,
            recipient TEXT,
            text TEXT NOT NULL,
            columns_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id)
        );

        CREATE TABLE IF NOT EXISTS dep_act_motions (
            motion_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            details_text TEXT NOT NULL,
            columns_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dep_act_member_motions (
            member_id TEXT NOT NULL,
            motion_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (motion_id) REFERENCES dep_act_motions(motion_id),
            PRIMARY KEY (member_id, motion_id)
        );

        CREATE TABLE IF NOT EXISTS dep_act_political_declarations (
            political_declaration_id TEXT PRIMARY KEY,
            member_id TEXT NOT NULL,
            source_url TEXT NOT NULL UNIQUE,
            text_url TEXT,
            title TEXT NOT NULL,
            full_text TEXT NOT NULL,
            details_text TEXT NOT NULL,
            columns_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id)
        );

        CREATE INDEX IF NOT EXISTS idx_member_laws_law_id
            ON dep_act_member_laws(law_id);
        CREATE INDEX IF NOT EXISTS idx_member_decision_projects_project_id
            ON dep_act_member_decision_projects(decision_project_id);
        CREATE INDEX IF NOT EXISTS idx_member_motions_motion_id
            ON dep_act_member_motions(motion_id);
        CREATE INDEX IF NOT EXISTS idx_political_declarations_member_id
            ON dep_act_political_declarations(member_id);

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
            relevance_label TEXT NOT NULL, -- constructive | neutral | non_constructive | unknown
            relevance_source TEXT NOT NULL DEFAULT 'constructiveness_baseline_v1',
            topics_json TEXT NOT NULL, -- JSON array of strings
            layer_a_json TEXT, -- JSON object with Layer A rubric signals (3-layer pipeline)
            confidence REAL, -- nullable in scaffolding stage
            evidence_chunk_ids_json TEXT NOT NULL, -- JSON array of strings
            analysis_version TEXT NOT NULL DEFAULT 'scaffold_v1',
            reasoning TEXT, -- one-sentence explanation from the LLM (NULL for baseline)
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
            topics_source TEXT NOT NULL DEFAULT 'keyword_baseline_v1', -- keyword_baseline_v1 | llm_v1
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
    )
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', '14')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )
    # Migrations for existing DBs — safe to run repeatedly.
    for migration in [
        "ALTER TABLE intervention_analysis ADD COLUMN relevance_source TEXT NOT NULL DEFAULT 'constructiveness_baseline_v1'",
        "ALTER TABLE session_topics ADD COLUMN topics_source TEXT NOT NULL DEFAULT 'keyword_baseline_v1'",
        "ALTER TABLE intervention_analysis ADD COLUMN reasoning TEXT",
        "ALTER TABLE intervention_analysis ADD COLUMN layer_a_json TEXT",
        "ALTER TABLE members ADD COLUMN bills_authored_total INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE members ADD COLUMN amendments_added_total INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE dep_act_laws ADD COLUMN adopted_law_identifier TEXT",
        "ALTER TABLE dep_act_laws ADD COLUMN motive_pdf_url TEXT",
        "ALTER TABLE dep_act_laws ADD COLUMN initiators_text TEXT",
        "ALTER TABLE dep_act_laws ADD COLUMN initiators_extracted_at TEXT",
        "ALTER TABLE dep_act_laws ADD COLUMN initiators_parse_error TEXT",
        "ALTER TABLE dep_act_laws ADD COLUMN initiators_source TEXT",
        "ALTER TABLE dep_act_member_laws ADD COLUMN is_initiator INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE dep_act_member_activity_crawl ADD COLUMN political_declarations_url TEXT",
        "ALTER TABLE dep_act_member_activity_crawl ADD COLUMN political_declarations_count INTEGER",
        "ALTER TABLE dep_act_member_activity_crawl ADD COLUMN political_declarations_stored INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE dep_act_questions_interpellations ADD COLUMN member_id TEXT",
        "ALTER TABLE dep_act_questions_interpellations ADD COLUMN identifier TEXT",
        "ALTER TABLE dep_act_questions_interpellations ADD COLUMN recipient TEXT",
    ]:
        try:
            conn.execute(migration)
        except sqlite3.OperationalError:
            pass  # Column already exists.

    try:
        conn.execute(
            """
            UPDATE dep_act_questions_interpellations
            SET member_id = (
                SELECT member_id
                FROM member_questions_interpellations mq
                WHERE mq.question_id = dep_act_questions_interpellations.question_id
                LIMIT 1
            )
            WHERE member_id IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM member_questions_interpellations mq
                  WHERE mq.question_id = dep_act_questions_interpellations.question_id
              )
            """
        )
    except sqlite3.OperationalError:
        pass
    conn.execute("DROP TABLE IF EXISTS member_questions_interpellations")
    try:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_questions_interpellations_member_id
                ON dep_act_questions_interpellations(member_id)
            """
        )
    except sqlite3.OperationalError:
        pass

    # Migrate old 'llm_v1' (no model suffix) to 'llm_v1:llama3.1:8b'.
    # topics_source format changed to 'llm_v1:{model}' to track which model produced each result.
    conn.execute(
        "UPDATE session_topics SET topics_source = 'llm_v1:llama3.1:8b' WHERE topics_source = 'llm_v1'"
    )


def init_db(db_path: Path) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure sibling state directories exist.
    for subdir in ("run_inputs", "run_prompts", "run_outputs", "external_prompts_output", "generated_prompts"):
        (db_path.parent / subdir).mkdir(parents=True, exist_ok=True)
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
