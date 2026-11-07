"""A resumed run fills pending LLM work without selecting completed analyses."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llm_agent import _load_intervention_ids
from llm_session_topics import _load_session_ids


class IncrementalResumeTests(unittest.TestCase):
    def test_topic_resume_preserves_completed_analyses_from_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            selection = Path(tmp) / "selection.json"
            selection.write_text(json.dumps({"files": []}))
            with sqlite3.connect(db) as conn:
                conn.executescript("""
                    CREATE TABLE session_chunks (session_id TEXT, stenogram_path TEXT);
                    CREATE TABLE session_topics (session_id TEXT, topics_source TEXT);
                    INSERT INTO session_chunks VALUES ('old', 'old.json'), ('done', 'done.json'), ('pending', 'pending.json');
                    INSERT INTO session_topics VALUES ('old', 'llm_v1:gpt-5-nano'), ('done', 'llm_v1:gpt-5-mini'), ('pending', 'keyword_baseline_v1');
                """)
            self.assertEqual(_load_session_ids(db, selection, "gpt-5-mini"), ["pending"])

    def test_intervention_resume_preserves_completed_results_and_excludes_unmatched_speakers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            selection = Path(tmp) / "selection.json"
            selection.write_text(json.dumps({"files": []}))
            with sqlite3.connect(db) as conn:
                conn.executescript("""
                    CREATE TABLE interventions_raw (
                        intervention_id TEXT, member_id TEXT, session_id TEXT,
                        session_date TEXT, speech_index INTEGER, stenogram_path TEXT
                    );
                    CREATE TABLE intervention_analysis (intervention_id TEXT, relevance_source TEXT);
                    INSERT INTO interventions_raw VALUES
                        ('done', 'member', 'session', '2026-01-01', 0, 'session.json'),
                        ('pending', 'member', 'session', '2026-01-01', 1, 'session.json'),
                        ('unmatched', NULL, 'session', '2026-01-01', 2, 'session.json');
                    INSERT INTO intervention_analysis VALUES ('done', 'llm_agent_v1'), ('pending', 'keyword_baseline_v1');
                """)
            self.assertEqual(_load_intervention_ids(db, "resume", selection), ["pending"])


if __name__ == "__main__":
    unittest.main()
