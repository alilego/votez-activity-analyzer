from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_local_models import (
    _apply_benchmark_scope_defaults,
    _append_summary_run,
    _ensure_run_exists,
    _find_missing_source_sessions,
    _finish_run,
    _load_input_session_file_map,
    _load_gold_targets,
    _load_summary_runs,
    _prepare_missing_gold_sessions,
    _parse_benchmark_model_spec,
    _reset_llm_outputs,
    _select_gold_speeches,
)
from init_db import init_db


class TestBenchmarkLocalModels(unittest.TestCase):
    def test_parse_benchmark_model_spec_uses_default_provider_for_plain_model(self):
        spec = _parse_benchmark_model_spec("gpt-5.4-mini", "openai")
        self.assertEqual(spec.provider, "openai")
        self.assertEqual(spec.model, "gpt-5.4-mini")

    def test_parse_benchmark_model_spec_allows_provider_prefixed_models(self):
        spec = _parse_benchmark_model_spec("ollama/qwen3:14b", "openai")
        self.assertEqual(spec.provider, "ollama")
        self.assertEqual(spec.model, "qwen3:14b")

    def test_limited_benchmark_scope_applies_cheaper_defaults(self):
        session_limit, only_hard = _apply_benchmark_scope_defaults(
            benchmark_scope="limited",
            session_limit=0,
            only_hard=False,
        )
        self.assertEqual(session_limit, 3)
        self.assertTrue(only_hard)

    def test_append_summary_run_preserves_previous_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"

            payload_one = _append_summary_run(summary_path, "2026-04-10T10:00:00+00:00", [{"model": "qwen3:14b"}])
            summary_path.write_text(json.dumps(payload_one), encoding="utf-8")

            payload_two = _append_summary_run(summary_path, "2026-04-10T11:00:00+00:00", [{"model": "gpt-5.4-mini"}])

            self.assertEqual(payload_two["run_started_at"], "2026-04-10T11:00:00+00:00")
            self.assertEqual(payload_two["results"], [{"model": "gpt-5.4-mini"}])
            self.assertEqual(len(payload_two["runs"]), 2)
            self.assertEqual(payload_two["runs"][0]["results"], [{"model": "qwen3:14b"}])
            self.assertEqual(payload_two["runs"][1]["results"], [{"model": "gpt-5.4-mini"}])

    def test_load_summary_runs_migrates_legacy_results_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(json.dumps({"results": [{"model": "qwen3:14b"}]}), encoding="utf-8")

            runs = _load_summary_runs(summary_path)

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["results"], [{"model": "qwen3:14b"}])
            self.assertIn("run_started_at", runs[0])

    def test_load_input_session_file_map_reads_session_ids_from_input_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            input_dir = tmp_root / "input" / "stenograme"
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "session_a.json").write_text(json.dumps({"session_id": "s1"}), encoding="utf-8")
            (input_dir / "session_b.json").write_text(json.dumps({"session_id": "s2"}), encoding="utf-8")

            with mock.patch("benchmark_local_models.ROOT", tmp_root):
                mapping = _load_input_session_file_map(input_dir)

            self.assertEqual(mapping["s1"], "input/stenograme/session_a.json")
            self.assertEqual(mapping["s2"], "input/stenograme/session_b.json")

    def test_prepare_missing_gold_sessions_uses_input_files_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_db = tmp_root / "state.sqlite"
            benchmark_root = tmp_root / "bench"
            input_dir = tmp_root / "input" / "stenograme"
            input_dir.mkdir(parents=True, exist_ok=True)
            init_db(source_db)
            _ensure_run_exists(source_db, "seed_run")

            with sqlite3.connect(source_db) as conn:
                conn.execute(
                    """
                    INSERT INTO session_chunks (
                        chunk_id, run_id, session_id, stenogram_path,
                        chunk_type, chunk_index, text, tokens_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("chunk-1", "seed_run", "s1", "input/stenograme/session_1.json", "speech", 1, "Text", "[]"),
                )
                conn.execute(
                    """
                    INSERT INTO interventions_raw (
                        intervention_id, run_id, session_id, stenogram_path,
                        speech_index, raw_speaker, normalized_speaker, text, text_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("iv-1", "seed_run", "s1", "input/stenograme/session_1.json", 1, "Speaker", "speaker", "Text", "hash"),
                )
                conn.commit()

            (input_dir / "session_2.json").write_text(json.dumps({"session_id": "s2"}), encoding="utf-8")

            with mock.patch("benchmark_local_models.ROOT", tmp_root), mock.patch(
                "benchmark_local_models._run_command"
            ) as run_command:
                prepared_db, prepared_session_ids, still_missing = _prepare_missing_gold_sessions(
                    source_db=source_db,
                    benchmark_root=benchmark_root,
                    input_dir=input_dir,
                    session_ids=["s1", "s2"],
                    run_started_at="2026-04-10T10:00:00+00:00",
                )

            self.assertEqual(prepared_session_ids, ["s2"])
            self.assertEqual(still_missing, [])
            self.assertEqual(prepared_db, benchmark_root / "_prepared_source" / "state.sqlite")
            run_command.assert_called_once()
            called_cmd = run_command.call_args.args[0]
            self.assertIn("scripts/analyze_interventions.py", called_cmd)

    def test_load_gold_targets_groups_by_session_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_path = Path(tmpdir) / "gold.json"
            gold_path.write_text(
                """{
                  "speeches": [
                    {"session_id": 2, "speech_index": 9},
                    {"session_id": 1, "speech_index": 5},
                    {"session_id": 1, "speech_index": 3},
                    {"session_id": 2, "speech_index": 9}
                  ]
                }""",
                encoding="utf-8",
            )
            self.assertEqual(_load_gold_targets(gold_path), {"1": [3, 5], "2": [9]})

    def test_benchmark_run_row_is_created_and_finished(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.sqlite"
            init_db(db_path)

            _ensure_run_exists(db_path, "benchmark_qwen3_14b")
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT run_id, status, finished_at FROM runs WHERE run_id = ?",
                    ("benchmark_qwen3_14b",),
                ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "benchmark_qwen3_14b")
            self.assertEqual(row[1], "running")
            self.assertIsNone(row[2])

            _finish_run(db_path, "benchmark_qwen3_14b", "completed", 18)
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT status, sessions_processed, finished_at FROM runs WHERE run_id = ?",
                    ("benchmark_qwen3_14b",),
                ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "completed")
            self.assertEqual(row[1], 18)
            self.assertIsNotNone(row[2])

    def test_select_gold_speeches_can_filter_to_medium_and_hard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_path = Path(tmpdir) / "gold.json"
            gold_path.write_text(
                """{
                  "speeches": [
                    {"session_id": 1, "speech_index": 1, "difficulty": "easy"},
                    {"session_id": 1, "speech_index": 2, "difficulty": "medium"},
                    {"session_id": 2, "speech_index": 3, "difficulty": "hard"}
                  ]
                }""",
                encoding="utf-8",
            )
            selected = _select_gold_speeches(gold_path, only_hard=True)
            self.assertEqual(
                [(str(item["session_id"]), int(item["speech_index"])) for item in selected],
                [("1", 2), ("2", 3)],
            )

    def test_reset_llm_outputs_can_preserve_existing_session_topics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.sqlite"
            init_db(db_path)
            _ensure_run_exists(db_path, "seed_run")

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO interventions_raw (
                        intervention_id, run_id, session_id, stenogram_path,
                        speech_index, raw_speaker, normalized_speaker, text, text_hash, member_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("iv-1", "seed_run", "s1", "session-1.txt", 1, "Speaker", "speaker", "Text", "hash", "m1"),
                )
                conn.execute(
                    """
                    INSERT INTO intervention_analysis (
                        intervention_id, run_id, relevance_label, confidence,
                        topics_json, relevance_source, evidence_chunk_ids_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("iv-1", "seed_run", "neutral", 0.9, "[]", "llm_agent_v1", "[]"),
                )
                conn.execute(
                    """
                    INSERT INTO session_topics (
                        session_id, run_id, stenogram_path, topics_json, topics_source
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("s1", "seed_run", "session-1.txt", "[]", "llm_v1:qwen3:14b"),
                )
                conn.commit()

            stats = _reset_llm_outputs(db_path, ["s1"], reset_session_topics=False)
            self.assertEqual(stats["deleted_intervention_analysis"], 1)
            self.assertEqual(stats["reset_session_topics"], 0)

            with sqlite3.connect(db_path) as conn:
                analysis_count = conn.execute("SELECT COUNT(*) FROM intervention_analysis").fetchone()[0]
                topics_source = conn.execute(
                    "SELECT topics_source FROM session_topics WHERE session_id = ?",
                    ("s1",),
                ).fetchone()[0]
            self.assertEqual(analysis_count, 0)
            self.assertEqual(topics_source, "llm_v1:qwen3:14b")

    def test_find_missing_source_sessions_reports_absent_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.sqlite"
            init_db(db_path)
            _ensure_run_exists(db_path, "seed_run")

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO session_chunks (
                        chunk_id, run_id, session_id, stenogram_path,
                        chunk_type, chunk_index, text, tokens_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("chunk-1", "seed_run", "s1", "session-1.txt", "speech", 1, "Text", "[]"),
                )
                conn.execute(
                    """
                    INSERT INTO session_chunks (
                        chunk_id, run_id, session_id, stenogram_path,
                        chunk_type, chunk_index, text, tokens_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("chunk-2", "seed_run", "s2", "session-2.txt", "speech", 1, "Text", "[]"),
                )
                conn.execute(
                    """
                    INSERT INTO interventions_raw (
                        intervention_id, run_id, session_id, stenogram_path,
                        speech_index, raw_speaker, normalized_speaker, text, text_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("iv-1", "seed_run", "s1", "session-1.txt", 1, "Speaker", "speaker", "Text", "hash"),
                )
                conn.commit()

            missing = _find_missing_source_sessions(db_path, ["s1", "s2", "s3"])
            self.assertEqual(missing["session_chunks"], ["s3"])
            self.assertEqual(missing["interventions_raw"], ["s2", "s3"])


if __name__ == "__main__":
    unittest.main()
