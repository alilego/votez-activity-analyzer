from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from init_db import init_db
import run_pipeline


class TestRunPipelineForcedStenogram(unittest.TestCase):
    def test_forced_stenogram_can_bootstrap_fresh_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "state.sqlite"
            candidate_file = tmp / "selected_stenograms.json"
            candidate_file.write_text("{}", encoding="utf-8")
            forced_path = "input/stenograme/stenograma_2025-02-10_1.json"
            init_db(db_path)

            def fake_subprocess_run(cmd, env=None, shell=False):
                if isinstance(cmd, list) and "scripts/analyze_interventions.py" in cmd:
                    with sqlite3.connect(db_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO session_chunks (
                                chunk_id, run_id, session_id, stenogram_path,
                                chunk_type, chunk_index, text, tokens_json
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "chunk-1",
                                "test_run",
                                "8851",
                                forced_path,
                                "speech",
                                1,
                                "Text",
                                "[]",
                            ),
                        )
                        conn.commit()
                return SimpleNamespace(returncode=0)

            argv = [
                "run_pipeline.py",
                "--db-path",
                str(db_path),
                "--run-id",
                "test_run",
                "--analyzer-mode",
                "llm",
                "--llm-provider",
                "openai",
                "--llm-model",
                "gpt-5-nano",
                "--stenogram",
                forced_path,
                "--skip-export",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "run_pipeline._write_candidate_file", return_value=candidate_file
            ), mock.patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
                exit_code = run_pipeline.main()

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
