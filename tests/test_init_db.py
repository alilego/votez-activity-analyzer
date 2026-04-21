from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

from init_db import init_db  # noqa: E402


class TestInitDbSchema(unittest.TestCase):
    def test_init_db_creates_full_activity_law_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.sqlite"
            init_db(db_path)

            conn = sqlite3.connect(db_path)
            try:
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(dep_act_laws)")
                }
                schema_version = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
            finally:
                conn.close()

        self.assertIn("initiators_source", columns)
        self.assertEqual(schema_version[0], "14")


if __name__ == "__main__":
    unittest.main()
