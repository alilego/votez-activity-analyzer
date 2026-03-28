from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_agent import _load_explicit_intervention_ids


class TestLlmAgentCliHelpers(unittest.TestCase):
    def test_load_explicit_intervention_ids_from_json_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ids.json"
            path.write_text(json.dumps(["iv:1", "iv:2", "iv:1"]), encoding="utf-8")
            self.assertEqual(_load_explicit_intervention_ids(path), ["iv:1", "iv:2"])

    def test_load_explicit_intervention_ids_from_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ids.txt"
            path.write_text("iv:3\n\niv:4\niv:3\n", encoding="utf-8")
            self.assertEqual(_load_explicit_intervention_ids(path), ["iv:3", "iv:4"])


if __name__ == "__main__":
    unittest.main()
