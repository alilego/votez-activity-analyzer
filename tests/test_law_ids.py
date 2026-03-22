from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from law_ids import extract_law_id_index_from_speeches, keep_only_allowed_law_id, allowed_law_ids


class TestLawIds(unittest.TestCase):
    def test_extract_common_patterns(self):
        speeches = [
            {"speech_index": 1, "text": "Raport la PL-x 211/2011 si OUG nr. 114/2018."},
            {"speech_index": 2, "text": "Se discuta Legea nr. 107/1996 si HG nr. 12/2020."},
        ]
        idx = extract_law_id_index_from_speeches(speeches)
        self.assertIn("PL-x 211/2011", idx)
        self.assertIn("OUG nr. 114/2018", idx)
        self.assertIn("Legea nr. 107/1996", idx)
        self.assertIn("HG nr. 12/2020", idx)
        self.assertEqual(idx["PL-x 211/2011"], [1])

    def test_generic_nr_only_with_context(self):
        speeches = [
            {"speech_index": 1, "text": "nr. 360/2023 este mentionat fara context."},
            {"speech_index": 2, "text": "Proiectul de lege nr. 360/2023 este important."},
        ]
        idx = extract_law_id_index_from_speeches(speeches)
        self.assertNotIn("nr. 360/2023", {k for k, v in idx.items() if 1 in v})
        self.assertIn("nr. 360/2023", idx)
        self.assertIn(2, idx["nr. 360/2023"])

    def test_keep_only_allowed(self):
        index = {"PL-x 45/2025": [3]}
        allowed = allowed_law_ids(index)
        self.assertEqual(keep_only_allowed_law_id("PL-x 45/2025", allowed), "PL-x 45/2025")
        self.assertIsNone(keep_only_allowed_law_id("PL-x 999/2025", allowed))


if __name__ == "__main__":
    unittest.main()

