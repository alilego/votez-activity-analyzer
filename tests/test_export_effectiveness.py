from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_effectiveness import _add_intervention, _empty_stats, _finalize_stats


class TestExportEffectiveness(unittest.TestCase):
    def test_letter_ratios_with_neutral_share_add_up_to_nearly_100_percent(self):
        stats = _empty_stats()
        _add_intervention(
            stats,
            word_count=1,
            letter_count=1,
            constructive=True,
            non_constructive=False,
        )
        _add_intervention(
            stats,
            word_count=1,
            letter_count=1,
            constructive=False,
            non_constructive=True,
        )
        _add_intervention(
            stats,
            word_count=1,
            letter_count=1,
            constructive=False,
            non_constructive=False,
        )

        finalized = _finalize_stats(stats)
        neutral_ratio = (
            finalized["total_letter_count"]
            - finalized["constructive_letter_count"]
            - finalized["non_constructive_letter_count"]
        ) / finalized["total_letter_count"]
        neutral_pct = round(neutral_ratio * 100, 2)

        self.assertEqual(finalized["letter_productivity_pct"], 33.33)
        self.assertEqual(finalized["counterproductiveness_pct"], 33.33)
        self.assertEqual(neutral_pct, 33.33)
        self.assertAlmostEqual(
            finalized["letter_productivity_pct"]
            + finalized["counterproductiveness_pct"]
            + neutral_pct,
            100.0,
            delta=0.02,
        )


if __name__ == "__main__":
    unittest.main()
