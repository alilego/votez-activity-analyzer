from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_accuracy import evaluate


class TestEvaluateAccuracy(unittest.TestCase):
    def test_evaluate_keeps_full_mismatch_text_and_reasoning_in_report(self):
        full_text = "A" * 180
        full_reasoning = "B" * 200
        gold_speeches = [
            {
                "id": 1,
                "session_id": "s1",
                "speech_index": 1,
                "raw_speaker": "Speaker",
                "expected_label": "constructive",
                "difficulty": "hard",
                "text": full_text,
                "labeling_notes": "notes",
                "expected_topics": [],
                "expected_law_ids": [],
            }
        ]
        predictions = {
            ("s1", 1): {
                "label": "neutral",
                "topics": [],
                "confidence": 0.8,
                "reasoning": full_reasoning,
            }
        }

        report = evaluate(gold_speeches, predictions, session_topics={})

        mismatch = report["mismatches"][0]
        self.assertEqual(mismatch["text_preview"], full_text)
        self.assertEqual(mismatch["pred_reasoning"], full_reasoning)


if __name__ == "__main__":
    unittest.main()
