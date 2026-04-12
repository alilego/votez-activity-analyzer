from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_interventions import TOPIC_TAXONOMY, _extract_topics


class TestAnalyzeInterventionsTopics(unittest.TestCase):
    def test_topic_taxonomy_labels_match_catalog_topics(self):
        payload = json.loads((ROOT / "config" / "topic_taxonomy.json").read_text(encoding="utf-8"))
        expected_labels = [item["label"] for item in payload["catalog_topics"]]
        actual_labels = [label for label, _keywords in TOPIC_TAXONOMY]

        self.assertEqual(actual_labels, expected_labels)

    def test_extract_topics_uses_catalog_label(self):
        topics = _extract_topics("Discutam despre PNRR si fonduri europene pentru investitii publice.")

        self.assertIn("Investitii publice si fonduri europene", topics)
        self.assertNotIn("Politica externa si afaceri europene", topics)

    def test_extract_topics_uses_token_equivalent_keywords(self):
        topics = _extract_topics("Prioritatea este finalizarea autostrazilor si a drumurilor rapide.")

        self.assertIn("Infrastructura rutiera", topics)
        self.assertNotIn("Politici privind migratia si azilul", topics)


if __name__ == "__main__":
    unittest.main()
