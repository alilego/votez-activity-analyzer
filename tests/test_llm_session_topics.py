from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_session_topics import (
    GPT5_NANO_MAX_OUTPUT_TOKENS,
    TOPICS_MAX_OUTPUT_TOKENS,
    _is_empty_llm_output,
    _topics_max_output_tokens,
)


class _FakeClient:
    def __init__(self, provider: str, model: str):
        self._provider = provider
        self._model = model


class TestLlmSessionTopics(unittest.TestCase):
    def test_gpt5_nano_uses_full_output_token_limit(self):
        client = _FakeClient("openai", "gpt-5-nano")

        self.assertEqual(_topics_max_output_tokens(client), GPT5_NANO_MAX_OUTPUT_TOKENS)
        self.assertEqual(_topics_max_output_tokens(client), 128_000)

    def test_gpt54_nano_uses_full_output_token_limit(self):
        client = _FakeClient("openai", "gpt-5.4-nano")

        self.assertEqual(_topics_max_output_tokens(client), GPT5_NANO_MAX_OUTPUT_TOKENS)

    def test_non_gpt5_nano_keeps_default_topic_budget(self):
        client = _FakeClient("openai", "gpt-4o-mini")

        self.assertEqual(_topics_max_output_tokens(client), TOPICS_MAX_OUTPUT_TOKENS)

    def test_empty_output_detection(self):
        self.assertTrue(_is_empty_llm_output(""))
        self.assertTrue(_is_empty_llm_output(" \n\t"))
        self.assertFalse(_is_empty_llm_output('{"matched_topics": []}'))


if __name__ == "__main__":
    unittest.main()
