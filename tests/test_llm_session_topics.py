from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_session_topics import (
    GPT5_NANO_MAX_OUTPUT_TOKENS,
    TOPICS_MAX_OUTPUT_TOKENS,
    _is_empty_llm_output,
    _topics_max_output_tokens,
    _save_failed_llm_output,
)
from openai_runtime import GPT5_MINI_COMPLETION_TOKEN_BUDGET


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

    def test_gpt5_mini_has_room_for_reasoning_and_visible_json(self):
        client = _FakeClient("openai", "gpt-5-mini")
        self.assertEqual(_topics_max_output_tokens(client), GPT5_MINI_COMPLETION_TOKEN_BUDGET)

    def test_ollama_keeps_its_existing_output_budget(self):
        client = _FakeClient("ollama", "gpt-5-mini")
        self.assertEqual(_topics_max_output_tokens(client), TOPICS_MAX_OUTPUT_TOKENS)

    def test_failed_output_records_response_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("llm_session_topics.RUN_OUTPUTS_DIR", Path(tmp)):
                path = _save_failed_llm_output(
                    run_id="test", session_id="9061", session_date="2026-09-02",
                    model="gpt-5-mini", stage="single_pass_parse_failed_raw",
                    error="Empty response", raw_text="",
                    response_diagnostics={"finish_reason": "length", "reasoning_tokens": 3072},
                )
                content = path.read_text()
                self.assertIn('"finish_reason": "length"', content)
                self.assertIn('"reasoning_tokens": 3072', content)

    def test_empty_output_detection(self):
        self.assertTrue(_is_empty_llm_output(""))
        self.assertTrue(_is_empty_llm_output(" \n\t"))
        self.assertFalse(_is_empty_llm_output('{"matched_topics": []}'))


if __name__ == "__main__":
    unittest.main()
