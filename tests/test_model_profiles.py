from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_profiles import (
    DEFAULT_MODEL_OLLAMA,
    get_model_profile,
    infer_ollama_num_ctx,
    model_supports_large_session_single_pass,
    normalize_model_name,
)


class TestModelProfiles(unittest.TestCase):
    def test_default_ollama_model_upgraded_to_14b(self):
        self.assertEqual(DEFAULT_MODEL_OLLAMA, "qwen3:14b")

    def test_known_profile_exposes_expected_vram(self):
        profile = get_model_profile("ollama", "qwen3:14b")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.num_ctx, 32768)
        self.assertEqual(profile.estimated_vram_gb, 10)

    def test_infer_ollama_num_ctx_from_suffix(self):
        self.assertEqual(infer_ollama_num_ctx("qwen3:14b-32k"), 32768)
        self.assertEqual(infer_ollama_num_ctx("llama3.1:8b-8k"), 8192)

    def test_unknown_model_defaults_to_safe_ctx(self):
        self.assertEqual(infer_ollama_num_ctx("mistral"), 8192)

    def test_large_session_support_depends_on_ctx(self):
        self.assertTrue(model_supports_large_session_single_pass("ollama", "qwen3:14b"))
        self.assertFalse(model_supports_large_session_single_pass("ollama", "llama3.1:8b-8k"))
        self.assertTrue(model_supports_large_session_single_pass("openai", "gpt-4o-mini"))

    def test_normalize_model_name_strips_latest_tag(self):
        self.assertEqual(normalize_model_name("qwen2.5:14b-32k:latest"), "qwen2.5:14b-32k")


if __name__ == "__main__":
    unittest.main()
