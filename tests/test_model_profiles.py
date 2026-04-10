from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_profiles import (
    DEFAULT_MODEL_OLLAMA,
    DEFAULT_PIPELINE_ARCHITECTURE_SELECTION,
    get_model_runtime_config,
    get_model_profile,
    infer_model_size_billions,
    infer_ollama_num_ctx,
    infer_preferred_pipeline_architecture,
    model_supports_large_session_single_pass,
    normalize_model_name,
    resolve_pipeline_architecture,
)


class TestModelProfiles(unittest.TestCase):
    def test_default_ollama_model_upgraded_to_14b(self):
        self.assertEqual(DEFAULT_MODEL_OLLAMA, "qwen3:14b")

    def test_pipeline_selection_defaults_to_auto(self):
        self.assertEqual(DEFAULT_PIPELINE_ARCHITECTURE_SELECTION, "auto")

    def test_known_profile_exposes_expected_vram(self):
        profile = get_model_profile("ollama", "qwen3:14b")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.num_ctx, 32768)
        self.assertEqual(profile.estimated_vram_gb, 10)
        self.assertEqual(profile.chunk_chars, 600)

    def test_infer_ollama_num_ctx_from_suffix(self):
        self.assertEqual(infer_ollama_num_ctx("qwen3:14b-32k"), 32768)
        self.assertEqual(infer_ollama_num_ctx("llama3.1:8b-8k"), 8192)

    def test_unknown_model_defaults_to_safe_ctx(self):
        self.assertEqual(infer_ollama_num_ctx("mistral"), 8192)

    def test_large_session_support_depends_on_ctx(self):
        self.assertTrue(model_supports_large_session_single_pass("ollama", "qwen3:14b"))
        self.assertFalse(model_supports_large_session_single_pass("ollama", "llama3.1:8b-8k"))
        self.assertTrue(model_supports_large_session_single_pass("openai", "gpt-4o-mini"))

    def test_infer_model_size_from_name(self):
        self.assertEqual(infer_model_size_billions("gemma3:27b"), 27.0)
        self.assertEqual(infer_model_size_billions("llama3.3:70b"), 70.0)
        self.assertIsNone(infer_model_size_billions("mistral"))

    def test_preferred_architecture_by_model_strength(self):
        self.assertEqual(infer_preferred_pipeline_architecture("ollama", "qwen3:14b"), "three_layer")
        self.assertEqual(infer_preferred_pipeline_architecture("ollama", "gemma3:27b"), "one_pass")
        self.assertEqual(infer_preferred_pipeline_architecture("openai", "gpt-4o-mini"), "one_pass")

    def test_resolve_pipeline_architecture_respects_auto_and_override(self):
        self.assertEqual(resolve_pipeline_architecture("ollama", "qwen3:14b", "auto"), "three_layer")
        self.assertEqual(resolve_pipeline_architecture("ollama", "gemma3:27b", "auto"), "one_pass")
        self.assertEqual(resolve_pipeline_architecture("ollama", "gemma3:27b", "three_layer"), "three_layer")

    def test_runtime_config_exposes_architecture_num_ctx_and_chunk_chars(self):
        local_cfg = get_model_runtime_config("ollama", "qwen3:14b")
        self.assertEqual(local_cfg.architecture, "three_layer")
        self.assertEqual(local_cfg.num_ctx, 32768)
        self.assertEqual(local_cfg.chunk_chars, 600)

        strong_local_cfg = get_model_runtime_config("ollama", "gemma3:27b")
        self.assertEqual(strong_local_cfg.architecture, "one_pass")
        self.assertEqual(strong_local_cfg.chunk_chars, 4000)

        openai_cfg = get_model_runtime_config("openai", "gpt-4o-mini")
        self.assertEqual(openai_cfg.architecture, "one_pass")
        self.assertEqual(openai_cfg.chunk_chars, 600)

    def test_normalize_model_name_strips_latest_tag(self):
        self.assertEqual(normalize_model_name("qwen2.5:14b-32k:latest"), "qwen2.5:14b-32k")


if __name__ == "__main__":
    unittest.main()
