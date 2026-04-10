from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openai_runtime import create_chat_completion, resolve_openai_service_tier


class _FakeCompletions:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _FakeChat:
    def __init__(self, side_effects):
        self.completions = _FakeCompletions(side_effects)


class _FakeClient:
    def __init__(self, provider: str, side_effects, service_tier: str | None = None):
        self._provider = provider
        self._openai_service_tier = service_tier
        self.chat = _FakeChat(side_effects)


class _FakeError(Exception):
    def __init__(self, message: str, status_code: int, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


class TestOpenaiRuntime(unittest.TestCase):
    def test_resolve_service_tier_defaults_to_flex_for_openai(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(resolve_openai_service_tier("openai"), "flex")

    def test_resolve_service_tier_uses_env_override(self):
        with mock.patch.dict("os.environ", {"OPENAI_SERVICE_TIER": "auto"}, clear=False):
            self.assertEqual(resolve_openai_service_tier("openai"), "auto")

    def test_non_openai_provider_has_no_service_tier(self):
        self.assertIsNone(resolve_openai_service_tier("ollama"))

    def test_create_chat_completion_adds_flex_for_openai(self):
        client = _FakeClient("openai", side_effects=[{"ok": True}], service_tier="flex")

        result = create_chat_completion(client, model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.chat.completions.calls[0]["service_tier"], "flex")

    def test_create_chat_completion_uses_max_completion_tokens_for_gpt5_models(self):
        client = _FakeClient("openai", side_effects=[{"ok": True}], service_tier="flex")

        result = create_chat_completion(
            client,
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=512,
        )

        self.assertEqual(result, {"ok": True})
        self.assertNotIn("max_tokens", client.chat.completions.calls[0])
        self.assertEqual(client.chat.completions.calls[0]["max_completion_tokens"], 512)

    def test_create_chat_completion_retries_resource_unavailable_with_auto(self):
        err = _FakeError(
            "429 Resource Unavailable",
            status_code=429,
            body={"error": {"message": "Resource unavailable for service_tier=flex", "code": "resource_unavailable"}},
        )
        client = _FakeClient("openai", side_effects=[err, {"ok": True}], service_tier="flex")

        result = create_chat_completion(client, model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.chat.completions.calls[0]["service_tier"], "flex")
        self.assertEqual(client.chat.completions.calls[1]["service_tier"], "auto")

    def test_create_chat_completion_retries_unsupported_flex_with_auto(self):
        err = _FakeError(
            "Unsupported service_tier",
            status_code=400,
            body={"error": {"message": "service_tier flex is not supported for this model", "code": "unsupported_value"}},
        )
        client = _FakeClient("openai", side_effects=[err, {"ok": True}], service_tier="flex")

        result = create_chat_completion(client, model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.chat.completions.calls[1]["service_tier"], "auto")

    def test_create_chat_completion_retries_with_max_completion_tokens_when_requested(self):
        err = _FakeError(
            "Unsupported parameter: max_tokens",
            status_code=400,
            body={
                "error": {
                    "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                    "code": "unsupported_parameter",
                }
            },
        )
        client = _FakeClient("openai", side_effects=[err, {"ok": True}], service_tier="flex")

        result = create_chat_completion(
            client,
            model="custom-openai-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.chat.completions.calls[0]["max_tokens"], 256)
        self.assertNotIn("max_tokens", client.chat.completions.calls[1])
        self.assertEqual(client.chat.completions.calls[1]["max_completion_tokens"], 256)

    def test_create_chat_completion_does_not_inject_service_tier_for_ollama(self):
        client = _FakeClient("ollama", side_effects=[{"ok": True}])

        create_chat_completion(client, model="qwen3:14b", messages=[{"role": "user", "content": "hi"}])

        self.assertNotIn("service_tier", client.chat.completions.calls[0])


if __name__ == "__main__":
    unittest.main()
