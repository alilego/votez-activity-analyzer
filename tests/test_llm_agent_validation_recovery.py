import sqlite3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_agent


class RecoveryTests(unittest.TestCase):
    def test_string_result_items_raise_retryable_validation_error(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"results": ["invalid", "items"]})))], usage=None)
        with mock.patch.object(llm_agent, "create_chat_completion", return_value=response), \
                mock.patch.object(llm_agent, "save_prompt"), \
                mock.patch.object(llm_agent, "_record_llm_usage"):
            with self.assertRaisesRegex(ValueError, "must be JSON objects"):
                llm_agent._call_llm(client=SimpleNamespace(_model="gpt-5-mini"), provider="openai",
                    session={}, session_topics=[], user_message_override="Test", system_prompt="Test")

    def test_invalid_overall_label_is_retried_before_storing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            text = "Finanţarea educaţiei trebuie să crească prin investiţii transparente şi control parlamentar asupra cheltuielilor publice."
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE interventions_raw (intervention_id TEXT, session_id TEXT, member_id TEXT, speech_index INTEGER, raw_speaker TEXT, text TEXT)")
                conn.execute("INSERT INTO interventions_raw VALUES (?, ?, ?, ?, ?, ?)",
                             ("pending", "session", "member", 0, "Domnul Ion Popescu", text))
            stored = []

            def tool(name, params):
                if name == "get_session":
                    return {"ok": True, "session": {}}
                if name == "get_session_topics":
                    return {"ok": True, "topics": []}
                if name == "store_intervention_analysis":
                    stored.append(params)
                    return {"ok": True}
                raise AssertionError(name)

            good = {
                "speech_index": 0, "constructiveness_label": "constructive",
                "topics": [], "confidence": .8, "evidence_quote": text,
                "reasoning": "Finanţarea educaţiei necesită investiţii transparente şi control parlamentar.",
            }
            bad = {**good, "constructiveness_label": "partial"}
            with mock.patch.object(llm_agent, "_call_llm", side_effect=[[bad], [good]]) as call, \
                    mock.patch.object(llm_agent, "apply_pre_llm_shortcuts", return_value=None):
                result = llm_agent.classify_session_interventions_one_pass(
                    server=mock.Mock(call=tool), session_id="session", intervention_ids=["pending"],
                    client=mock.Mock(), provider="openai",
                    config={"max_topics_per_intervention": 5, "max_topic_length": 100}, db_path=db,
                )
            self.assertEqual(result, {"classified": 1, "errors": 0, "error_log": []})
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["constructiveness_label"], "constructive")
            self.assertEqual(call.call_count, 2)
            self.assertIn("Invalid label", call.call_args.kwargs["system_prompt"])


if __name__ == "__main__":
    unittest.main()
