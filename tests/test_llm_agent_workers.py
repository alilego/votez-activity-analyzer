import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_agent


class OwnedServer:
    def __init__(self, **kwargs):
        self.owner = threading.get_ident()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        assert self.owner == threading.get_ident()

    def call(self, name, params):
        assert self.owner == threading.get_ident()
        return {"ok": True, "config": {}}


class WorkerTests(unittest.TestCase):
    def test_one_failed_session_does_not_cancel_later_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            with sqlite3.connect(db) as conn:
                conn.executescript('''
                    CREATE TABLE interventions_raw (intervention_id TEXT, session_id TEXT, session_date TEXT, speech_index INTEGER);
                    INSERT INTO interventions_raw VALUES ('a','s1','2026-01-01',0),('b','s2','2026-01-01',0),('c','s3','2026-01-01',0);
                ''')
            seen = []
            def classify(**kwargs):
                seen.append(kwargs['session_id'])
                if kwargs['session_id'] == 's1':
                    raise ValueError('Malformed model item')
                return {'classified': 1, 'errors': 0, 'error_log': []}
            with mock.patch.object(llm_agent, 'MCPServer', OwnedServer), \
                    mock.patch.object(llm_agent, '_build_client', return_value=SimpleNamespace(close=lambda: None)), \
                    mock.patch.object(llm_agent, 'classify_session_interventions', side_effect=classify):
                result = llm_agent.run_agent(db, 'resume', ['a','b','c'], 'gpt-5-mini', provider='openai', workers=2)
            self.assertCountEqual(seen, ['s1','s2','s3'])
            self.assertEqual(result['classified'], 2)
            self.assertEqual(result['errors'], 1)
            self.assertEqual(result['error_log'][0]['session_id'], 's1')

    def test_workers_own_connections_and_process_disjoint_session_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            with sqlite3.connect(db) as conn:
                conn.executescript("""
                    CREATE TABLE interventions_raw (intervention_id TEXT, session_id TEXT, session_date TEXT, speech_index INTEGER);
                    INSERT INTO interventions_raw VALUES ('a', 's1', '2026-01-01', 0), ('b', 's2', '2026-01-01', 0), ('untargeted', 's1', '2026-01-01', 1);
                """)
            barrier = threading.Barrier(2)
            seen = []
            clients = []

            def client_factory(*args):
                client = SimpleNamespace(close=mock.Mock())
                clients.append(client)
                return client

            def classify(**kwargs):
                self.assertEqual(kwargs["server"].owner, threading.get_ident())
                barrier.wait(timeout=5)
                seen.append((kwargs["session_id"], kwargs["intervention_ids"]))
                llm_agent._record_llm_usage(
                    SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)),
                    "system", "user", "output",
                )
                return {"classified": 1, "errors": 0, "error_log": []}

            with mock.patch.object(llm_agent, "MCPServer", OwnedServer), \
                    mock.patch.object(llm_agent, "_build_client", side_effect=client_factory), \
                    mock.patch.object(llm_agent, "classify_session_interventions", side_effect=classify):
                result = llm_agent.run_agent(db, "resume", ["a", "b"], "gpt-5-mini",
                                             provider="openai", workers=2)
            self.assertCountEqual(seen, [("s1", ["a"]), ("s2", ["b"])])
            self.assertEqual(result["classified"], 2)
            self.assertEqual(result["errors"], 0)
            self.assertEqual(result["usage"]["calls"], 2)
            self.assertEqual(result["usage"]["completion_tokens"], 10)
            self.assertEqual(len(clients), 2)
            for client in clients:
                client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
