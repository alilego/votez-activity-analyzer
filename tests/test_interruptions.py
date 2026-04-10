from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_agent import _build_intervention_message, _classify_interruption_type, _merge_continuation_text
from intervention_layers.rules import apply_pre_llm_shortcuts
from intervention_layers.prompts import (
    build_layer_a_user_message,
    build_layer_b_user_message,
    build_layer_c_user_message,
)


class TestClassifyInterruptionType(unittest.TestCase):
    """Tests for _classify_interruption_type."""

    def test_time_overrun_finalizati(self):
        self.assertEqual(
            _classify_interruption_type("Vă rog să finalizați!"),
            "time_overrun",
        )

    def test_time_overrun_expirat(self):
        self.assertEqual(
            _classify_interruption_type("Vă rog să concluzionaţi! A expirat timpul."),
            "time_overrun",
        )

    def test_time_overrun_microphone_cut(self):
        self.assertEqual(
            _classify_interruption_type(
                "... cheltuielile... (i se întrerupe microfonul.)"
            ),
            "time_overrun",
        )

    def test_time_overrun_doua_minute(self):
        self.assertEqual(
            _classify_interruption_type(
                "Doamnă, vă rog, cu rugămintea să vă încadrați în cele două minute."
            ),
            "time_overrun",
        )

    def test_procedure_violation_regulament(self):
        self.assertEqual(
            _classify_interruption_type("Așa spune Regulamentul!"),
            "procedure_violation",
        )

    def test_procedure_violation_la_subiect(self):
        self.assertEqual(
            _classify_interruption_type("Vă rog să rămâneți la subiect!"),
            "procedure_violation",
        )

    def test_procedure_violation_retrag_cuvantul(self):
        self.assertEqual(
            _classify_interruption_type("Vă retrag cuvântul!"),
            "procedure_violation",
        )

    def test_routine_multumesc(self):
        self.assertEqual(
            _classify_interruption_type("Mulţumesc."),
            "routine",
        )

    def test_routine_handoff(self):
        self.assertEqual(
            _classify_interruption_type("Domnul deputat Câciu."),
            "routine",
        )

    def test_routine_va_rog(self):
        self.assertEqual(
            _classify_interruption_type("Vă rog."),
            "routine",
        )

    def test_routine_floor_procedura(self):
        self.assertEqual(
            _classify_interruption_type("Pe procedură!"),
            "routine",
        )

    def test_none_for_substantive_speech(self):
        self.assertIsNone(
            _classify_interruption_type(
                "Acest proiect de lege privind reforma în domeniul sănătății "
                "necesită o analiză aprofundată a impactului bugetar. Propunem "
                "amendamente la articolele 12 și 15 care să reflecte nevoile "
                "reale ale cetățenilor. Comisia a analizat în detaliu toate "
                "propunerile și a ajuns la concluzia că este nevoie de modificări."
            )
        )

    def test_none_for_empty(self):
        self.assertIsNone(_classify_interruption_type(""))

    def test_none_for_long_text(self):
        self.assertIsNone(
            _classify_interruption_type("cuvânt " * 100)
        )


class TestMergeContinuation(unittest.TestCase):
    """Tests for _merge_continuation_text with interruption types."""

    def _make_speeches(self, entries):
        """Build a list of speech dicts from (speaker, text) tuples."""
        return [
            {"speech_index": i, "raw_speaker": s, "text": t}
            for i, (s, t) in enumerate(entries)
        ]

    def test_time_overrun_always_merges(self):
        speeches = self._make_speeches([
            ("A", "Prima parte a discursului meu privind bugetul"),
            ("Chair", "Vă rog să finalizați!"),
            ("A", "Și în concluzie, susținem proiectul."),
        ])
        text, indices, int_type = _merge_continuation_text(speeches, 2)
        self.assertEqual(len(indices), 2)
        self.assertIn("Prima parte", text)
        self.assertIn("concluzie", text)
        self.assertEqual(int_type, "time_overrun")

    def test_procedure_violation_does_not_merge(self):
        speeches = self._make_speeches([
            ("A", "Sunteți toți niște hoți și corupți!"),
            ("Chair", "Așa spune Regulamentul! Rămâneți la subiect!"),
            ("A", "Dar cetățenii au dreptul să știe!"),
        ])
        text, indices, int_type = _merge_continuation_text(speeches, 2)
        self.assertEqual(len(indices), 1)
        self.assertNotIn("hoți", text)
        self.assertEqual(int_type, "procedure_violation")

    def test_routine_with_continuation_signal_merges(self):
        speeches = self._make_speeches([
            ("A", "Vreau să subliniez că... (i se întrerupe microfonul.)"),
            ("Chair", "Mulțumesc."),
            ("A", "...acest proiect are probleme grave."),
        ])
        text, indices, int_type = _merge_continuation_text(speeches, 2)
        self.assertEqual(len(indices), 2)
        self.assertIn("subliniez", text)

    def test_routine_without_continuation_signal_no_merge(self):
        speeches = self._make_speeches([
            ("A", "Sunt de acord cu propunerea."),
            ("Chair", "Mulțumesc."),
            ("A", "Următorul punct pe ordinea de zi."),
        ])
        text, indices, int_type = _merge_continuation_text(speeches, 2)
        self.assertEqual(len(indices), 1)

    def test_no_interruption_no_merge(self):
        speeches = self._make_speeches([
            ("A", "Prima parte."),
            ("B", "Sunt de acord."),
            ("A", "A doua parte."),
        ])
        text, indices, int_type = _merge_continuation_text(speeches, 2)
        self.assertEqual(len(indices), 1)
        self.assertIsNone(int_type)

    def test_chain_merge_time_overrun(self):
        speeches = self._make_speeches([
            ("A", "Prima parte a discursului."),
            ("Chair", "Vă rog să finalizați!"),
            ("A", "A doua parte continuă discuția."),
            ("Chair", "Vă rog să finalizați!"),
            ("A", "Și în final, concluzionez."),
        ])
        text, indices, int_type = _merge_continuation_text(speeches, 4)
        self.assertEqual(len(indices), 3)
        self.assertIn("Prima parte", text)
        self.assertIn("final", text)
        self.assertEqual(int_type, "time_overrun")


class TestPreLLMShortcutProtection(unittest.TestCase):
    """Pre-LLM shortcuts must be suppressed for procedure_violation."""

    def test_shortcut_blocked_for_procedure_violation(self):
        result = apply_pre_llm_shortcuts(
            "Mulţumesc.",
            interruption_type="procedure_violation",
        )
        self.assertIsNone(result)

    def test_shortcut_allowed_for_time_overrun(self):
        result = apply_pre_llm_shortcuts(
            "Mulţumesc.",
            interruption_type="time_overrun",
        )
        self.assertIsNotNone(result)

    def test_shortcut_allowed_without_interruption(self):
        result = apply_pre_llm_shortcuts("Mulţumesc.")
        self.assertIsNotNone(result)


class TestInterruptionPromptHint(unittest.TestCase):
    """Interruption context hint is injected into prompts."""

    _session = {"session_date": "2026-01-15"}
    _speech = {"speech_index": 1, "raw_speaker": "Speaker A", "text": "Test speech."}

    def test_layer_a_includes_hint_for_procedure_violation(self):
        msg = build_layer_a_user_message(
            session=self._session,
            session_topics=[],
            target_speech=self._speech,
            interruption_context="procedure_violation",
        )
        self.assertIn("Interruption context", msg)
        self.assertIn("procedural violation", msg)

    def test_layer_a_no_hint_for_time_overrun(self):
        msg = build_layer_a_user_message(
            session=self._session,
            session_topics=[],
            target_speech=self._speech,
            interruption_context="time_overrun",
        )
        self.assertNotIn("Interruption context", msg)

    def test_layer_a_no_hint_without_interruption(self):
        msg = build_layer_a_user_message(
            session=self._session,
            session_topics=[],
            target_speech=self._speech,
        )
        self.assertNotIn("Interruption context", msg)

    def test_layer_b_includes_hint(self):
        msg = build_layer_b_user_message(
            session=self._session,
            session_topics=[],
            target_speech=self._speech,
            layer_a_output={"policy_proposal": "partial"},
            interruption_context="procedure_violation",
        )
        self.assertIn("Interruption context", msg)

    def test_layer_c_includes_hint(self):
        msg = build_layer_c_user_message(
            session=self._session,
            session_topics=[],
            target_speech=self._speech,
            layer_a_output={"policy_proposal": "partial"},
            layer_b_output={"constructiveness_label": "neutral", "confidence": 0.7},
            qa_reasons=["low_confidence"],
            interruption_context="procedure_violation",
        )
        self.assertIn("Interruption context", msg)
        self.assertIn("benefit citizens", msg)


class TestContextRichPromptFormatting(unittest.TestCase):
    _session = {
        "session_date": "2026-01-15",
        "initial_notes": "Ședința este condusă de domnul Ion Popescu. Dezbatere privind finanțe publice locale.",
    }
    _session_topics = [{"label": "Finanțe publice locale"}]
    _agenda = [{"item_number": 1, "title": "Modificarea art. 5 privind finanțele publice locale"}]
    _target = {
        "speech_index": 15,
        "raw_speaker": "Deputat Maria Ionescu",
        "text": "Articolul 5 privind finanțele publice locale trebuie amendat pentru primării.",
    }
    _context = [
        {
            "speech_index": 11,
            "raw_speaker": "Deputat Andrei Vasilescu",
            "text": "Discutăm modificarea articolului 5 și impactul asupra finanțelor publice locale.",
        },
        {
            "speech_index": 12,
            "raw_speaker": "Deputat Maria Ionescu",
            "text": "Primăriile au nevoie de clarificări privind raportul comisiei.",
        },
        {
            "speech_index": 13,
            "raw_speaker": "Domnul Ion Popescu",
            "text": "Trecem la următorul vorbitor.",
        },
        {
            "speech_index": 14,
            "raw_speaker": "Deputat Elena Matei",
            "text": "Raportul comisiei pentru finanțe locale trebuie clarificat înainte de vot.",
        },
    ]

    def test_layer_a_message_adds_active_thread_and_role_hints(self):
        msg = build_layer_a_user_message(
            session=self._session,
            session_topics=self._session_topics,
            target_speech=self._target,
            context_speeches=self._context,
            agenda=self._agenda,
        )
        self.assertIn("Likely active debate thread", msg)
        self.assertIn("Candidate from session topic: Finanțe publice locale", msg)
        self.assertIn("Immediate context (3 previous speeches; highest priority; do NOT classify)", msg)
        self.assertIn("Earlier previous context (1 speeches; lower priority; do NOT classify)", msg)
        self.assertIn("Domnul Ion Popescu (chair, immediate context)", msg)
        self.assertIn("Deputat Elena Matei (immediately before target)", msg)
        self.assertIn("Deputat Maria Ionescu (same speaker, immediate context)", msg)

    def test_one_pass_message_adds_active_thread_and_role_hints(self):
        msg = _build_intervention_message(
            session=self._session,
            session_topics=self._session_topics,
            speeches=[self._target],
            context_speeches=self._context,
            agenda=self._agenda,
        )
        self.assertIn("Likely active debate thread", msg)
        self.assertIn("Context use guidance", msg)
        self.assertIn("Immediate context (3 previous speeches; highest priority; do NOT classify)", msg)
        self.assertIn("Earlier previous context (1 speeches; lower priority; do NOT classify)", msg)
        self.assertIn("Domnul Ion Popescu (chair, immediate context)", msg)
        self.assertIn("Deputat Elena Matei (immediately before target)", msg)


if __name__ == "__main__":
    unittest.main()
