from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from intervention_layers.rules import (
    apply_deterministic_rules,
    apply_pre_llm_shortcuts,
    detect_committee_report,
    detect_session_chair_procedural,
    extract_session_chairs,
)


class TestPreLLMShortcuts(unittest.TestCase):
    """Tests for pre-LLM shortcuts that bypass the entire LLM pipeline."""

    def test_short_greeting_mulțumesc(self):
        result = apply_pre_llm_shortcuts("Mulţumesc.")
        self.assertIsNotNone(result)
        self.assertEqual(result["decision"]["constructiveness_label"], "neutral")
        self.assertIn("greeting/thanks", result["reason"])

    def test_short_greeting_va_rog(self):
        result = apply_pre_llm_shortcuts("Vă rog să finalizaţi!")
        self.assertIsNotNone(result)
        self.assertEqual(result["decision"]["constructiveness_label"], "neutral")

    def test_short_greeting_buna_ziua(self):
        result = apply_pre_llm_shortcuts("Bună ziua, stimaţi colegi.")
        self.assertIsNotNone(result)
        self.assertEqual(result["decision"]["constructiveness_label"], "neutral")

    def test_greeting_too_long_no_shortcut(self):
        """11-word greeting should NOT trigger the ultra-short rule."""
        result = apply_pre_llm_shortcuts(
            "Mulţumesc foarte mult domnule preşedinte pentru această ocazie de a vorbi astăzi."
        )
        self.assertIsNone(result)

    def test_vote_announcement_short(self):
        result = apply_pre_llm_shortcuts(
            "Proiectul de lege rămâne la votul final."
        )
        self.assertIsNotNone(result)
        self.assertIn("vote announcement", result["reason"])

    def test_vote_supun_la_vot(self):
        result = apply_pre_llm_shortcuts(
            "Supun la vot proiectul de lege în ansamblul său."
        )
        self.assertIsNotNone(result)
        self.assertIn("vote announcement", result["reason"])

    def test_vote_cine_este_pentru(self):
        result = apply_pre_llm_shortcuts(
            "Cine este pentru? Dar contra? Abţineri?"
        )
        self.assertIsNotNone(result)
        self.assertIn("vote announcement", result["reason"])

    def test_vote_long_speech_no_shortcut(self):
        """Vote mention in a long substantive speech should not trigger shortcut."""
        long = "Stimaţi colegi, " + " ".join(["cuvânt"] * 60) + " supun la vot."
        result = apply_pre_llm_shortcuts(long)
        self.assertIsNone(result)

    def test_chair_name_call(self):
        result = apply_pre_llm_shortcuts("Domnul Câciu.")
        self.assertIsNotNone(result)
        self.assertIn("name-call", result["reason"])

    def test_chair_name_call_doamna(self):
        result = apply_pre_llm_shortcuts("Doamna Şerban.")
        self.assertIsNotNone(result)
        self.assertIn("name-call", result["reason"])

    def test_floor_response_da(self):
        result = apply_pre_llm_shortcuts("Da.")
        self.assertIsNotNone(result)
        self.assertIn("floor response", result["reason"])

    def test_floor_response_nu(self):
        result = apply_pre_llm_shortcuts("Nu.")
        self.assertIsNotNone(result)

    def test_floor_response_prezent(self):
        result = apply_pre_llm_shortcuts("Prezent.")
        self.assertIsNotNone(result)

    def test_substantive_speech_no_shortcut(self):
        result = apply_pre_llm_shortcuts(
            "Acest proiect de lege privind reforma în domeniul sănătăţii necesită "
            "o analiză aprofundată a impactului bugetar. Propunem amendamente la "
            "articolele 12 şi 15 care să reflecte nevoile reale ale cetăţenilor."
        )
        self.assertIsNone(result)

    def test_empty_string_no_shortcut(self):
        result = apply_pre_llm_shortcuts("")
        self.assertIsNone(result)

    def test_synthetic_layer_a_shape(self):
        result = apply_pre_llm_shortcuts("Mulţumesc.")
        self.assertIn("layer_a", result)
        la = result["layer_a"]
        self.assertEqual(la["procedural_content"], "yes")
        self.assertEqual(la["policy_proposal"], "no")
        self.assertEqual(la["policy_analysis"], "no")
        self.assertEqual(la["partisan_rhetoric"], "no")


class TestCommitteeReportDetection(unittest.TestCase):
    def test_detects_committee_report(self):
        text = (
            "În conformitate cu prevederile art. 95 din Regulamentul Camerei "
            "Deputaţilor, Comisia pentru buget, finanţe şi bănci a fost sesizată "
            "pentru dezbatere în fond cu Proiectul de Lege privind modificarea "
            "Codului fiscal. Raportul comisiei propune adoptarea proiectului de lege. "
            + "Text suplimentar. " * 10
        )
        self.assertTrue(detect_committee_report(text))

    def test_short_mention_not_report(self):
        self.assertFalse(detect_committee_report("Raportul comisiei este scurt."))

    def test_no_markers(self):
        self.assertFalse(
            detect_committee_report("Un discurs fără referiri la comisii. " * 10)
        )

    def test_committee_report_as_constructive_candidate(self):
        layer_a = {
            "policy_proposal": "no",
            "policy_analysis": "partial",
            "public_interest_orientation": "partial",
            "partisan_rhetoric": "no",
            "legislative_engagement": "partial",
            "procedural_content": "yes",
            "argumentation_quality": "weak",
        }
        report_text = (
            "În conformitate cu prevederile art. 95 din Regulamentul Camerei "
            "Deputaților, Comisia pentru sănătate și familie a fost sesizată "
            "pentru dezbatere în fond cu Propunerea legislativă pentru modificarea "
            "Legii nr. 95 din 2006 privind reforma în domeniul sănătății. "
            "Raportul comisiei propune respingerea propunerii legislative. "
            + "Argumentele principale ale comisiei sunt următoarele. " * 5
        )
        r = apply_deterministic_rules(layer_a, speech_text=report_text)
        self.assertIn("constructive", r["candidate_labels"])


class TestSessionChairDetection(unittest.TestCase):
    def test_extract_chairs_from_notes(self):
        notes = (
            "Lucrările şedinţei au fost conduse de domnul Vasile-Daniel Suciu, "
            "vicepreşedintele Camerei Deputaţilor."
        )
        chairs = extract_session_chairs(notes)
        found = any("Suciu" in name for name in chairs)
        self.assertTrue(found, f"Expected 'Suciu' in chairs: {chairs}")

    def test_extract_chairs_empty(self):
        self.assertEqual(extract_session_chairs(""), set())
        self.assertEqual(extract_session_chairs(None), set())

    def test_chair_procedural_detected(self):
        chairs = {"Vasile-Daniel Suciu"}
        self.assertTrue(
            detect_session_chair_procedural(
                "Mulţumesc. Trecem la următorul punct.",
                "Domnul Vasile-Daniel Suciu",
                chairs,
            )
        )

    def test_chair_long_speech_not_procedural(self):
        chairs = {"Vasile-Daniel Suciu"}
        long_text = " ".join(["cuvânt"] * 50)
        self.assertFalse(
            detect_session_chair_procedural(
                long_text,
                "Domnul Vasile-Daniel Suciu",
                chairs,
            )
        )

    def test_non_chair_not_detected(self):
        chairs = {"Vasile-Daniel Suciu"}
        self.assertFalse(
            detect_session_chair_procedural(
                "Mulţumesc. Vă rog să continuăm.",
                "Domnul George Simion",
                chairs,
            )
        )

    def test_chair_procedural_neutral_shortcut(self):
        layer_a = {
            "policy_proposal": "no",
            "policy_analysis": "no",
            "public_interest_orientation": "no",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "partial",
            "argumentation_quality": "none",
        }
        r = apply_deterministic_rules(
            layer_a,
            speech_text="Mulţumesc. Trecem la următorul punct.",
            is_session_chair=True,
        )
        self.assertEqual(r["shortcut_label"], "neutral")
        self.assertIn("session chair", r["shortcut_reason"])


if __name__ == "__main__":
    unittest.main()
