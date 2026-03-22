from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from intervention_layers.orchestrator import build_shortcut_decision, merge_for_compatibility
from intervention_layers.qa import evaluate_qa_triggers
from intervention_layers.rules import apply_deterministic_rules, apply_pre_llm_shortcuts
from law_extractor import (
    extract_law_references,
    build_session_law_index,
    parse_agenda_from_notes,
    format_agenda_for_prompt,
    validate_law_ids,
    SessionLawIndex,
)


class TestInterventionLayers(unittest.TestCase):
    def test_pure_procedural_line_shortcut(self):
        layer_a = {
            "speech_index": 1,
            "policy_proposal": "no",
            "policy_analysis": "no",
            "public_interest_orientation": "no",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "yes",
            "argumentation_quality": "none",
            "primary_function": "procedural",
            "reasoning": "Intervenția este strict procedurală.",
            "evidence_quote": "Vă rog, trecem la următorul punct.",
        }
        r = apply_deterministic_rules(layer_a)
        self.assertEqual(r["shortcut_label"], "neutral")
        d = build_shortcut_decision(layer_a, r)
        self.assertIsNotNone(d)
        self.assertEqual(d["constructiveness_label"], "neutral")

    def test_partisan_attack_no_substance_candidate(self):
        layer_a = {
            "policy_proposal": "no",
            "policy_analysis": "no",
            "public_interest_orientation": "partial",
            "partisan_rhetoric": "yes",
            "legislative_engagement": "no",
            "procedural_content": "no",
            "argumentation_quality": "none",
        }
        r = apply_deterministic_rules(layer_a)
        self.assertIn("non_constructive", r["candidate_labels"])

    def test_evidence_based_criticism_constructive_candidate(self):
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "partial",
            "procedural_content": "no",
            "argumentation_quality": "strong",
        }
        r = apply_deterministic_rules(layer_a)
        self.assertIn("constructive", r["candidate_labels"])

    def test_substantive_opposition_constructive_candidate(self):
        layer_a = {
            "policy_proposal": "yes",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "partial",
            "legislative_engagement": "yes",
            "procedural_content": "no",
            "argumentation_quality": "strong",
        }
        r = apply_deterministic_rules(layer_a)
        self.assertIn("constructive", r["candidate_labels"])

    def test_mixed_speech_triggers_qa(self):
        layer_a = {
            "policy_proposal": "yes",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "yes",
            "legislative_engagement": "partial",
            "procedural_content": "partial",
            "argumentation_quality": "weak",
            "primary_function": "mixed",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.62,
            "topics": ["buget"],
            "reasoning": "Discurs mixt.",
        }
        reasons = evaluate_qa_triggers(layer_a, layer_b, "Text lung cu atacuri și propuneri.", [])
        self.assertIn("primary_function_mixed", reasons)
        self.assertIn("analysis_and_partisan_conflict", reasons)
        self.assertIn("low_confidence", reasons)

    def test_committee_report_constructive_candidate(self):
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "yes",
            "procedural_content": "partial",
            "argumentation_quality": "strong",
        }
        r = apply_deterministic_rules(layer_a)
        self.assertIn("constructive", r["candidate_labels"])

    def test_very_short_procedural_suppresses_qa(self):
        """Phase 2.4: very_short_speech suppressed for clearly procedural shorts."""
        layer_a = {
            "policy_proposal": "no",
            "policy_analysis": "no",
            "public_interest_orientation": "no",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "yes",
            "argumentation_quality": "none",
            "primary_function": "procedural",
        }
        layer_b = {
            "constructiveness_label": "neutral",
            "confidence": 0.8,
            "topics": [],
            "reasoning": "Scurt.",
        }
        reasons = evaluate_qa_triggers(layer_a, layer_b, "Mulțumesc.", [])
        self.assertNotIn("very_short_speech", reasons)

    def test_very_short_substantive_triggers_qa(self):
        """Short speech with substantive content should still trigger QA."""
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "partial",
            "public_interest_orientation": "partial",
            "partisan_rhetoric": "partial",
            "legislative_engagement": "no",
            "procedural_content": "no",
            "argumentation_quality": "weak",
            "primary_function": "mixed",
        }
        layer_b = {
            "constructiveness_label": "neutral",
            "confidence": 0.8,
            "topics": [],
            "reasoning": "Scurt dar substanțial.",
        }
        reasons = evaluate_qa_triggers(layer_a, layer_b, "Propun respingerea.", [])
        self.assertIn("very_short_speech", reasons)

    def test_topic_inference_from_context_trigger(self):
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "partial",
            "procedural_content": "no",
            "argumentation_quality": "weak",
            "primary_function": "substantive_support",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.77,
            "topics": [],
            "reasoning": "Face trimitere la tema discutată.",
        }
        session_topics = [{"label": "salariul minim"}, {"label": "digitalizare"}]
        reasons = evaluate_qa_triggers(
            layer_a,
            layer_b,
            "Această măsură privind salariul minim trebuie implementată gradual.",
            session_topics,
        )
        self.assertIn("missing_topics_despite_topic_reference", reasons)

    def test_backward_compatible_merge_shape(self):
        layer_a = {
            "speech_index": 42,
            "policy_proposal": "yes",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "yes",
            "procedural_content": "no",
            "argumentation_quality": "strong",
        }
        decision = {
            "speech_index": 42,
            "constructiveness_label": "constructive",
            "confidence": 0.88,
            "topics": ["PL-x 45/2025"],
            "reasoning": "Are conținut substanțial.",
            "evidence_quote": "Comisia propune adoptarea cu amendamente.",
        }
        out = merge_for_compatibility(layer_a, decision, qa_action="confirmed")
        self.assertEqual(out["speech_index"], 42)
        self.assertEqual(out["constructiveness_label"], "constructive")
        self.assertIn("policy_proposal", out)
        self.assertIn("policy_analysis", out)
        self.assertIn("public_interest_orientation", out)
        self.assertIn("partisan_rhetoric", out)
        self.assertIn("legislative_engagement", out)
        self.assertIn("procedural_content", out)
        self.assertIn("argumentation_quality", out)
        self.assertIn("topics", out)
        self.assertIn("reasoning", out)
        self.assertIn("evidence_quote", out)


class TestPreLLMShortcuts(unittest.TestCase):
    """Phase 2.3: Pre-LLM deterministic shortcuts."""

    def test_greeting_shortcut(self):
        result = apply_pre_llm_shortcuts("Mulțumesc, domnule președinte.")
        self.assertIsNotNone(result)
        self.assertEqual(result["shortcut_label"], "neutral")
        self.assertGreaterEqual(result["shortcut_confidence"], 0.90)

    def test_ultra_short_da(self):
        result = apply_pre_llm_shortcuts("Da.")
        self.assertIsNotNone(result)
        self.assertEqual(result["shortcut_label"], "neutral")

    def test_ultra_short_nu(self):
        result = apply_pre_llm_shortcuts("Nu.")
        self.assertIsNotNone(result)
        self.assertEqual(result["shortcut_label"], "neutral")

    def test_vote_announcement_shortcut(self):
        result = apply_pre_llm_shortcuts("Supun la vot proiectul de lege. Cine este pentru?")
        self.assertIsNotNone(result)
        self.assertEqual(result["shortcut_label"], "neutral")

    def test_chair_procedural_shortcut(self):
        result = apply_pre_llm_shortcuts("Are cuvântul domnul deputat Popescu.")
        self.assertIsNotNone(result)
        self.assertEqual(result["shortcut_label"], "neutral")

    def test_committee_report_candidate(self):
        text = (
            "Raportul Comisiei pentru buget propune adoptarea proiectului de lege "
            "privind modificarea Legii nr. 107/1996 cu amendamentele discutate "
            "în comisie. Avem aviz favorabil de la Consiliul Legislativ."
        )
        result = apply_pre_llm_shortcuts(text)
        self.assertIsNotNone(result)
        self.assertIsNone(result.get("shortcut_label"))
        self.assertIn("constructive", result.get("candidate_labels", []))

    def test_substantive_speech_no_shortcut(self):
        text = (
            "Propun modificarea articolului 45 din Legea nr. 273/2006 privind "
            "finanțele publice locale, pentru a asigura o distribuție mai echitabilă "
            "a fondurilor către comunitățile rurale."
        )
        result = apply_pre_llm_shortcuts(text)
        self.assertIsNone(result)

    def test_attack_speech_no_shortcut(self):
        result = apply_pre_llm_shortcuts("Hoți! Rușine!")
        # Short attack vocabulary should NOT get neutral shortcut
        self.assertIsNone(result)


class TestLawExtractor(unittest.TestCase):
    """Phase 2.1: Law reference extraction."""

    def test_plx_basic(self):
        refs = extract_law_references("PL-x 211/2011 a fost retrimis.")
        self.assertTrue(any(r.ref_type == "plx" and r.number == "211/2011" for r in refs))

    def test_legea_nr(self):
        refs = extract_law_references("Legea nr. 107/1996 privind apele.")
        self.assertTrue(any(r.ref_type == "lege" and r.number == "107/1996" for r in refs))

    def test_oug(self):
        refs = extract_law_references("OUG nr. 114/2018 a fost contestată.")
        self.assertTrue(any(r.ref_type == "oug" and r.number == "114/2018" for r in refs))

    def test_ordonanta_de_urgenta(self):
        refs = extract_law_references("Ordonanța de urgență nr. 114/2018 trebuie abrogată.")
        self.assertTrue(any(r.number == "114/2018" for r in refs))

    def test_hg(self):
        refs = extract_law_references("HG nr. 905/2017 reglementează situația.")
        self.assertTrue(any(r.ref_type == "hg" and r.number == "905/2017" for r in refs))

    def test_hotararea_guvernului(self):
        refs = extract_law_references("Hotărârea Guvernului nr. 905/2017 este aplicabilă.")
        self.assertTrue(any(r.number == "905/2017" for r in refs))

    def test_directiva_ue(self):
        refs = extract_law_references("Directiva UE 2019/1024 privind datele deschise.")
        self.assertTrue(any(r.ref_type == "directiva" and r.number == "2019/1024" for r in refs))

    def test_regulamentul_ue(self):
        refs = extract_law_references("Regulamentul UE 2016/679 (GDPR) este aplicabil.")
        self.assertTrue(any(r.ref_type == "regulament" and r.number == "2016/679" for r in refs))

    def test_generic_nr_with_context(self):
        refs = extract_law_references("Comisia a examinat proiectul nr. 360/2023.")
        self.assertTrue(any(r.number == "360/2023" for r in refs))

    def test_generic_nr_without_context(self):
        refs = extract_law_references("Am primit nr. 360/2023 de telefoane azi.")
        # No legislative context → should not extract
        self.assertFalse(any(r.number == "360/2023" for r in refs))

    def test_multiple_refs_in_one_text(self):
        text = "PL-x 211/2011 privind Legea nr. 107/1996 și OUG nr. 114/2018."
        refs = extract_law_references(text)
        numbers = {r.number for r in refs}
        self.assertIn("211/2011", numbers)
        self.assertIn("107/1996", numbers)
        self.assertIn("114/2018", numbers)

    def test_deduplication(self):
        text = "Legea nr. 107/1996 este importantă. Mai menționez Legea nr. 107/1996."
        refs = extract_law_references(text)
        canon = [r.canonical_id for r in refs]
        self.assertEqual(len(canon), len(set(c.lower() for c in canon)))

    def test_empty_text(self):
        self.assertEqual(extract_law_references(""), [])
        self.assertEqual(extract_law_references(None), [])


class TestSessionLawIndex(unittest.TestCase):
    """Phase 2.1: Per-session law index."""

    def test_build_index(self):
        speeches = [
            {"text": "Discutăm PL-x 211/2011 privind Legea apelor."},
            {"text": "Aceasta este o intervenție procedurală."},
            {"text": "OUG nr. 114/2018 trebuie modificată, la fel ca Legea nr. 107/1996."},
        ]
        idx = build_session_law_index("8851", "", speeches)
        self.assertTrue(len(idx.all_law_ids) >= 2)
        self.assertIn(0, idx.speech_to_laws)
        self.assertIn(2, idx.speech_to_laws)
        self.assertNotIn(1, idx.speech_to_laws)

    def test_format_for_prompt(self):
        idx = SessionLawIndex(session_id="test")
        idx.add("PL-x 211/2011", 0)
        idx.add("Legea nr. 107/1996", 0)
        idx.add("Legea nr. 107/1996", 5)
        text = idx.format_for_prompt()
        self.assertIn("PL-x 211/2011", text)
        self.assertIn("Legea nr. 107/1996", text)
        self.assertIn("Pre-extracted", text)

    def test_empty_index(self):
        idx = SessionLawIndex(session_id="test")
        self.assertEqual(idx.format_for_prompt(), "")


class TestAgendaParsing(unittest.TestCase):
    """Phase 2.2: Agenda extraction from initial notes."""

    def test_numbered_agenda(self):
        notes = """Ordinea de zi:
1. Proiectul de Lege privind Legea nr. 107/1996
2. Dezbaterea OUG nr. 114/2018
3. Diverse"""
        agenda = parse_agenda_from_notes(notes)
        self.assertTrue(len(agenda) >= 2)
        law_ids_found = []
        for item in agenda:
            law_ids_found.extend(item.law_ids)
        numbers = set()
        for lid in law_ids_found:
            import re
            nums = re.findall(r"\d+/\d{4}", lid)
            numbers.update(nums)
        self.assertIn("107/1996", numbers)
        self.assertIn("114/2018", numbers)

    def test_bulleted_agenda(self):
        notes = """- Raportul Comisiei privind PL-x 45/2025
- Modificarea Legii nr. 273/2006"""
        agenda = parse_agenda_from_notes(notes)
        self.assertTrue(len(agenda) >= 1)

    def test_format_for_prompt(self):
        from law_extractor import AgendaItem
        agenda = [
            AgendaItem(item_number=1, title="Legea apelor", law_ids=["Legea nr. 107/1996"]),
            AgendaItem(item_number=2, title="OUG urgență", law_ids=["OUG nr. 114/2018"]),
        ]
        text = format_agenda_for_prompt(agenda)
        self.assertIn("Session agenda", text)
        self.assertIn("Legea nr. 107/1996", text)

    def test_empty_notes(self):
        self.assertEqual(parse_agenda_from_notes(""), [])
        self.assertEqual(parse_agenda_from_notes(None), [])


class TestValidateLawIds(unittest.TestCase):
    """Phase 2.1: LLM law ID validation."""

    def test_exact_match(self):
        idx = SessionLawIndex(session_id="test")
        idx.add("PL-x 211/2011", 0)
        idx.add("Legea nr. 107/1996", 1)
        result = validate_law_ids(["PL-x 211/2011", "Legea nr. 107/1996"], idx)
        self.assertEqual(len(result), 2)

    def test_partial_number_match(self):
        idx = SessionLawIndex(session_id="test")
        idx.add("Legea nr. 107/1996", 0)
        result = validate_law_ids(["Legea apelor 107/1996"], idx)
        self.assertEqual(len(result), 1)

    def test_reject_hallucinated(self):
        idx = SessionLawIndex(session_id="test")
        idx.add("PL-x 211/2011", 0)
        result = validate_law_ids(["Legea nr. 999/2099"], idx)
        self.assertEqual(len(result), 0)

    def test_empty_index_passes_through(self):
        idx = SessionLawIndex(session_id="test")
        result = validate_law_ids(["PL-x 211/2011"], idx)
        self.assertEqual(len(result), 1)


class TestQATightenedThresholds(unittest.TestCase):
    """Phase 2.4: Tightened QA trigger thresholds."""

    def test_confidence_070_threshold(self):
        """Confidence at 0.68 (was OK before, now triggers low_confidence)."""
        layer_a = {
            "policy_proposal": "yes",
            "policy_analysis": "partial",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "partial",
            "procedural_content": "no",
            "argumentation_quality": "strong",
            "primary_function": "substantive_support",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.68,
            "topics": ["buget"],
        }
        reasons = evaluate_qa_triggers(
            layer_a, layer_b,
            "Propun alocarea de fonduri suplimentare pentru educație rurală.",
            [],
        )
        self.assertIn("low_confidence", reasons)

    def test_confidence_072_no_trigger(self):
        """Confidence at 0.72 should NOT trigger low_confidence with new threshold."""
        layer_a = {
            "policy_proposal": "yes",
            "policy_analysis": "partial",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "partial",
            "procedural_content": "no",
            "argumentation_quality": "strong",
            "primary_function": "substantive_support",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.72,
            "topics": ["buget"],
        }
        reasons = evaluate_qa_triggers(
            layer_a, layer_b,
            "Propun alocarea de fonduri suplimentare pentru educație rurală.",
            [],
        )
        self.assertNotIn("low_confidence", reasons)


if __name__ == "__main__":
    unittest.main()

