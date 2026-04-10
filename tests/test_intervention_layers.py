from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from intervention_layers.orchestrator import build_shortcut_decision, merge_for_compatibility
from intervention_layers.qa import evaluate_qa_triggers
from intervention_layers.rules import apply_deterministic_rules
from intervention_layers.schemas import validate_layer_a_item


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

    def test_very_short_intervention_triggers_qa(self):
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "partial",
            "public_interest_orientation": "partial",
            "partisan_rhetoric": "partial",
            "legislative_engagement": "no",
            "procedural_content": "yes",
            "argumentation_quality": "weak",
            "primary_function": "procedural",
        }
        layer_b = {
            "constructiveness_label": "neutral",
            "confidence": 0.8,
            "topics": [],
            "reasoning": "Scurt.",
        }
        reasons = evaluate_qa_triggers(layer_a, layer_b, "Mulțumesc.", [])
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

    def test_low_confidence_threshold_raised_to_070(self):
        """Confidence 0.67 used to be above 0.65 (no trigger), now below 0.70."""
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "partial",
            "public_interest_orientation": "partial",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "no",
            "argumentation_quality": "weak",
            "primary_function": "substantive_support",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.67,
            "topics": ["buget"],
            "reasoning": "Propunere moderată.",
        }
        reasons = evaluate_qa_triggers(
            layer_a, layer_b,
            "Acest proiect de lege necesită o analiză mai aprofundată a impactului bugetar.",
            [],
        )
        self.assertIn("low_confidence", reasons)

    def test_confidence_070_does_not_trigger(self):
        layer_a = {
            "policy_proposal": "yes",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "yes",
            "procedural_content": "no",
            "argumentation_quality": "strong",
            "primary_function": "substantive_support",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.70,
            "topics": ["buget"],
            "reasoning": "Propunere clară.",
        }
        reasons = evaluate_qa_triggers(
            layer_a, layer_b,
            "Proiectul de lege privind bugetul are nevoie de amendamente semnificative la articolul 12.",
            [],
        )
        self.assertNotIn("low_confidence", reasons)

    def test_very_short_speech_suppressed_by_deterministic_candidates(self):
        """When deterministic rules provide candidates, very_short_speech is skipped."""
        layer_a = {
            "policy_proposal": "no",
            "policy_analysis": "no",
            "public_interest_orientation": "no",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "partial",
            "argumentation_quality": "none",
            "primary_function": "procedural",
        }
        layer_b = {
            "constructiveness_label": "neutral",
            "confidence": 0.80,
            "topics": [],
            "reasoning": "Scurt.",
        }
        reasons = evaluate_qa_triggers(
            layer_a, layer_b, "Ordinea de zi.", [],
            deterministic_candidates=["neutral"],
        )
        self.assertNotIn("very_short_speech", reasons)

    def test_very_short_speech_and_condition(self):
        """≤2 sentences but >25 words should NOT trigger (AND instead of OR)."""
        layer_a = {
            "policy_proposal": "partial",
            "policy_analysis": "partial",
            "public_interest_orientation": "partial",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "no",
            "argumentation_quality": "weak",
            "primary_function": "substantive_support",
        }
        layer_b = {
            "constructiveness_label": "constructive",
            "confidence": 0.75,
            "topics": ["educație"],
            "reasoning": "Propunere.",
        }
        long_2_sentences = (
            "Distribuție și furnizare a energiei termice pentru populație în sistem "
            "centralizat se traduce practic printr-un surplus de fonduri atât la "
            "bugetul de stat cât și la bugetele locale. Aceasta este o problemă."
        )
        reasons = evaluate_qa_triggers(layer_a, layer_b, long_2_sentences, [])
        self.assertNotIn("very_short_speech", reasons)

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
            "debate_advancement": "yes",
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
        self.assertEqual(out["debate_advancement"], "yes")
        self.assertIn("topics", out)
        self.assertIn("reasoning", out)
        self.assertIn("evidence_quote", out)

    def test_validate_layer_a_accepts_debate_advancement(self):
        item = {
            "speech_index": 7,
            "policy_proposal": "partial",
            "policy_analysis": "yes",
            "public_interest_orientation": "yes",
            "partisan_rhetoric": "no",
            "legislative_engagement": "partial",
            "procedural_content": "partial",
            "argumentation_quality": "strong",
            "debate_advancement": "yes",
            "primary_function": "substantive_support",
            "reasoning": "Clarifică impactul amendamentului.",
            "evidence_quote": "Articolul 5 schimbă formula de calcul.",
        }
        out = validate_layer_a_item(item)
        self.assertEqual(out["debate_advancement"], "yes")


if __name__ == "__main__":
    unittest.main()
