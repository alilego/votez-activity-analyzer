import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_adopted_laws import (  # noqa: E402
    DEFAULT_MODEL_OPENAI_LAW_ANALYSIS,
    DEFAULT_PROVIDER,
    LawInput,
    _iter_laws_to_analyze,
    analyze_law,
    build_reader_summary,
    format_reader_card,
    parse_json_object,
)
from init_db import init_db  # noqa: E402


def _stage_a(symbolic: bool = False):
    return {
        "affected_legal_acts": [],
        "changed_articles": [],
        "new_obligations": [] if symbolic else ["Conducatorii auto trebuie sa respecte noua regula indicata in text."],
        "new_rights": [],
        "penalties_costs_or_sanctions": [],
        "institutions_responsible": ["Parlamentul Romaniei"] if symbolic else ["Politia Romana"],
        "implementation_date": None,
        "target_groups": ["publicul larg"] if symbolic else ["conducatori auto"],
        "budget_implications": None,
        "mostly_symbolic": symbolic,
        "source_fragments": ["se instituie ziua de 29 octombrie"] if symbolic else ["Ordonanta de urgenta nr.195/2002 se completeaza"],
        "missing_information": [],
    }


def _stage_b(impact_type: str, symbolic: bool = False):
    return {
        "plain_language_title": "Zi comemorativa Regina Maria" if symbolic else "Regula noua pentru circulatia rutiera",
        "one_sentence_summary": (
            "Legea instituie o zi comemorativa nationala."
            if symbolic
            else "Legea modifica regulile de circulatie rutiera pentru conducatori auto."
        ),
        "what_changes": ["Se marcheaza anual o zi comemorativa."] if symbolic else ["Se completeaza OUG 195/2002."],
        "who_is_affected": ["institutii publice", "cetateni interesati"] if symbolic else ["soferi", "Politia Romana"],
        "practical_impact": ["Impact practic direct redus."] if symbolic else ["Soferii pot avea obligatii noi in trafic."],
        "impact_type": impact_type,
        "impact_direction": "neutral" if symbolic else "mixed",
        "citizen_relevance_score": 2 if symbolic else 4,
        "public_interest_score": 3,
        "impact_magnitude_score": 1 if symbolic else 3,
        "clarity_score": 4 if symbolic else 3,
        "implementation_feasibility_score": 5 if symbolic else 3,
        "evidence_quality_score": 2,
        "risk_of_narrow_interest_score": 1 if symbolic else 3,
        "symbolic_only": symbolic,
        "requires_budget": None,
        "creates_new_bureaucracy": False,
        "affects_many_citizens": False if symbolic else True,
        "possible_risks_or_criticism": [] if symbolic else ["Textul poate necesita norme de aplicare clare."],
        "missing_information": [],
        "source_fragments": ["se instituie ziua"] if symbolic else ["OUG 195/2002 se completeaza"],
    }


def _critic_payload(law: LawInput, interp: dict, factual: dict):
    return {
        "unsupported_claims": [],
        "missing_caveats": [],
        "final_analysis": {
            "law_id": law.law_id,
            "original_title": law.title,
            "plain_language_title": interp["plain_language_title"],
            "one_sentence_summary": interp["one_sentence_summary"],
            "what_changes": interp["what_changes"],
            "who_is_affected": interp["who_is_affected"],
            "practical_impact": interp["practical_impact"],
            "impact_type": interp["impact_type"],
            "impact_direction": interp["impact_direction"],
            "citizen_relevance_score": interp["citizen_relevance_score"],
            "public_interest_score": interp["public_interest_score"],
            "impact_magnitude_score": interp["impact_magnitude_score"],
            "clarity_score": interp["clarity_score"],
            "implementation_feasibility_score": interp["implementation_feasibility_score"],
            "evidence_quality_score": interp["evidence_quality_score"],
            "risk_of_narrow_interest_score": interp["risk_of_narrow_interest_score"],
            "symbolic_only": interp["symbolic_only"],
            "requires_budget": interp["requires_budget"],
            "creates_new_bureaucracy": interp["creates_new_bureaucracy"],
            "affects_many_citizens": interp["affects_many_citizens"],
            "affected_legal_acts": factual["affected_legal_acts"],
            "institutions_responsible": factual["institutions_responsible"],
            "obligations_created": factual["new_obligations"],
            "rights_created": factual["new_rights"],
            "penalties_or_costs": factual["penalties_costs_or_sanctions"],
            "implementation_date": factual["implementation_date"],
            "possible_risks_or_criticism": interp["possible_risks_or_criticism"],
            "missing_information": interp["missing_information"],
            "confidence_score": 4,
            "source_fragments": interp["source_fragments"],
        },
    }


class TestAnalyzeAdoptedLaws(unittest.TestCase):
    def _run_case(self, law: LawInput, factual: dict, interp: dict):
        prompts = []

        def caller(stage_name: str, system_prompt: str, user_prompt: str) -> str:
            prompts.append((stage_name, user_prompt))
            self.assertIn(law.extracted_text[:25], user_prompt)
            if stage_name == "stage_a_factual_extraction":
                return json.dumps(factual)
            if stage_name == "stage_b_citizen_interpretation":
                return json.dumps(interp)
            return json.dumps(_critic_payload(law, interp, factual))

        analysis = analyze_law(law, caller)
        self.assertEqual(analysis.law_id, law.law_id)
        self.assertGreaterEqual(analysis.confidence_score, 1)
        self.assertIn("Title:", format_reader_card(analysis))
        self.assertEqual(build_reader_summary(analysis)["law_id"], law.law_id)
        self.assertEqual([name for name, _ in prompts], [
            "stage_a_factual_extraction",
            "stage_b_citizen_interpretation",
            "stage_c_validation_critic",
        ])
        return analysis

    def test_symbolic_law_regina_maria(self):
        law = LawInput(
            law_id="law:regina_maria",
            title="Proiect de Lege pentru instituirea zilei de 29 octombrie ca Ziua Reginei Maria",
            status="adoptata",
            source_url="https://example.test/regina",
            extracted_text="Art. 1. Se instituie ziua de 29 octombrie ca Ziua Reginei Maria.",
        )
        analysis = self._run_case(law, _stage_a(symbolic=True), _stage_b("symbolic", symbolic=True))
        self.assertTrue(analysis.symbolic_only)
        self.assertEqual(analysis.impact_type, "symbolic")

    def test_traffic_law_modifying_oug_195_2002(self):
        law = LawInput(
            law_id="law:traffic",
            title="Propunere legislativa pentru completarea OUG nr.195/2002 privind circulatia pe drumurile publice",
            status="adoptata",
            source_url="https://example.test/traffic",
            extracted_text="Art. I. Ordonanta de urgenta nr.195/2002 privind circulatia pe drumurile publice se completeaza.",
        )
        factual = _stage_a(symbolic=False)
        factual["affected_legal_acts"] = ["OUG nr.195/2002 privind circulatia pe drumurile publice"]
        analysis = self._run_case(law, factual, _stage_b("safety", symbolic=False))
        self.assertEqual(analysis.impact_type, "safety")
        self.assertIn("OUG nr.195/2002 privind circulatia pe drumurile publice", analysis.affected_legal_acts)

    def test_unclear_low_information_text(self):
        law = LawInput(
            law_id="law:unclear",
            title="BP249/19.06.2025",
            status="adoptata",
            source_url="https://example.test/unclear",
            extracted_text="Articol unic. Se aproba prezenta lege.",
        )
        factual = _stage_a(symbolic=False)
        factual.update(
            {
                "new_obligations": [],
                "target_groups": [],
                "mostly_symbolic": False,
                "source_fragments": ["Se aproba prezenta lege"],
                "missing_information": ["Textul disponibil nu explica masura concreta."],
            }
        )
        interp = _stage_b("unclear", symbolic=False)
        interp.update(
            {
                "plain_language_title": "Lege cu impact neclar din textul disponibil",
                "one_sentence_summary": "Textul extras nu permite identificarea clara a schimbarii practice.",
                "impact_direction": "unclear",
                "citizen_relevance_score": 1,
                "public_interest_score": 1,
                "impact_magnitude_score": 1,
                "clarity_score": 1,
                "implementation_feasibility_score": 1,
                "evidence_quality_score": 1,
                "risk_of_narrow_interest_score": 3,
                "affects_many_citizens": None,
                "missing_information": ["Lipseste descrierea schimbarii materiale."],
            }
        )
        analysis = self._run_case(law, factual, interp)
        self.assertEqual(analysis.impact_type, "unclear")
        self.assertEqual(analysis.clarity_score, 1)

    def test_invalid_json_retries(self):
        law = LawInput(
            law_id="law:retry",
            title="Lege test",
            status="adoptata",
            source_url="https://example.test/retry",
            extracted_text="Art. 1. Se instituie o regula.",
        )
        factual = _stage_a(symbolic=True)
        interp = _stage_b("symbolic", symbolic=True)
        calls = {"stage_a_factual_extraction": 0}

        def caller(stage_name: str, system_prompt: str, user_prompt: str) -> str:
            if stage_name == "stage_a_factual_extraction":
                calls[stage_name] += 1
                if calls[stage_name] == 1:
                    return "{bad json"
                return json.dumps(factual)
            if stage_name == "stage_b_citizen_interpretation":
                return json.dumps(interp)
            return json.dumps(_critic_payload(law, interp, factual))

        analysis = analyze_law(law, caller, max_retries=2)
        self.assertEqual(analysis.law_id, "law:retry")
        self.assertEqual(calls["stage_a_factual_extraction"], 2)

    def test_parse_json_object_recovers_wrapped_json_and_names_empty_response(self):
        self.assertEqual(parse_json_object("Here is JSON:\n{\"ok\": true}")["ok"], True)
        with self.assertRaisesRegex(ValueError, "empty response"):
            parse_json_object("")

    def test_defaults_to_openai_gpt_5_mini(self):
        self.assertEqual(DEFAULT_PROVIDER, "openai")
        self.assertEqual(DEFAULT_MODEL_OPENAI_LAW_ANALYSIS, "gpt-5-mini")

    def test_iter_laws_skips_existing_analysis_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            init_db(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text,
                        columns_json, adopted_law_identifier,
                        adopted_law_text_json, adopted_law_analysis_json
                    )
                    VALUES
                        ('law:new', 'https://example.test/new', 'L1/2026', 'New law',
                         'New law', '[]', 'Lege 1/2026',
                         '{"full_text":"Art. 1. Text nou."}', NULL),
                        ('law:done', 'https://example.test/done', 'L2/2026', 'Done law',
                         'Done law', '[]', 'Lege 2/2026',
                         '{"full_text":"Art. 1. Text existent."}', '{"law_id":"law:done"}')
                    """
                )
                conn.commit()

                rows = _iter_laws_to_analyze(conn, force=False, limit=None)
                self.assertEqual([row["law_id"] for row in rows], ["law:new"])

                forced = _iter_laws_to_analyze(conn, force=True, limit=None)
                self.assertEqual(
                    [row["law_id"] for row in forced],
                    ["law:done", "law:new"],
                )


if __name__ == "__main__":
    unittest.main()
