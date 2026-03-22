from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agenda import extract_agenda_from_session


class TestAgendaExtraction(unittest.TestCase):
    def test_empty_inputs(self):
        result = extract_agenda_from_session("", [])
        self.assertEqual(result, [])

    def test_notes_only_chair_info(self):
        notes = (
            "Şedinţa a început la ora 16.00. Lucrările şedinţei au fost conduse "
            "de domnul deputat Vasile-Daniel Suciu, vicepreşedinte al Camerei Deputaţilor."
        )
        result = extract_agenda_from_session(notes, [])
        self.assertEqual(result, [])

    def test_chair_introduces_pl_x(self):
        speeches = [
            {
                "text": (
                    "Intrăm în ordinea de zi. Proiectul de Lege privind aprobarea "
                    "Ordonanţei Guvernului nr. 33/2024, transmis cu adresa PL-x 556/2024. "
                    "Comisia pentru buget prezintă raportul."
                ),
            },
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertGreaterEqual(len(result), 1)
        item = result[0]
        self.assertEqual(item["item_number"], 1)
        self.assertIn("PL-x 556/2024", item["law_id"])
        self.assertIn("Proiectul de Lege", item["title"])

    def test_phcd_reference(self):
        speeches = [
            {
                "text": (
                    "Proiectul de Hotărâre privind aprobarea componenţei nominale "
                    "şi a conducerii Delegaţiei, PHCD 20/2025. "
                    "Proiectul de hotărâre a fost distribuit."
                ),
            },
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0]["law_id"], "PHCD 20/2025")

    def test_propunere_legislativa(self):
        speeches = [
            {
                "text": (
                    "Propunerea legislativă pentru modificarea şi completarea "
                    "Legii cadastrului, transmisă cu PL-x 210/2014. "
                    "Raportul comisiei sesizate în fond."
                ),
            },
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("PL-x 210/2014", result[0]["law_id"])

    def test_deduplication_by_law_id(self):
        speeches = [
            {
                "text": (
                    "Proiectul de Lege privind aprobarea OUG, PL-x 556/2024. "
                    "Raportul comisiei."
                ),
            },
            {
                "text": (
                    "Raport asupra Proiectului de Lege privind aprobarea, "
                    "PL-x 556/2024. Comisia adoptă raportul."
                ),
            },
        ]
        result = extract_agenda_from_session("", speeches)
        pl_ids = [item["law_id"] for item in result if item.get("law_id") == "PL-x 556/2024"]
        self.assertEqual(len(pl_ids), 1)

    def test_multiple_agenda_items(self):
        speeches = [
            {
                "text": (
                    "Proiectul de Lege pentru modificarea art. 5, "
                    "transmis cu PL-x 433/2022. Raportul comisiei."
                ),
            },
            {
                "text": (
                    "Proiectul de Lege privind instituirea săptămânii "
                    "15-21 mai, PL-x 593/2024. Raportul comisiei."
                ),
            },
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["item_number"], 1)
        self.assertEqual(result[1]["item_number"], 2)

    def test_skips_vote_confirmation(self):
        speeches = [
            {"text": "Proiectul de lege rămâne la votul final."},
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertEqual(result, [])

    def test_skips_very_short_text(self):
        speeches = [
            {"text": "Vă mulţumesc."},
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertEqual(result, [])

    def test_max_items_respected(self):
        speeches = [
            {
                "text": (
                    f"Proiectul de Lege privind actul normativ {i}, "
                    f"PL-x {i}/2025. Raportul comisiei."
                ),
            }
            for i in range(1, 35)
        ]
        result = extract_agenda_from_session("", speeches, max_items=5)
        self.assertLessEqual(len(result), 5)

    def test_oug_reference(self):
        speeches = [
            {
                "text": (
                    "Proiectul de Lege privind aprobarea Ordonanţei de urgenţă "
                    "a Guvernului nr. 212/2020. Raportul comisiei."
                ),
            },
        ]
        result = extract_agenda_from_session("", speeches)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("OUG nr. 212/2020", result[0]["law_id"])


if __name__ == "__main__":
    unittest.main()
