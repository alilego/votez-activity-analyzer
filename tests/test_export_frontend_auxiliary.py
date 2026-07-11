from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_deputy_activity import ensure_activity_schema  # noqa: E402
from export_frontend_auxiliary import (  # noqa: E402
    ADOPTED_LAW_READER_SUMMARIES,
    ANALYSIS_INTERVALS,
    CAMERA_EVOLUTION,
    SENAT_EVOLUTION,
    export_frontend_auxiliary,
)
from init_db import init_db  # noqa: E402


class ExportFrontendAuxiliaryTests(unittest.TestCase):
    def test_exports_reader_summaries_and_updates_party_evolution_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite"
            output_dir = root / "outputs"
            frontend_data_dir = root / "frontend_data"
            (frontend_data_dir / "activity_analizer").mkdir(parents=True)

            db_path = init_db(db_path)
            with sqlite3.connect(db_path) as conn:
                ensure_activity_schema(conn)
                conn.executemany(
                    """
                    INSERT INTO members (
                        member_id, source_member_id, chamber, name, normalized_name,
                        party_id, profile_url, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("deputat_1", "1", "deputat", "A One", "a one", "PSD", "https://example.test/1", "2026-07-09 12:00:00"),
                        ("deputat_2", "2", "deputat", "B Two", "b two", "PNL", "https://example.test/2", "2026-07-09 12:00:00"),
                        ("senator_1", "1", "senator", "C Three", "c three", "USR", "https://example.test/3", "2026-07-09 12:00:00"),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, adopted_law_identifier,
                        title, details_text, columns_json, adopted_law_reader_summary
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:cdep:1",
                        "https://example.test/law",
                        "L1/2026",
                        "Lege 1/2026",
                        "Title",
                        "Details",
                        "[]",
                        json.dumps(
                            {
                                "schema_version": 1,
                                "law_id": "law:cdep:1",
                                "title": "Plain title",
                                "what_it_does": "Does one thing",
                                "practical_impact": ["Impact"],
                                "who_is_affected": ["Citizens"],
                            }
                        ),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO runs (run_id, started_at, finished_at, status)
                    VALUES ('run:test', '2026-06-01T00:00:00+00:00', '2026-06-01T00:01:00+00:00', 'completed')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO interventions_raw (
                        intervention_id, run_id, session_id, session_date, stenogram_path,
                        speech_index, raw_speaker, normalized_speaker, member_id, text, text_hash
                    )
                    VALUES ('i1', 'run:test', 's1', '2026-06-30', 'steno.json', 1, 'A One', 'a one', 'deputat_1', 'Text', 'hash')
                    """
                )
                conn.commit()

            historical_camera = [
                {
                    "year": 2024,
                    "month": 12,
                    "day": 23,
                    "total_seats": 2,
                    "parties": [{"name": "PSD", "seats": 2, "percentage": 100.0}],
                }
            ]
            historical_senat = [
                {
                    "year": 2024,
                    "month": 12,
                    "day": 23,
                    "total_seats": 1,
                    "parties": [{"name": "USR", "seats": 1, "percentage": 100.0}],
                }
            ]
            (frontend_data_dir / CAMERA_EVOLUTION).write_text(json.dumps(historical_camera), encoding="utf-8")
            (frontend_data_dir / SENAT_EVOLUTION).write_text(json.dumps(historical_senat), encoding="utf-8")
            (frontend_data_dir / "activity_analizer" / ANALYSIS_INTERVALS).write_text(
                json.dumps({"partyEvolution": {"startDate": "2024-12-23", "endDate": "2024-12-23"}}),
                encoding="utf-8",
            )

            result = export_frontend_auxiliary(
                db_path=db_path,
                output_dir=output_dir,
                frontend_data_dir=frontend_data_dir,
            )

            self.assertEqual(result["reader_summaries"], 1)
            summaries = json.loads((output_dir / ADOPTED_LAW_READER_SUMMARIES).read_text())
            self.assertEqual(summaries["law:cdep:1"]["what_it_does"], "Does one thing")
            self.assertEqual(summaries["law:cdep:1"]["adopted_law_identifier"], "Lege 1/2026")

            camera = json.loads((output_dir / CAMERA_EVOLUTION).read_text())
            self.assertEqual(len(camera), 2)
            self.assertEqual((camera[-1]["year"], camera[-1]["month"], camera[-1]["day"]), (2026, 7, 9))
            self.assertEqual(camera[-1]["total_seats"], 2)

            senat = json.loads((output_dir / SENAT_EVOLUTION).read_text())
            self.assertEqual(len(senat), 2)
            self.assertEqual((senat[-1]["year"], senat[-1]["month"], senat[-1]["day"]), (2026, 7, 9))

            intervals = json.loads((output_dir / ANALYSIS_INTERVALS).read_text())
            self.assertEqual(intervals["partyEvolution"]["endDate"], "2026-07-09")
            self.assertEqual(intervals["constructiveInterventions"]["endDate"], "2026-06-30")


if __name__ == "__main__":
    unittest.main()
