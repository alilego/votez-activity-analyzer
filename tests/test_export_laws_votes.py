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
from export_laws_votes import export_laws_votes_json  # noqa: E402
from init_db import init_db  # noqa: E402


def _insert_member(
    conn: sqlite3.Connection,
    *,
    member_id: str,
    name: str,
    normalized_name: str,
    party_id: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO members (
            member_id, source_member_id, chamber, name, normalized_name,
            party_id, profile_url
        )
        VALUES (?, ?, 'deputat', ?, ?, ?, ?)
        """,
        (
            member_id,
            member_id.removeprefix("deputat_"),
            name,
            normalized_name,
            party_id,
            f"https://example.test/{member_id}",
        ),
    )


class ExportLawsVotesTests(unittest.TestCase):
    def test_groups_by_day_and_counts_distinct_yes_per_party(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.sqlite"
            init_db(db_path)
            conn = sqlite3.connect(db_path)
            try:
                ensure_activity_schema(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text, columns_json,
                        law_status, adopted_law_pdf_url,
                        adopted_law_text_json, adopted_law_analysis_json, adopted_law_reader_summary
                    )
                    VALUES (
                        'law:test:1', 'https://example.test/law1', 'PL-x 1/2025',
                        'Title', 'Details', '[]',
                        'adoptata_in_parlament', 'https://pdf.example/law.pdf',
                        '{"pages":[]}', '{"impact":"low"}', 'Short summary'
                    )
                    """
                )
                _insert_member(
                    conn,
                    member_id="deputat_a",
                    name="Alpha One",
                    normalized_name="alpha one",
                    party_id="PNL",
                )
                _insert_member(
                    conn,
                    member_id="deputat_b",
                    name="Beta Two",
                    normalized_name="beta two",
                    party_id="USR",
                )
                _insert_member(
                    conn,
                    member_id="deputat_c",
                    name="Gamma Dup",
                    normalized_name="gamma dup",
                    party_id="PNL",
                )
                # Same deputy, two final-style YES rows same calendar day → count once.
                conn.executemany(
                    """
                    INSERT INTO dep_act_laws_votes (
                        member_normalized_name, law_id, vote_date, vote_type, vote
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        ("alpha one", "law:test:1", "2025-06-01 10:00", "final_adoption", "YES"),
                        ("alpha one", "law:test:1", "2025-06-01 14:00", "final_vote", "YES"),
                        ("beta two", "law:test:1", "2025-06-01 11:00", "final_adoption", "YES"),
                        ("gamma dup", "law:test:1", "2025-06-02 09:00", "final_adoption", "YES"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            out_path = Path(tmp) / "laws_votes.json"
            n = export_laws_votes_json(db_path=db_path, output_path=out_path)
            self.assertEqual(n, 1)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("generated_at", data)
            law = data["laws"][0]
            self.assertEqual(law["law_id"], "law:test:1")
            self.assertEqual(law["status"], "adoptata_in_parlament")
            self.assertEqual(law["details"]["details_text"], "Details")
            self.assertEqual(law["details"]["columns_json"], [])
            self.assertEqual(law["details"]["adopted_law_pdf_url"], "https://pdf.example/law.pdf")
            self.assertEqual(law["details"]["adopted_law_text"], {"pages": []})
            self.assertEqual(law["details"]["adopted_law_analysis"], {"impact": "low"})
            self.assertEqual(law["details"]["adopted_law_summary"], "Short summary")

            sessions = {s["vote_date"]: s["parties"] for s in law["voting_sessions"]}
            self.assertEqual(set(sessions), {"2025-06-01", "2025-06-02"})
            day1 = {p["party_id"]: p["member_count"] for p in sessions["2025-06-01"]}
            self.assertEqual(day1.get("PNL"), 1)
            self.assertEqual(day1.get("USR"), 1)
            day2 = {p["party_id"]: p["member_count"] for p in sessions["2025-06-02"]}
            self.assertEqual(day2.get("PNL"), 1)


if __name__ == "__main__":
    unittest.main()
