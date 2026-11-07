import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import crawl_deputy_activity as crawler
from init_db import init_db


def add_law(conn, law_id):
    conn.execute("INSERT INTO dep_act_laws (law_id, source_url, identifier, title, details_text, columns_json) VALUES (?, ?, ?, ?, ?, ?)",
                 (law_id, f"https://example.test/{law_id}", law_id, law_id, "Details", "[]"))
    conn.execute("INSERT INTO dep_act_member_laws (member_id, law_id) VALUES ('deputat_cdep_2024_2_1', ?)", (law_id,))


def add_member(conn):
    conn.execute("INSERT INTO members (member_id,source_member_id,chamber,name,normalized_name,party_id,profile_url) VALUES ('deputat_cdep_2024_2_1','cdep_2024_2_1','deputat','Test Member','test member','TST','https://www.cdep.ro/ords/pls/parlam/structura2015.mp?leg=2024&cam=2&idm=1')")


class NewLawScopeTests(unittest.TestCase):
    def test_excluding_existing_laws_happens_before_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db) as conn:
                add_member(conn)
                for law_id in ("a_old", "b_new", "c_new"):
                    add_law(conn, law_id)
                records = crawler.load_law_records_for_initiator_hydration(
                    conn, member_ids=["deputat_cdep_2024_2_1"], exclude_law_ids={"a_old"}, limit=1)
                self.assertEqual([r.record_id for r in records], ["b_new"])
                cached_records = crawler.load_law_records_for_initiator_hydration(
                    conn, member_ids=["deputat_cdep_2024_2_1"], exclude_law_ids={"a_old"}, limit=1,
                    law_initiator_pdf_dir=Path(tmp), prefer_without_cached_pdf_when_limited=True)
                self.assertEqual([r.record_id for r in cached_records], ["b_new"])

    def test_crawl_snapshots_old_ids_before_discovering_new_laws(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db) as conn:
                add_member(conn)
                add_law(conn, "old")
            member = {"member_id": "deputat_cdep_2024_2_1", "name": "Test Member", "profile_url": "https://www.cdep.ro/ords/pls/parlam/structura2015.mp?leg=2024&cam=2&idm=1", "chamber": "deputat"}
            baseline_path = Path(tmp) / "law_baseline.json"

            def crawl_member(conn, **kwargs):
                add_law(conn, "new")
                return {}, {}, None

            with mock.patch.object(crawler, "_load_deputies", return_value=[member]), \
                    mock.patch.object(crawler, "_select_members", return_value=[member]), \
                    mock.patch.object(crawler, "crawl_member", side_effect=crawl_member), \
                    mock.patch.object(crawler, "run_law_initiator_hydration_phase") as hydrate:
                result = crawler.crawl_deputy_activity(
                    db_path=db, input_path=Path(tmp) / "input.json", hydrate_law_initiators=True,
                    hydrate_new_laws_only=True, new_laws_baseline=baseline_path, sleep_seconds=0)
            self.assertEqual(result, 0)
            self.assertEqual(hydrate.call_args.kwargs["exclude_law_ids"], {"old"})
            # A resumed crawl must still hydrate "new", now already stored.
            with mock.patch.object(crawler, "_load_deputies", return_value=[member]), \
                    mock.patch.object(crawler, "crawl_member", return_value=({}, {}, None)), \
                    mock.patch.object(crawler, "run_law_initiator_hydration_phase") as hydrate:
                crawler.crawl_deputy_activity(db_path=db,input_path=Path(tmp)/"input.json",
                    hydrate_law_initiators=True,hydrate_new_laws_only=True,
                    only_hydrate_law_initiators=True,new_laws_baseline=baseline_path,sleep_seconds=0)
            self.assertEqual(hydrate.call_args.kwargs["exclude_law_ids"], {"old"})


if __name__ == "__main__":
    unittest.main()
