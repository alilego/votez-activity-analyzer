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
from export_activity import (  # noqa: E402
    _party_majority_threshold,
    _slugify_name,
    build_member_activity_snapshot,
    build_party_activity_snapshot,
    export_activity_snapshots,
)
from init_db import init_db  # noqa: E402


def _insert_member(
    conn: sqlite3.Connection,
    *,
    member_id: str,
    name: str,
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
            name.lower(),
            party_id,
            f"https://example.test/{member_id}",
        ),
    )


def _insert_law(
    conn: sqlite3.Connection,
    *,
    law_id: str,
    title: str,
    adopted_law_identifier: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_laws (
            law_id, source_url, identifier, title, details_text, columns_json,
            adopted_law_identifier
        )
        VALUES (?, ?, ?, ?, ?, '[]', ?)
        """,
        (
            law_id,
            f"https://example.test/{law_id}",
            law_id,
            title,
            title,
            adopted_law_identifier,
        ),
    )


def _link_member_law(
    conn: sqlite3.Connection,
    *,
    member_id: str,
    law_id: str,
    is_initiator: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_member_laws (member_id, law_id, is_initiator)
        VALUES (?, ?, ?)
        """,
        (member_id, law_id, 1 if is_initiator else 0),
    )


def _insert_motion(conn: sqlite3.Connection, *, motion_id: str, title: str) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_motions (motion_id, source_url, title, details_text, columns_json)
        VALUES (?, ?, ?, ?, '[]')
        """,
        (motion_id, f"https://example.test/{motion_id}", title, title),
    )


def _link_member_motion(conn: sqlite3.Connection, *, member_id: str, motion_id: str) -> None:
    conn.execute(
        "INSERT INTO dep_act_member_motions (member_id, motion_id) VALUES (?, ?)",
        (member_id, motion_id),
    )


def _insert_project(conn: sqlite3.Connection, *, project_id: str, title: str) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_decision_projects (
            decision_project_id, source_url, identifier, title, details_text, columns_json
        )
        VALUES (?, ?, ?, ?, ?, '[]')
        """,
        (project_id, f"https://example.test/{project_id}", project_id, title, title),
    )


def _link_member_project(conn: sqlite3.Connection, *, member_id: str, project_id: str) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_member_decision_projects (member_id, decision_project_id)
        VALUES (?, ?)
        """,
        (member_id, project_id),
    )


def _insert_question(
    conn: sqlite3.Connection,
    *,
    question_id: str,
    member_id: str,
    text: str,
    recipient: str = "Ministerul X",
    identifier: str = "nr.1",
) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_questions_interpellations (
            question_id, source_url, text, columns_json,
            member_id, identifier, recipient
        )
        VALUES (?, ?, ?, '[]', ?, ?, ?)
        """,
        (question_id, f"https://example.test/{question_id}", text, member_id, identifier, recipient),
    )


def _insert_declaration(
    conn: sqlite3.Connection,
    *,
    declaration_id: str,
    member_id: str,
    title: str,
) -> None:
    conn.execute(
        """
        INSERT INTO dep_act_political_declarations (
            political_declaration_id, member_id, source_url, text_url,
            title, full_text, details_text, columns_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, '[]')
        """,
        (
            declaration_id,
            member_id,
            f"https://example.test/{declaration_id}",
            f"https://example.test/{declaration_id}/text",
            title,
            title + " body text.",
            title,
        ),
    )


class TestExportActivityHelpers(unittest.TestCase):
    def test_party_majority_threshold_small_party_uses_half(self):
        # 16 members -> ceil(16/2) = 8 -> min(10, 8) = 8
        self.assertEqual(_party_majority_threshold(16), 8)

    def test_party_majority_threshold_large_party_caps_at_ten(self):
        self.assertEqual(_party_majority_threshold(129), 10)

    def test_party_majority_threshold_singleton_party(self):
        self.assertEqual(_party_majority_threshold(1), 1)

    def test_party_majority_threshold_empty_party(self):
        self.assertEqual(_party_majority_threshold(0), 0)

    def test_slugify_name_strips_diacritics_and_lowercases(self):
        self.assertEqual(_slugify_name("Șipoș Eugen-Cristian"), "sipos-eugen-cristian")

    def test_slugify_name_empty_falls_back_to_unknown(self):
        self.assertEqual(_slugify_name(""), "unknown")


class TestExportActivityAgainstFixtureDB(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "state.sqlite"
        init_db(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        ensure_activity_schema(self.conn)
        self._seed()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def _seed(self) -> None:
        # Two AUR members, two PSD members, one Neafiliat member.
        _insert_member(self.conn, member_id="deputat_1", name="Ana Pop", party_id="AUR")
        _insert_member(self.conn, member_id="deputat_2", name="Ben Ionescu", party_id="AUR")
        _insert_member(self.conn, member_id="deputat_3", name="Carmen Dan", party_id="PSD")
        _insert_member(self.conn, member_id="deputat_4", name="Dan Radu", party_id="PSD")
        _insert_member(self.conn, member_id="deputat_5", name="Eva Mihai", party_id=None)

        # Law 1: initiated by deputat_1 (AUR), co-supported by deputat_2, deputat_3, adopted.
        _insert_law(
            self.conn,
            law_id="law:aur_init",
            title="AUR-initiated law",
            adopted_law_identifier="Lege 1/2025",
        )
        _link_member_law(self.conn, member_id="deputat_1", law_id="law:aur_init", is_initiator=True)
        _link_member_law(self.conn, member_id="deputat_2", law_id="law:aur_init")
        _link_member_law(self.conn, member_id="deputat_3", law_id="law:aur_init")

        # Law 2: no AUR member initiated; both AUR members only support
        # (threshold for AUR=2 members is ceil(2/2)=1 -> min(10,1)=1).
        _insert_law(self.conn, law_id="law:both_supp", title="Majority-supported only law")
        _link_member_law(self.conn, member_id="deputat_1", law_id="law:both_supp")
        _link_member_law(self.conn, member_id="deputat_2", law_id="law:both_supp")

        # Motion 1: supported by 3 members total (both AUR + one PSD).
        _insert_motion(self.conn, motion_id="motion:1", title="Motion one")
        _link_member_motion(self.conn, member_id="deputat_1", motion_id="motion:1")
        _link_member_motion(self.conn, member_id="deputat_2", motion_id="motion:1")
        _link_member_motion(self.conn, member_id="deputat_3", motion_id="motion:1")

        # Decision project: deputat_1 (AUR) + deputat_3 (PSD).
        _insert_project(self.conn, project_id="decision:1", title="DP one")
        _link_member_project(self.conn, member_id="deputat_1", project_id="decision:1")
        _link_member_project(self.conn, member_id="deputat_3", project_id="decision:1")

        # Questions + declarations for deputat_1.
        _insert_question(self.conn, question_id="q:1", member_id="deputat_1", text="Q about X")
        _insert_question(self.conn, question_id="q:2", member_id="deputat_3", text="Q about Y")
        _insert_declaration(
            self.conn,
            declaration_id="decl:1",
            member_id="deputat_1",
            title="A political matter",
        )
        self.conn.commit()

    def test_export_writes_member_and_party_files(self):
        out_dir = self.tmp_path / "out"
        result = export_activity_snapshots(self.conn, output_dir=out_dir)
        self.assertEqual(result.members_written, 5)
        # AUR and PSD (Neafiliat member's party_id is NULL -> skipped)
        self.assertEqual(result.parties_written, 2)
        members_dir = out_dir / "members"
        parties_dir = out_dir / "parties"
        self.assertTrue((members_dir / "activity_deputat_1_ana-pop.json").exists())
        self.assertTrue((parties_dir / "activity_aur.json").exists())
        self.assertTrue((parties_dir / "activity_psd.json").exists())

    def test_member_snapshot_rolls_up_initiators_and_supporters(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        member_path = out_dir / "members" / "activity_deputat_1_ana-pop.json"
        data = json.loads(member_path.read_text())
        self.assertEqual(data["member_id"], "deputat_1")
        self.assertEqual(data["party_id"], "AUR")
        # The two laws this member is linked to:
        laws_by_id = {law["law_id"]: law for law in data["laws"]}
        initiated = laws_by_id["law:aur_init"]
        self.assertTrue(initiated["is_initiator"])
        self.assertTrue(initiated["is_adopted"])
        self.assertEqual(
            {p["party_name"] for p in initiated["initiator_parties"]}, {"AUR"}
        )
        self.assertEqual(
            {p["party_name"] for p in initiated["supporter_parties"]}, {"AUR", "PSD"}
        )
        supported = laws_by_id["law:both_supp"]
        self.assertFalse(supported["is_initiator"])
        self.assertFalse(supported["is_adopted"])

    def test_member_motion_co_supporting_parties_exclude_self(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        data = json.loads(
            (out_dir / "members" / "activity_deputat_1_ana-pop.json").read_text()
        )
        motion = data["motions"][0]
        counts = {p["party_name"]: p["members_count"] for p in motion["co_supporting_parties"]}
        # deputat_1 is excluded from the AUR count: 2 AUR supporters - self = 1.
        self.assertEqual(counts["AUR"], 1)
        self.assertEqual(counts["PSD"], 1)

    def test_member_decision_projects_group_by_party(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        data = json.loads(
            (out_dir / "members" / "activity_deputat_1_ana-pop.json").read_text()
        )
        project = data["decision_projects"][0]
        parties = {p["party_name"]: p for p in project["collaborating_parties"]}
        self.assertEqual(
            {m["member_id"] for m in parties["AUR"]["members"]}, {"deputat_1"}
        )
        self.assertEqual(
            {m["member_id"] for m in parties["PSD"]["members"]}, {"deputat_3"}
        )

    def test_member_without_party_falls_back_to_sentinel(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        data = json.loads(
            (out_dir / "members" / "activity_deputat_5_eva-mihai.json").read_text()
        )
        self.assertEqual(data["party_id"], "Neafiliat (no party)")

    def test_party_snapshot_aur_classifies_initiated_and_majority(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        data = json.loads((out_dir / "parties" / "activity_aur.json").read_text())
        self.assertEqual(data["members_count"], 2)
        self.assertEqual(data["majority_threshold"], 1)
        initiated_ids = {law["law_id"] for law in data["laws_initiated"]}
        majority_ids = {law["law_id"] for law in data["laws_majority_supported_only"]}
        self.assertEqual(initiated_ids, {"law:aur_init"})
        self.assertEqual(majority_ids, {"law:both_supp"})
        # The initiated law must not leak into the majority-supported bucket.
        self.assertFalse(initiated_ids & majority_ids)
        aur_initiated_law = data["laws_initiated"][0]
        self.assertEqual(
            {m["member_id"] for m in aur_initiated_law["party_initiators"]}, {"deputat_1"}
        )

    def test_party_snapshot_psd_no_initiation_no_majority(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        data = json.loads((out_dir / "parties" / "activity_psd.json").read_text())
        # PSD has 2 members, threshold=1; only deputat_3 supports the AUR-initiated
        # law (and it was initiated by AUR, not PSD) -> majority entry for PSD.
        self.assertEqual(data["majority_threshold"], 1)
        majority = {law["law_id"] for law in data["laws_majority_supported_only"]}
        self.assertEqual(majority, {"law:aur_init"})

    def test_party_questions_aggregates_members(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        aur = json.loads((out_dir / "parties" / "activity_aur.json").read_text())
        self.assertEqual(len(aur["questions_and_interpellations"]), 1)
        self.assertEqual(
            aur["questions_and_interpellations"][0]["asked_by"]["member_id"], "deputat_1"
        )

    def test_party_motion_majority_includes_under_threshold(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        aur = json.loads((out_dir / "parties" / "activity_aur.json").read_text())
        self.assertEqual(len(aur["motions_majority_supported"]), 1)
        motion = aur["motions_majority_supported"][0]
        self.assertEqual(motion["party_supporters_count"], 2)
        counts = {p["party_name"]: p["members_count"] for p in motion["all_supporting_parties"]}
        self.assertEqual(counts["AUR"], 2)
        self.assertEqual(counts["PSD"], 1)

    def test_export_is_idempotent_and_cleans_stale_files(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        stale = out_dir / "members" / "activity_deputat_999_ghost.json"
        stale.write_text("{}", encoding="utf-8")
        self.assertTrue(stale.exists())
        export_activity_snapshots(self.conn, output_dir=out_dir)
        self.assertFalse(stale.exists(), "stale activity_* files must be removed on rewrite")

    def test_export_preserves_unrelated_files_in_output_dir(self):
        out_dir = self.tmp_path / "out"
        export_activity_snapshots(self.conn, output_dir=out_dir)
        unrelated = out_dir / "members" / "notes.txt"
        unrelated.write_text("keep me", encoding="utf-8")
        export_activity_snapshots(self.conn, output_dir=out_dir)
        self.assertTrue(unrelated.exists())
        self.assertEqual(unrelated.read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
