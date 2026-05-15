#!/usr/bin/env python3
"""
Export per-member and per-party activity snapshots from the crawler DB.

Writes two folders:

    outputs/activity/members/activity_<member_id>_<slug>.json
    outputs/activity/parties/activity_<party_slug>.json

Each member snapshot contains the member's identity plus four activity
blocks (motions, questions & interpellations, political declarations,
decision projects) and a laws block. The laws / decision projects /
motions blocks are enriched with party-level roll-ups (who else
collaborated, grouped by party).

Each party snapshot aggregates its members' activity:
  * `laws_initiated`         — any party member is an initiator.
  * `laws_majority_supported_only` — no party member initiated, but
    `min(10, ceil(party_total/2))` of them are supporters.
  * `questions_and_interpellations` — all party members' Q&I.
  * `motions_majority_supported` — at least `min(10, ceil(party_total/2))`
    of the party's members supported the motion.

Design notes
------------
* The DB has no `parties` table — `members.party_id` doubles as the
  party's display label (e.g. "PSD", "AUR"). We emit both `party_id`
  and `party_name` with the same value so the JSON shape is forward
  compatible if a mapping is introduced later.
* "Supports" a law means any row in `dep_act_member_laws`. "Initiates"
  is the same row with `is_initiator=1`. Initiators are always a
  subset of supporters. The per-member output therefore exposes `laws`
  (everything the member is linked to) with a per-law `is_initiator`
  flag, and also rolls up both `initiator_parties` and
  `supporter_parties`.
* On each export we wipe the two output folders first. This keeps the
  snapshots idempotent: files for deleted members / parties never
  linger.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ACTIVITY_OUTPUT_DIR = Path("outputs/activity")


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _slugify_name(value: str) -> str:
    """Unicode-tolerant slug suitable for filenames (matches export_outputs)."""
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def _safe_columns(columns_json: str | None) -> list[Any]:
    if not columns_json:
        return []
    try:
        value = json.loads(columns_json)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return value
    return []


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


# ---------------------------------------------------------------------------
# DB loaders. Each loader does exactly one query and returns plain dicts so
# downstream aggregations can stay pure Python (and unit-testable).
# ---------------------------------------------------------------------------


def _load_members_index(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT member_id, chamber, name, normalized_name, party_id, profile_url
        FROM members
        ORDER BY member_id
        """
    ).fetchall()
    return {
        member_id: {
            "member_id": member_id,
            "chamber": chamber,
            "name": name,
            "normalized_name": normalized_name,
            "party_id": party_id,
            "profile_url": profile_url,
        }
        for (
            member_id,
            chamber,
            name,
            normalized_name,
            party_id,
            profile_url,
        ) in rows
    }


def _party_id_to_total(members: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for info in members.values():
        party_id = info.get("party_id") or ""
        if not party_id:
            continue
        totals[party_id] = totals.get(party_id, 0) + 1
    return totals


def _party_majority_threshold(total_members: int) -> int:
    """
    `min(10, ceil(party_total / 2))`. Returns at least 1 when total >= 1
    so one-member parties still have a non-zero threshold (avoids a
    divide-by-zero-style edge case in the threshold comparison).
    """
    if total_members <= 0:
        return 0
    return min(10, max(1, math.ceil(total_members / 2)))


def _load_motions(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT motion_id, source_url, title, details_text, columns_json
        FROM dep_act_motions
        """
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for motion_id, source_url, title, details_text, columns_json in rows:
        out[motion_id] = {
            "motion_id": motion_id,
            "source_url": source_url,
            "title": title,
            "details_text": details_text,
            "columns": _safe_columns(columns_json),
        }
    return out


def _load_member_motions(
    conn: sqlite3.Connection,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rows = conn.execute(
        "SELECT member_id, motion_id FROM dep_act_member_motions"
    ).fetchall()
    by_member: dict[str, list[str]] = {}
    by_motion: dict[str, list[str]] = {}
    for member_id, motion_id in rows:
        by_member.setdefault(member_id, []).append(motion_id)
        by_motion.setdefault(motion_id, []).append(member_id)
    return by_member, by_motion


def _load_decision_projects(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT decision_project_id, source_url, identifier, title,
               details_text, columns_json
        FROM dep_act_decision_projects
        """
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for (
        decision_project_id,
        source_url,
        identifier,
        title,
        details_text,
        columns_json,
    ) in rows:
        out[decision_project_id] = {
            "decision_project_id": decision_project_id,
            "source_url": source_url,
            "identifier": identifier,
            "title": title,
            "details_text": details_text,
            "columns": _safe_columns(columns_json),
        }
    return out


def _load_member_decision_projects(
    conn: sqlite3.Connection,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rows = conn.execute(
        "SELECT member_id, decision_project_id FROM dep_act_member_decision_projects"
    ).fetchall()
    by_member: dict[str, list[str]] = {}
    by_project: dict[str, list[str]] = {}
    for member_id, project_id in rows:
        by_member.setdefault(member_id, []).append(project_id)
        by_project.setdefault(project_id, []).append(member_id)
    return by_member, by_project


def _load_laws(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT law_id, source_url, identifier, title, details_text, columns_json,
               adopted_law_identifier, motive_pdf_url, initiators_text,
               initiators_source
        FROM dep_act_laws
        """
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for (
        law_id,
        source_url,
        identifier,
        title,
        details_text,
        columns_json,
        adopted_law_identifier,
        motive_pdf_url,
        initiators_text,
        initiators_source,
    ) in rows:
        out[law_id] = {
            "law_id": law_id,
            "source_url": source_url,
            "identifier": identifier,
            "title": title,
            "details_text": details_text,
            "columns": _safe_columns(columns_json),
            "adopted_law_identifier": adopted_law_identifier,
            "is_adopted": _nonempty(adopted_law_identifier),
            "motive_pdf_url": motive_pdf_url,
            "initiators_text": initiators_text,
            "initiators_source": initiators_source,
        }
    return out


def _load_member_laws(
    conn: sqlite3.Connection,
) -> tuple[
    dict[str, list[tuple[str, bool]]],
    dict[str, list[tuple[str, bool]]],
]:
    """Return (by_member, by_law) where each list entry is (id, is_initiator)."""
    rows = conn.execute(
        "SELECT member_id, law_id, is_initiator FROM dep_act_member_laws"
    ).fetchall()
    by_member: dict[str, list[tuple[str, bool]]] = {}
    by_law: dict[str, list[tuple[str, bool]]] = {}
    for member_id, law_id, is_initiator in rows:
        flag = bool(is_initiator)
        by_member.setdefault(member_id, []).append((law_id, flag))
        by_law.setdefault(law_id, []).append((member_id, flag))
    return by_member, by_law


def _load_questions_by_member(
    conn: sqlite3.Connection,
) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT question_id, member_id, source_url, identifier, text, recipient,
               columns_json
        FROM dep_act_questions_interpellations
        WHERE member_id IS NOT NULL
        ORDER BY question_id
        """
    ).fetchall()
    by_member: dict[str, list[dict[str, Any]]] = {}
    for (
        question_id,
        member_id,
        source_url,
        identifier,
        text,
        recipient,
        columns_json,
    ) in rows:
        by_member.setdefault(member_id, []).append(
            {
                "question_id": question_id,
                "source_url": source_url,
                "identifier": identifier,
                "text": text,
                "recipient": recipient,
                "columns": _safe_columns(columns_json),
            }
        )
    return by_member


def _load_declarations_by_member(
    conn: sqlite3.Connection,
) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT political_declaration_id, member_id, source_url, text_url, title,
               full_text, details_text, columns_json
        FROM dep_act_political_declarations
        ORDER BY political_declaration_id
        """
    ).fetchall()
    by_member: dict[str, list[dict[str, Any]]] = {}
    for (
        declaration_id,
        member_id,
        source_url,
        text_url,
        title,
        full_text,
        details_text,
        columns_json,
    ) in rows:
        by_member.setdefault(member_id, []).append(
            {
                "political_declaration_id": declaration_id,
                "source_url": source_url,
                "text_url": text_url,
                "title": title,
                "full_text": full_text,
                "details_text": details_text,
                "columns": _safe_columns(columns_json),
            }
        )
    return by_member


# ---------------------------------------------------------------------------
# Aggregation helpers.
# ---------------------------------------------------------------------------


def _member_stub(member: dict[str, Any]) -> dict[str, Any]:
    """Minimal {member_id, name} object used inside roll-ups."""
    return {
        "member_id": member["member_id"],
        "name": member.get("name"),
    }


def _group_by_party(
    member_ids: list[str],
    members: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group a list of member ids by party, returning a sorted list of
    `{party_id, party_name, members_count, members: [...]}` blocks.

    Members whose party_id is NULL are collected under the sentinel
    label `"Neafiliat (no party)"` so the output never loses a member.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for mid in member_ids:
        info = members.get(mid)
        if info is None:
            buckets.setdefault("Unknown", []).append(
                {"member_id": mid, "name": None}
            )
            continue
        key = info.get("party_id") or "Neafiliat (no party)"
        buckets.setdefault(key, []).append(_member_stub(info))
    out: list[dict[str, Any]] = []
    for party_id in sorted(buckets.keys()):
        bucket = sorted(buckets[party_id], key=lambda x: (x["name"] or "", x["member_id"]))
        out.append(
            {
                "party_id": party_id,
                "party_name": party_id,
                "members_count": len(bucket),
                "members": bucket,
            }
        )
    return out


def _party_member_counts(
    member_ids: list[str],
    members: dict[str, dict[str, Any]],
    *,
    exclude_member_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return a `[{party_id, party_name, members_count}]` list counting how
    many of `member_ids` belong to each party. When `exclude_member_id`
    is given, it is subtracted from its own party's count (but the
    party still appears in the list — this is the "include his party
    but count the OTHER members" behaviour).
    """
    counts: dict[str, int] = {}
    for mid in member_ids:
        info = members.get(mid)
        if info is None:
            continue
        key = info.get("party_id") or "Neafiliat (no party)"
        counts[key] = counts.get(key, 0) + 1
    if exclude_member_id and exclude_member_id in members:
        key = members[exclude_member_id].get("party_id") or "Neafiliat (no party)"
        if key in counts:
            counts[key] -= 1
    return [
        {
            "party_id": party_id,
            "party_name": party_id,
            "members_count": counts[party_id],
        }
        for party_id in sorted(counts.keys())
    ]


def _motion_supporter_count_for_party(
    supporter_ids: list[str],
    members: dict[str, dict[str, Any]],
    party_id: str,
) -> int:
    """How many members of `party_id` appear in `supporter_ids`."""
    return sum(
        1
        for mid in supporter_ids
        if members.get(mid, {}).get("party_id") == party_id
    )


# ---------------------------------------------------------------------------
# Per-member snapshot builder.
# ---------------------------------------------------------------------------


def _build_member_motions_block(
    member_id: str,
    member_motion_ids: list[str],
    motions: dict[str, dict[str, Any]],
    motion_supporters: dict[str, list[str]],
    members: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for motion_id in sorted(member_motion_ids):
        motion = motions.get(motion_id)
        if motion is None:
            continue
        supporters = motion_supporters.get(motion_id, [])
        co_support = _party_member_counts(
            supporters, members, exclude_member_id=member_id
        )
        entry = dict(motion)
        entry["co_supporting_parties"] = co_support
        out.append(entry)
    return out


def _build_member_decision_projects_block(
    member_decision_project_ids: list[str],
    projects: dict[str, dict[str, Any]],
    project_collaborators: dict[str, list[str]],
    members: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for project_id in sorted(member_decision_project_ids):
        project = projects.get(project_id)
        if project is None:
            continue
        collaborators = project_collaborators.get(project_id, [])
        entry = dict(project)
        entry["collaborating_parties"] = _group_by_party(collaborators, members)
        out.append(entry)
    return out


def _build_member_laws_block(
    member_id: str,
    member_law_rows: list[tuple[str, bool]],
    laws: dict[str, dict[str, Any]],
    law_memberships: dict[str, list[tuple[str, bool]]],
    members: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for law_id, is_initiator_for_member in sorted(member_law_rows):
        law = laws.get(law_id)
        if law is None:
            continue
        memberships = law_memberships.get(law_id, [])
        supporter_ids = [mid for mid, _ in memberships]
        initiator_ids = [mid for mid, flag in memberships if flag]
        entry = dict(law)
        entry["is_initiator"] = bool(is_initiator_for_member)
        entry["initiator_parties"] = _group_by_party(initiator_ids, members)
        entry["supporter_parties"] = _group_by_party(supporter_ids, members)
        out.append(entry)
    return out


def build_member_activity_snapshot(
    member: dict[str, Any],
    *,
    members: dict[str, dict[str, Any]],
    motions: dict[str, dict[str, Any]],
    motion_supporters: dict[str, list[str]],
    member_motion_ids: list[str],
    decision_projects: dict[str, dict[str, Any]],
    project_collaborators: dict[str, list[str]],
    member_project_ids: list[str],
    laws: dict[str, dict[str, Any]],
    law_memberships: dict[str, list[tuple[str, bool]]],
    member_law_rows: list[tuple[str, bool]],
    questions: list[dict[str, Any]],
    declarations: list[dict[str, Any]],
) -> dict[str, Any]:
    member_id = member["member_id"]
    party_id = member.get("party_id") or "Neafiliat (no party)"
    return {
        "generated_at": _iso_now(),
        "member_id": member_id,
        "name": member.get("name"),
        "chamber": member.get("chamber"),
        "party_id": party_id,
        "party_name": party_id,
        "profile_url": member.get("profile_url"),
        "motions": _build_member_motions_block(
            member_id,
            member_motion_ids,
            motions,
            motion_supporters,
            members,
        ),
        "questions_and_interpellations": list(questions),
        "political_declarations": list(declarations),
        "decision_projects": _build_member_decision_projects_block(
            member_project_ids,
            decision_projects,
            project_collaborators,
            members,
        ),
        "laws": _build_member_laws_block(
            member_id,
            member_law_rows,
            laws,
            law_memberships,
            members,
        ),
    }


# ---------------------------------------------------------------------------
# Per-party snapshot builder.
# ---------------------------------------------------------------------------


def _member_ids_in_party(
    members: dict[str, dict[str, Any]], party_id: str
) -> set[str]:
    return {
        mid for mid, info in members.items() if info.get("party_id") == party_id
    }


def _build_party_laws_blocks(
    party_id: str,
    party_member_ids: set[str],
    majority_threshold: int,
    laws: dict[str, dict[str, Any]],
    law_memberships: dict[str, list[tuple[str, bool]]],
    members: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initiated: list[dict[str, Any]] = []
    majority_supported: list[dict[str, Any]] = []
    for law_id in sorted(law_memberships.keys()):
        memberships = law_memberships[law_id]
        party_initiators = [
            mid for mid, flag in memberships if flag and mid in party_member_ids
        ]
        party_supporters = [mid for mid, _ in memberships if mid in party_member_ids]
        law = laws.get(law_id)
        if law is None:
            continue
        supporter_ids = [mid for mid, _ in memberships]
        initiator_ids = [mid for mid, flag in memberships if flag]
        if party_initiators:
            entry = dict(law)
            entry["party_initiators"] = [
                _member_stub(members[mid]) for mid in sorted(party_initiators)
            ]
            entry["party_supporters_count"] = len(party_supporters)
            entry["initiator_parties"] = _group_by_party(initiator_ids, members)
            entry["supporter_parties"] = _group_by_party(supporter_ids, members)
            initiated.append(entry)
            continue
        # No initiator from this party — check the majority rule.
        supporter_count_this_party = len(party_supporters)
        if (
            majority_threshold > 0
            and supporter_count_this_party >= majority_threshold
        ):
            entry = dict(law)
            entry["party_supporters_count"] = supporter_count_this_party
            entry["party_supporters"] = [
                _member_stub(members[mid]) for mid in sorted(party_supporters)
            ]
            entry["initiator_parties"] = _group_by_party(initiator_ids, members)
            entry["supporter_parties"] = _group_by_party(supporter_ids, members)
            majority_supported.append(entry)
    return initiated, majority_supported


def _build_party_questions_block(
    party_member_ids: set[str],
    members: dict[str, dict[str, Any]],
    questions_by_member: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mid in sorted(party_member_ids):
        for q in questions_by_member.get(mid, []):
            entry = dict(q)
            entry["asked_by"] = _member_stub(members[mid])
            out.append(entry)
    return out


def _build_party_motions_block(
    party_id: str,
    party_member_ids: set[str],
    majority_threshold: int,
    motions: dict[str, dict[str, Any]],
    motion_supporters: dict[str, list[str]],
    members: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for motion_id in sorted(motion_supporters.keys()):
        supporters = motion_supporters[motion_id]
        party_supporters = [mid for mid in supporters if mid in party_member_ids]
        if not majority_threshold or len(party_supporters) < majority_threshold:
            continue
        motion = motions.get(motion_id)
        if motion is None:
            continue
        entry = dict(motion)
        entry["party_supporters_count"] = len(party_supporters)
        entry["party_supporters"] = [
            _member_stub(members[mid]) for mid in sorted(party_supporters)
        ]
        entry["all_supporting_parties"] = _party_member_counts(supporters, members)
        out.append(entry)
    return out


def build_party_activity_snapshot(
    party_id: str,
    *,
    members: dict[str, dict[str, Any]],
    laws: dict[str, dict[str, Any]],
    law_memberships: dict[str, list[tuple[str, bool]]],
    motions: dict[str, dict[str, Any]],
    motion_supporters: dict[str, list[str]],
    questions_by_member: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    party_member_ids = _member_ids_in_party(members, party_id)
    total = len(party_member_ids)
    threshold = _party_majority_threshold(total)
    laws_initiated, laws_majority_supported = _build_party_laws_blocks(
        party_id,
        party_member_ids,
        threshold,
        laws,
        law_memberships,
        members,
    )
    return {
        "generated_at": _iso_now(),
        "party_id": party_id,
        "party_name": party_id,
        "members_count": total,
        "majority_threshold": threshold,
        "majority_rule": (
            "min(10, ceil(members_count / 2)); a law/motion is included "
            "when the count of party members who (non-initiator) supported "
            "it is >= this threshold."
        ),
        "laws_initiated": laws_initiated,
        "laws_majority_supported_only": laws_majority_supported,
        "questions_and_interpellations": _build_party_questions_block(
            party_member_ids, members, questions_by_member
        ),
        "motions_majority_supported": _build_party_motions_block(
            party_id,
            party_member_ids,
            threshold,
            motions,
            motion_supporters,
            members,
        ),
    }


# ---------------------------------------------------------------------------
# Top-level export entry point.
# ---------------------------------------------------------------------------


class ExportResult:
    """Lightweight container so the caller can report what was written."""

    __slots__ = ("members_written", "parties_written", "output_dir")

    def __init__(
        self, members_written: int, parties_written: int, output_dir: Path
    ) -> None:
        self.members_written = members_written
        self.parties_written = parties_written
        self.output_dir = output_dir

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ExportResult(members={self.members_written}, "
            f"parties={self.parties_written}, dir={self.output_dir})"
        )


def _wipe_activity_dir(directory: Path) -> None:
    """Remove `activity_*.json` files from `directory` (leave other files alone)."""
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and path.name.startswith("activity_") and path.suffix == ".json":
            path.unlink()


def _write_member_file(
    members_dir: Path, member: dict[str, Any], snapshot: dict[str, Any]
) -> Path:
    name_slug = _slugify_name(member.get("name") or member["member_id"])
    path = members_dir / f"activity_{member['member_id']}_{name_slug}.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _write_party_file(
    parties_dir: Path, party_id: str, snapshot: dict[str, Any]
) -> Path:
    slug = _slugify_name(party_id)
    path = parties_dir / f"activity_{slug}.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def export_activity_snapshots(
    conn: sqlite3.Connection,
    *,
    output_dir: Path = DEFAULT_ACTIVITY_OUTPUT_DIR,
    progress_prefix: str = "",
) -> ExportResult:
    """Write one JSON file per member and per party describing their crawled
    activity. Idempotent: previous `activity_*.json` files in the two
    target folders are removed before new ones are written.
    """
    prefix = f"{progress_prefix} " if progress_prefix else ""
    members_dir = output_dir / "members"
    parties_dir = output_dir / "parties"
    members_dir.mkdir(parents=True, exist_ok=True)
    parties_dir.mkdir(parents=True, exist_ok=True)
    _wipe_activity_dir(members_dir)
    _wipe_activity_dir(parties_dir)

    members = _load_members_index(conn)
    motions = _load_motions(conn)
    member_motions, motion_supporters = _load_member_motions(conn)
    decision_projects = _load_decision_projects(conn)
    member_projects, project_collaborators = _load_member_decision_projects(conn)
    laws = _load_laws(conn)
    member_laws, law_memberships = _load_member_laws(conn)
    questions_by_member = _load_questions_by_member(conn)
    declarations_by_member = _load_declarations_by_member(conn)

    print(
        f"{prefix}Activity export: {len(members)} member(s), "
        f"{len(motions)} motion(s), {len(decision_projects)} decision "
        f"project(s), {len(laws)} law(s), "
        f"{sum(len(v) for v in questions_by_member.values())} question(s), "
        f"{sum(len(v) for v in declarations_by_member.values())} declaration(s).",
        flush=True,
    )

    members_written = 0
    for member_id, member in members.items():
        snapshot = build_member_activity_snapshot(
            member,
            members=members,
            motions=motions,
            motion_supporters=motion_supporters,
            member_motion_ids=member_motions.get(member_id, []),
            decision_projects=decision_projects,
            project_collaborators=project_collaborators,
            member_project_ids=member_projects.get(member_id, []),
            laws=laws,
            law_memberships=law_memberships,
            member_law_rows=member_laws.get(member_id, []),
            questions=questions_by_member.get(member_id, []),
            declarations=declarations_by_member.get(member_id, []),
        )
        _write_member_file(members_dir, member, snapshot)
        members_written += 1

    parties_written = 0
    party_ids = sorted(
        {info.get("party_id") for info in members.values() if info.get("party_id")}
    )
    for party_id in party_ids:
        snapshot = build_party_activity_snapshot(
            party_id,
            members=members,
            laws=laws,
            law_memberships=law_memberships,
            motions=motions,
            motion_supporters=motion_supporters,
            questions_by_member=questions_by_member,
        )
        _write_party_file(parties_dir, party_id, snapshot)
        parties_written += 1

    print(
        f"{prefix}Activity export complete: wrote {members_written} "
        f"member file(s) to {members_dir} and {parties_written} "
        f"party file(s) to {parties_dir}.",
        flush=True,
    )
    return ExportResult(members_written, parties_written, output_dir)
