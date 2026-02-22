#!/usr/bin/env python3
"""
Export frontend JSON artifacts from local SQLite state.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from init_db import DEFAULT_DB_PATH, init_db


def _map_label(label: str) -> str:
    if label in {"relevant", "neutral", "non_relevant"}:
        return label
    # Scaffolding fallback until classifier is integrated.
    return "neutral"


def _safe_topics(topics_json: str) -> list[str]:
    try:
        data = json.loads(topics_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            value = item.strip()
            if value:
                out.append(value)
    return out


def _top_topics(counter: Counter[str], limit: int = 20) -> list[dict]:
    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [{"topic": topic, "count": count} for topic, count in ranked[:limit]]


def _clear_json_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for f in path.glob("*.json"):
        f.unlink(missing_ok=True)


def _slugify_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def _load_session_links() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(Path("input/stenograme").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        session_id = str(data.get("session_id", "")).strip()
        source_url = str(data.get("source_url", "")).strip()
        if session_id and source_url:
            mapping[session_id] = source_url
    return mapping


def export_outputs(db_path: Path, output_dir: Path) -> tuple[int, int]:
    init_db(db_path)
    session_links = _load_session_links()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                iv.member_id,
                m.name,
                m.party_id,
                iv.session_id,
                iv.session_date,
                iv.stenogram_path,
                iv.text,
                COALESCE(ia.relevance_label, 'unknown') AS relevance_label,
                COALESCE(ia.topics_json, '[]') AS topics_json,
                ia.confidence
            FROM interventions_raw iv
            JOIN members m ON m.member_id = iv.member_id
            LEFT JOIN intervention_analysis ia ON ia.intervention_id = iv.intervention_id
            WHERE iv.member_id IS NOT NULL
            ORDER BY iv.member_id, iv.session_date, iv.session_id, iv.speech_index
            """
        ).fetchall()

    member_data: dict[str, dict] = {}
    for row in rows:
        (
            member_id,
            member_name,
            party_id,
            session_id,
            session_date,
            stenogram_path,
            text,
            relevance_label_raw,
            topics_json,
            confidence,
        ) = row
        relevance_label = _map_label(relevance_label_raw)
        topics = _safe_topics(topics_json)
        confidence_value = float(confidence) if confidence is not None else 0.0
        stenogram_name = Path(stenogram_path).name

        if member_id not in member_data:
            member_data[member_id] = {
                "member_id": member_id,
                "name": member_name,
                "party_id": party_id,
                "party_name": party_id,
                "counts": {"relevant": 0, "neutral": 0, "non_relevant": 0},
                "topics_counter": Counter(),
                "interventions": {"relevant": [], "neutral": [], "non_relevant": []},
            }

        md = member_data[member_id]
        md["counts"][relevance_label] += 1
        md["topics_counter"].update(topics)
        md["interventions"][relevance_label].append(
            {
                "session_id": session_id,
                "session_date": session_date,
                "text": text or "",
                "topics": topics,
                "confidence": confidence_value,
                "stenogram_name": stenogram_name,
                "stenogram_link": session_links.get(str(session_id), ""),
            }
        )

    members_dir = output_dir / "members"
    parties_dir = output_dir / "parties"
    _clear_json_files(members_dir)
    _clear_json_files(parties_dir)

    members_index = []
    for member_id in sorted(member_data.keys()):
        md = member_data[member_id]
        counts = md["counts"]
        interventions_total = counts["relevant"] + counts["neutral"] + counts["non_relevant"]
        top_topics = _top_topics(md["topics_counter"])
        members_index.append(
            {
                "member_id": md["member_id"],
                "name": md["name"],
                "party_id": md["party_id"],
                "party_name": md["party_name"],
                "interventions_total": interventions_total,
                "relevant_count": counts["relevant"],
                "neutral_count": counts["neutral"],
                "non_relevant_count": counts["non_relevant"],
                "top_topics": top_topics,
            }
        )

        member_detail = {
            "member_id": md["member_id"],
            "name": md["name"],
            "party_id": md["party_id"],
            "party_name": md["party_name"],
            "stats": {
                "interventions_total": interventions_total,
                "relevant_count": counts["relevant"],
                "neutral_count": counts["neutral"],
                "non_relevant_count": counts["non_relevant"],
            },
            "top_topics": top_topics,
            "interventions": md["interventions"],
        }
        member_name_slug = _slugify_name(md["name"])
        (members_dir / f"interventions_{member_id}_{member_name_slug}.json").write_text(
            json.dumps(member_detail, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    members_index = sorted(members_index, key=lambda x: (-x["interventions_total"], x["member_id"]))
    (members_dir / "interventions_index.json").write_text(
        json.dumps(members_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    party_to_members: dict[str, list[dict]] = defaultdict(list)
    for member_entry in members_index:
        party_key = member_entry["party_id"] if member_entry["party_id"] else "unknown"
        party_to_members[party_key].append(member_entry)

    parties_index = []
    for party_id in sorted(party_to_members.keys()):
        members = party_to_members[party_id]
        party_name = party_id if party_id != "unknown" else "Unknown"

        counts = {
            "relevant": sum(m["relevant_count"] for m in members),
            "neutral": sum(m["neutral_count"] for m in members),
            "non_relevant": sum(m["non_relevant_count"] for m in members),
        }
        interventions_total = counts["relevant"] + counts["neutral"] + counts["non_relevant"]

        topic_counter: Counter[str] = Counter()
        for m in members:
            for topic in m["top_topics"]:
                topic_counter[topic["topic"]] += int(topic["count"])
        top_topics = _top_topics(topic_counter)

        party_index_entry = {
            "party_id": party_id,
            "party_name": party_name,
            "members_count": len(members),
            "interventions_total": interventions_total,
            "relevant_count": counts["relevant"],
            "neutral_count": counts["neutral"],
            "non_relevant_count": counts["non_relevant"],
            "top_topics": top_topics,
        }
        parties_index.append(party_index_entry)

        party_detail = {
            "party_id": party_id,
            "party_name": party_name,
            "stats": {
                "members_count": len(members),
                "interventions_total": interventions_total,
                "relevant_count": counts["relevant"],
                "neutral_count": counts["neutral"],
                "non_relevant_count": counts["non_relevant"],
            },
            "top_topics": top_topics,
            "members": [
                {
                    "member_id": m["member_id"],
                    "name": m["name"],
                    "interventions_total": m["interventions_total"],
                    "relevant_count": m["relevant_count"],
                    "neutral_count": m["neutral_count"],
                    "non_relevant_count": m["non_relevant_count"],
                    "top_topics": m["top_topics"],
                }
                for m in sorted(members, key=lambda x: (-x["interventions_total"], x["member_id"]))
            ],
        }
        (parties_dir / f"interventions_{party_id}.json").write_text(
            json.dumps(party_detail, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    parties_index = sorted(parties_index, key=lambda x: (-x["interventions_total"], x["party_id"]))
    (parties_dir / "interventions_index.json").write_text(
        json.dumps(parties_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return len(members_index), len(parties_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export frontend JSON outputs from DB state.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory root (default: outputs)",
    )
    args = parser.parse_args()

    members_count, parties_count = export_outputs(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
    )
    print(
        "Export completed: "
        f"{members_count} members, {parties_count} parties. "
        f"Output root: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
