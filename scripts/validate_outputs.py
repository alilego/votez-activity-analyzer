#!/usr/bin/env python3
"""
Validate exported frontend artifacts under outputs/.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _party_key(value):
    return value if value else "unknown"


def _is_sorted_count_then_id(items: list[dict], count_key: str, id_key: str) -> bool:
    expected = sorted(items, key=lambda x: (-int(x[count_key]), str(x[id_key])))
    return items == expected


def _is_sorted_topics(topics: list[dict]) -> bool:
    expected = sorted(topics, key=lambda x: (-int(x["count"]), str(x["topic"])))
    return topics == expected


def _slugify_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def validate_outputs(output_dir: Path) -> list[str]:
    errors: list[str] = []
    members_dir = output_dir / "members"
    parties_dir = output_dir / "parties"
    topics_dir = output_dir / "topics"

    members_index_path = members_dir / "interventions_index.json"
    parties_index_path = parties_dir / "interventions_index.json"
    if not members_index_path.exists():
        errors.append(f"Missing file: {members_index_path}")
        return errors
    if not parties_index_path.exists():
        errors.append(f"Missing file: {parties_index_path}")
        return errors

    members_index = _load_json(members_index_path)
    parties_index = _load_json(parties_index_path)
    if not isinstance(members_index, list):
        errors.append("members/interventions_index.json must be a JSON array.")
        return errors
    if not isinstance(parties_index, list):
        errors.append("parties/interventions_index.json must be a JSON array.")
        return errors

    # Members index + detail checks.
    required_member_keys = {
        "member_id",
        "name",
        "party_id",
        "party_name",
        "interventions_total",
        "constructive_count",
        "neutral_count",
        "non_constructive_count",
        "top_topics",
    }
    for m in members_index:
        if set(required_member_keys) - set(m.keys()):
            errors.append(f"Member index entry missing keys: {m.get('member_id', '<unknown>')}")
            continue
        total = int(m["interventions_total"])
        parts_sum = int(m["constructive_count"]) + int(m["neutral_count"]) + int(m["non_constructive_count"])
        if total != parts_sum:
            errors.append(f"Member index count mismatch for {m['member_id']}: total != label sum")
        topics = m["top_topics"]
        if not isinstance(topics, list):
            errors.append(f"top_topics must be list for member {m['member_id']}")
        else:
            if len(topics) > 20:
                errors.append(f"top_topics > 20 for member {m['member_id']}")
            if not _is_sorted_topics(topics):
                errors.append(f"top_topics unsorted for member {m['member_id']}")

        detail_path = members_dir / f"interventions_{m['member_id']}_{_slugify_name(m['name'])}.json"
        if not detail_path.exists():
            errors.append(f"Missing member detail file: {detail_path.name}")
            continue
        detail = _load_json(detail_path)
        stats = detail.get("stats", {})
        interventions = detail.get("interventions", {})
        if int(stats.get("interventions_total", -1)) != total:
            errors.append(f"Member detail total mismatch for {m['member_id']}")
        if int(stats.get("constructive_count", -1)) != int(m["constructive_count"]):
            errors.append(f"Member detail constructive_count mismatch for {m['member_id']}")
        if int(stats.get("neutral_count", -1)) != int(m["neutral_count"]):
            errors.append(f"Member detail neutral_count mismatch for {m['member_id']}")
        if int(stats.get("non_constructive_count", -1)) != int(m["non_constructive_count"]):
            errors.append(f"Member detail non_constructive_count mismatch for {m['member_id']}")
        for label in ("constructive", "neutral", "non_constructive"):
            if not isinstance(interventions.get(label, []), list):
                errors.append(f"Member detail interventions.{label} is not list for {m['member_id']}")
            elif len(interventions[label]) != int(m[f"{label}_count"]):
                errors.append(f"Member detail interventions.{label} length mismatch for {m['member_id']}")
            else:
                for entry in interventions[label]:
                    link = str(entry.get("stenogram_link", "")).strip()
                    if not link:
                        errors.append(f"Missing stenogram_link for member {m['member_id']} ({label})")

    if not _is_sorted_count_then_id(members_index, "interventions_total", "member_id"):
        errors.append("members/interventions_index.json is not deterministically sorted.")

    # Parties checks.
    required_party_keys = {
        "party_id",
        "party_name",
        "members_count",
        "interventions_total",
        "constructive_count",
        "neutral_count",
        "non_constructive_count",
        "top_topics",
    }
    by_party = {}
    for m in members_index:
        key = _party_key(m["party_id"])
        if key not in by_party:
            by_party[key] = {
                "members_count": 0,
                "interventions_total": 0,
                "constructive_count": 0,
                "neutral_count": 0,
                "non_constructive_count": 0,
            }
        agg = by_party[key]
        agg["members_count"] += 1
        agg["interventions_total"] += int(m["interventions_total"])
        agg["constructive_count"] += int(m["constructive_count"])
        agg["neutral_count"] += int(m["neutral_count"])
        agg["non_constructive_count"] += int(m["non_constructive_count"])

    for p in parties_index:
        if set(required_party_keys) - set(p.keys()):
            errors.append(f"Party index entry missing keys: {p.get('party_id', '<unknown>')}")
            continue
        pid = p["party_id"]
        agg = by_party.get(pid)
        if agg is None:
            errors.append(f"Party index contains unknown party_id with no members: {pid}")
            continue
        for key in ("members_count", "interventions_total", "constructive_count", "neutral_count", "non_constructive_count"):
            if int(p[key]) != int(agg[key]):
                errors.append(f"Party index mismatch for {pid} on {key}")
        if len(p["top_topics"]) > 20:
            errors.append(f"top_topics > 20 for party {pid}")
        if not _is_sorted_topics(p["top_topics"]):
            errors.append(f"top_topics unsorted for party {pid}")

        detail_path = parties_dir / f"interventions_{pid}.json"
        if not detail_path.exists():
            errors.append(f"Missing party detail file: {detail_path.name}")
            continue
        detail = _load_json(detail_path)
        stats = detail.get("stats", {})
        for key in ("members_count", "interventions_total", "constructive_count", "neutral_count", "non_constructive_count"):
            if int(stats.get(key, -1)) != int(p[key]):
                errors.append(f"Party detail mismatch for {pid} on {key}")
        members_list = detail.get("members", [])
        if len(members_list) != int(p["members_count"]):
            errors.append(f"Party detail member list length mismatch for {pid}")
        if not _is_sorted_count_then_id(members_list, "interventions_total", "member_id"):
            errors.append(f"Party detail members not sorted for {pid}")

    if not _is_sorted_count_then_id(parties_index, "interventions_total", "party_id"):
        errors.append("parties/interventions_index.json is not deterministically sorted.")

    # Global canonical topics checks (new artifact).
    topics_index_path = topics_dir / "interventions_topics_index.json"
    if not topics_index_path.exists():
        errors.append(f"Missing file: {topics_index_path}")
        return errors
    topics_index = _load_json(topics_index_path)
    if not isinstance(topics_index, dict):
        errors.append("topics/interventions_topics_index.json must be a JSON object.")
        return errors
    top_topics = topics_index.get("top_topics", [])
    if not isinstance(top_topics, list):
        errors.append("topics/interventions_topics_index.json: top_topics must be a list.")
        return errors
    prev_count = None
    prev_topic = None
    for item in top_topics:
        if not isinstance(item, dict):
            errors.append("topics/interventions_topics_index.json: each top_topics item must be an object.")
            continue
        if "topic" not in item or "count" not in item:
            errors.append("topics/interventions_topics_index.json: item missing topic/count.")
            continue
        topic = str(item["topic"])
        count = int(item["count"])
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            errors.append(f"topics/interventions_topics_index.json: aliases must be list for topic '{topic}'.")
        if prev_count is not None:
            if count > prev_count or (count == prev_count and topic < str(prev_topic)):
                errors.append("topics/interventions_topics_index.json: top_topics is not sorted by count desc, topic asc.")
                break
        prev_count = count
        prev_topic = topic

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exported outputs.")
    parser.add_argument("--output-dir", default="outputs", help="Output root directory (default: outputs)")
    args = parser.parse_args()

    errors = validate_outputs(Path(args.output_dir))
    if errors:
        print(f"Validation failed: {len(errors)} issue(s)")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Validation passed: outputs are internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
