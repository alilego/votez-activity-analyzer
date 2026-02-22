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
        "relevant_count",
        "neutral_count",
        "non_relevant_count",
        "top_topics",
    }
    for m in members_index:
        if set(required_member_keys) - set(m.keys()):
            errors.append(f"Member index entry missing keys: {m.get('member_id', '<unknown>')}")
            continue
        total = int(m["interventions_total"])
        parts_sum = int(m["relevant_count"]) + int(m["neutral_count"]) + int(m["non_relevant_count"])
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
        if int(stats.get("relevant_count", -1)) != int(m["relevant_count"]):
            errors.append(f"Member detail relevant_count mismatch for {m['member_id']}")
        if int(stats.get("neutral_count", -1)) != int(m["neutral_count"]):
            errors.append(f"Member detail neutral_count mismatch for {m['member_id']}")
        if int(stats.get("non_relevant_count", -1)) != int(m["non_relevant_count"]):
            errors.append(f"Member detail non_relevant_count mismatch for {m['member_id']}")
        for label in ("relevant", "neutral", "non_relevant"):
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
        "relevant_count",
        "neutral_count",
        "non_relevant_count",
        "top_topics",
    }
    by_party = {}
    for m in members_index:
        key = _party_key(m["party_id"])
        if key not in by_party:
            by_party[key] = {
                "members_count": 0,
                "interventions_total": 0,
                "relevant_count": 0,
                "neutral_count": 0,
                "non_relevant_count": 0,
            }
        agg = by_party[key]
        agg["members_count"] += 1
        agg["interventions_total"] += int(m["interventions_total"])
        agg["relevant_count"] += int(m["relevant_count"])
        agg["neutral_count"] += int(m["neutral_count"])
        agg["non_relevant_count"] += int(m["non_relevant_count"])

    for p in parties_index:
        if set(required_party_keys) - set(p.keys()):
            errors.append(f"Party index entry missing keys: {p.get('party_id', '<unknown>')}")
            continue
        pid = p["party_id"]
        agg = by_party.get(pid)
        if agg is None:
            errors.append(f"Party index contains unknown party_id with no members: {pid}")
            continue
        for key in ("members_count", "interventions_total", "relevant_count", "neutral_count", "non_relevant_count"):
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
        for key in ("members_count", "interventions_total", "relevant_count", "neutral_count", "non_relevant_count"):
            if int(stats.get(key, -1)) != int(p[key]):
                errors.append(f"Party detail mismatch for {pid} on {key}")
        members_list = detail.get("members", [])
        if len(members_list) != int(p["members_count"]):
            errors.append(f"Party detail member list length mismatch for {pid}")
        if not _is_sorted_count_then_id(members_list, "interventions_total", "member_id"):
            errors.append(f"Party detail members not sorted for {pid}")

    if not _is_sorted_count_then_id(parties_index, "interventions_total", "party_id"):
        errors.append("parties/interventions_index.json is not deterministically sorted.")

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
