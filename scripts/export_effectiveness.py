#!/usr/bin/env python3
"""
Export word- and letter-weighted productivity metrics from the current SQLite state.

Productivity is:

    words/letters in LLM-processed constructive interventions
    ---------------------------------------------------------
    words/letters in all LLM-processed interventions

The export is intentionally separate from the main intervention JSON files.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from init_db import DEFAULT_DB_PATH


DEFAULT_OUTPUT_DIR = Path("outputs/productivity")
DEFAULT_ANALYSIS_SOURCE = "llm_agent_v1"
WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def _clear_json_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for f in path.glob("*.json"):
        f.unlink(missing_ok=True)


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _letter_count(text: str) -> int:
    return len(LETTER_RE.findall(text or ""))


def _empty_stats() -> dict[str, Any]:
    return {
        "interventions_processed": 0,
        "constructive_interventions": 0,
        "non_constructive_interventions": 0,
        "total_word_count": 0,
        "constructive_word_count": 0,
        "non_constructive_word_count": 0,
        "total_letter_count": 0,
        "constructive_letter_count": 0,
        "non_constructive_letter_count": 0,
    }


def _add_intervention(
    stats: dict[str, Any],
    *,
    word_count: int,
    letter_count: int,
    constructive: bool,
    non_constructive: bool,
) -> None:
    stats["interventions_processed"] += 1
    stats["total_word_count"] += int(word_count)
    stats["total_letter_count"] += int(letter_count)
    if constructive:
        stats["constructive_interventions"] += 1
        stats["constructive_word_count"] += int(word_count)
        stats["constructive_letter_count"] += int(letter_count)
    if non_constructive:
        stats["non_constructive_interventions"] += 1
        stats["non_constructive_word_count"] += int(word_count)
        stats["non_constructive_letter_count"] += int(letter_count)


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    total_words = int(stats.get("total_word_count", 0))
    constructive_words = int(stats.get("constructive_word_count", 0))
    total_letters = int(stats.get("total_letter_count", 0))
    constructive_letters = int(stats.get("constructive_letter_count", 0))
    non_constructive_letters = int(stats.get("non_constructive_letter_count", 0))
    word_ratio = (constructive_words / total_words) if total_words else None
    letter_ratio = (constructive_letters / total_letters) if total_letters else None
    counterproductiveness_ratio = (
        non_constructive_letters / total_letters
        if total_letters
        else None
    )
    out = dict(stats)
    out["word_productivity_ratio"] = word_ratio
    out["word_productivity_pct"] = round(word_ratio * 100, 2) if word_ratio is not None else None
    out["letter_productivity_ratio"] = letter_ratio
    out["letter_productivity_pct"] = round(letter_ratio * 100, 2) if letter_ratio is not None else None
    # Generic productivity follows the letter-based metric.
    out["productivity_ratio"] = out["letter_productivity_ratio"]
    out["productivity_pct"] = out["letter_productivity_pct"]
    out["counterproductiveness_ratio"] = counterproductiveness_ratio
    out["counterproductiveness_pct"] = (
        round(counterproductiveness_ratio * 100, 2)
        if counterproductiveness_ratio is not None
        else None
    )
    return out


def _load_rows(db_path: Path, analysis_source: str) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT
                iv.intervention_id,
                iv.member_id,
                iv.raw_speaker,
                iv.normalized_speaker,
                iv.text,
                ia.relevance_label,
                m.name,
                m.party_id
            FROM intervention_analysis ia
            JOIN interventions_raw iv ON iv.intervention_id = ia.intervention_id
            LEFT JOIN members m ON m.member_id = iv.member_id
            WHERE ia.relevance_source = ?
            ORDER BY iv.member_id, m.party_id, iv.session_date, iv.session_id, iv.speech_index
            """,
            (analysis_source,),
        ).fetchall()


def _sort_productivity(entries: list[dict[str, Any]], id_key: str) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda x: (
            -(x.get("productivity_pct") if x.get("productivity_pct") is not None else -1),
            -int(x.get("total_letter_count", 0)),
            str(x.get(id_key) or ""),
        ),
    )


def export_productivity(
    *,
    db_path: Path,
    output_dir: Path,
    analysis_source: str = DEFAULT_ANALYSIS_SOURCE,
) -> dict[str, int]:
    rows = _load_rows(db_path, analysis_source)
    generated_at = _utc_now_iso()

    members: dict[str, dict[str, Any]] = {}
    parties: dict[str, dict[str, Any]] = {}
    parliament_total = _empty_stats()
    non_parliament_total = _empty_stats()
    all_total = _empty_stats()

    for row in rows:
        member_id = row["member_id"]
        party_id = row["party_id"] or "unknown"
        party_name = party_id if party_id != "unknown" else "Unknown"
        word_count = _word_count(row["text"] or "")
        letter_count = _letter_count(row["text"] or "")
        constructive = row["relevance_label"] == "constructive"
        non_constructive = row["relevance_label"] == "non_constructive"

        _add_intervention(
            all_total,
            word_count=word_count,
            letter_count=letter_count,
            constructive=constructive,
            non_constructive=non_constructive,
        )

        if member_id:
            _add_intervention(
                parliament_total,
                word_count=word_count,
                letter_count=letter_count,
                constructive=constructive,
                non_constructive=non_constructive,
            )

            if member_id not in members:
                members[member_id] = {
                    "member_id": member_id,
                    "name": row["name"] or row["normalized_speaker"] or row["raw_speaker"] or "",
                    "party_id": row["party_id"],
                    "party_name": row["party_id"],
                    **_empty_stats(),
                }
            _add_intervention(
                members[member_id],
                word_count=word_count,
                letter_count=letter_count,
                constructive=constructive,
                non_constructive=non_constructive,
            )

            if party_id not in parties:
                parties[party_id] = {
                    "party_id": party_id,
                    "party_name": party_name,
                    "members": {},
                    **_empty_stats(),
                }
            _add_intervention(
                parties[party_id],
                word_count=word_count,
                letter_count=letter_count,
                constructive=constructive,
                non_constructive=non_constructive,
            )
            parties[party_id]["members"].setdefault(
                member_id,
                {
                    "member_id": member_id,
                    "name": members[member_id]["name"],
                    "party_id": members[member_id]["party_id"],
                    "party_name": members[member_id]["party_name"],
                    **_empty_stats(),
                },
            )
            _add_intervention(
                parties[party_id]["members"][member_id],
                word_count=word_count,
                letter_count=letter_count,
                constructive=constructive,
                non_constructive=non_constructive,
            )
        else:
            _add_intervention(
                non_parliament_total,
                word_count=word_count,
                letter_count=letter_count,
                constructive=constructive,
                non_constructive=non_constructive,
            )

    members_dir = output_dir / "members"
    parties_dir = output_dir / "parties"
    _clear_json_files(output_dir)
    _clear_json_files(members_dir)
    _clear_json_files(parties_dir)

    metadata = {
        "generated_at": generated_at,
        "analysis_source": analysis_source,
        "productivity_definition": (
            "constructive_word_count / total_word_count and "
            "constructive_letter_count / total_letter_count, restricted to interventions "
            "already processed with the selected LLM analysis source"
        ),
        "counterproductiveness_definition": (
            "non_constructive_letter_count / total_letter_count, restricted to interventions "
            "already processed with the selected LLM analysis source"
        ),
        "word_count_source": "interventions_raw.text",
        "letter_count_source": "alphabetic Unicode characters in interventions_raw.text",
    }

    member_index: list[dict[str, Any]] = []
    for member_id, data in sorted(members.items()):
        finalized = _finalize_stats(data)
        detail = {**metadata, **finalized}
        member_slug = _slugify_name(finalized["name"])
        member_path = members_dir / f"productivity_{member_id}_{member_slug}.json"
        member_path.write_text(json.dumps(detail, ensure_ascii=True, indent=2), encoding="utf-8")

        member_index.append(
            {
                "member_id": finalized["member_id"],
                "name": finalized["name"],
                "party_id": finalized["party_id"],
                "party_name": finalized["party_name"],
                "interventions_processed": finalized["interventions_processed"],
                "constructive_interventions": finalized["constructive_interventions"],
                "non_constructive_interventions": finalized["non_constructive_interventions"],
                "total_word_count": finalized["total_word_count"],
                "constructive_word_count": finalized["constructive_word_count"],
                "non_constructive_word_count": finalized["non_constructive_word_count"],
                "total_letter_count": finalized["total_letter_count"],
                "constructive_letter_count": finalized["constructive_letter_count"],
                "non_constructive_letter_count": finalized["non_constructive_letter_count"],
                "word_productivity_ratio": finalized["word_productivity_ratio"],
                "word_productivity_pct": finalized["word_productivity_pct"],
                "letter_productivity_ratio": finalized["letter_productivity_ratio"],
                "letter_productivity_pct": finalized["letter_productivity_pct"],
                "productivity_ratio": finalized["productivity_ratio"],
                "productivity_pct": finalized["productivity_pct"],
                "counterproductiveness_ratio": finalized["counterproductiveness_ratio"],
                "counterproductiveness_pct": finalized["counterproductiveness_pct"],
            }
        )

    member_index = _sort_productivity(member_index, "member_id")
    (members_dir / "productivity_index.json").write_text(
        json.dumps(member_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    party_index: list[dict[str, Any]] = []
    for party_id, data in sorted(parties.items()):
        member_summaries = [
            _finalize_stats(member_data)
            for member_data in data.pop("members").values()
        ]
        member_summaries = _sort_productivity(member_summaries, "member_id")
        finalized = _finalize_stats(data)
        finalized["members_count"] = len(member_summaries)
        detail = {
            **metadata,
            **finalized,
            "members": member_summaries,
        }
        party_path = parties_dir / f"productivity_{party_id}.json"
        party_path.write_text(json.dumps(detail, ensure_ascii=True, indent=2), encoding="utf-8")

        party_index.append(
            {
                "party_id": finalized["party_id"],
                "party_name": finalized["party_name"],
                "members_count": finalized["members_count"],
                "interventions_processed": finalized["interventions_processed"],
                "constructive_interventions": finalized["constructive_interventions"],
                "non_constructive_interventions": finalized["non_constructive_interventions"],
                "total_word_count": finalized["total_word_count"],
                "constructive_word_count": finalized["constructive_word_count"],
                "non_constructive_word_count": finalized["non_constructive_word_count"],
                "total_letter_count": finalized["total_letter_count"],
                "constructive_letter_count": finalized["constructive_letter_count"],
                "non_constructive_letter_count": finalized["non_constructive_letter_count"],
                "word_productivity_ratio": finalized["word_productivity_ratio"],
                "word_productivity_pct": finalized["word_productivity_pct"],
                "letter_productivity_ratio": finalized["letter_productivity_ratio"],
                "letter_productivity_pct": finalized["letter_productivity_pct"],
                "productivity_ratio": finalized["productivity_ratio"],
                "productivity_pct": finalized["productivity_pct"],
                "counterproductiveness_ratio": finalized["counterproductiveness_ratio"],
                "counterproductiveness_pct": finalized["counterproductiveness_pct"],
            }
        )

    party_index = _sort_productivity(party_index, "party_id")
    (parties_dir / "productivity_index.json").write_text(
        json.dumps(party_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    total_payload = {
        **metadata,
        "parliament_members": _finalize_stats(parliament_total),
        "non_parliament_speakers": _finalize_stats(non_parliament_total),
        "all_llm_processed_speeches": _finalize_stats(all_total),
    }
    (output_dir / "productivity_total.json").write_text(
        json.dumps(total_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    return {
        "rows": len(rows),
        "members": len(member_index),
        "parties": len(party_index),
        "non_parliament_interventions": int(non_parliament_total["interventions_processed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export word- and letter-weighted productivity percentages from current DB state."
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--analysis-source",
        default=DEFAULT_ANALYSIS_SOURCE,
        help=f"Analysis source to include (default: {DEFAULT_ANALYSIS_SOURCE})",
    )
    args = parser.parse_args()

    summary = export_productivity(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
        analysis_source=args.analysis_source,
    )
    output_dir = Path(args.output_dir)
    print(
        "Productivity export complete: "
        f"{summary['rows']} LLM-processed intervention(s), "
        f"{summary['members']} member(s), "
        f"{summary['parties']} party file(s), "
        f"{summary['non_parliament_interventions']} non-parliament intervention(s)."
    )
    print("Outputs:")
    print(f"  Total:   {output_dir / 'productivity_total.json'}")
    print(f"  Members: {output_dir / 'members' / 'productivity_index.json'}")
    print(f"           {output_dir / 'members' / 'productivity_{member_id}_{name_slug}.json'}")
    print(f"  Parties: {output_dir / 'parties' / 'productivity_index.json'}")
    print(f"           {output_dir / 'parties' / 'productivity_{party_id}.json'}")
    print("Format:")
    print("  Each entry includes member_id/name/party_id/party_name when applicable,")
    print("  processed, constructive, and non-constructive intervention counts,")
    print("  total/constructive/non-constructive word and letter counts,")
    print("  word/letter productivity ratios, and letter-based counterproductiveness")
    print("  ratios plus percentages. Total JSON has parliament_members,")
    print("  non_parliament_speakers, and all_llm_processed_speeches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
