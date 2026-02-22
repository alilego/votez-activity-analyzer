#!/usr/bin/env python3
"""
Analyzer entrypoint used by run_pipeline.py.

Current slice:
- load selected stenograms
- normalize speakers
- resolve speakers to members using registry snapshots
- persist raw interventions + unmatched speakers
- store per-run summary in state DB and JSON artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from init_db import DEFAULT_DB_PATH, init_db


@dataclass(frozen=True)
class SessionStats:
    path: str
    session_id: str
    stenograma_date: str
    speeches_count: int


def _load_selected_paths(stenogram_list_path: Path) -> list[str]:
    payload = json.loads(stenogram_list_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("Invalid stenogram list format: expected key 'files' as list.")
    return [str(x) for x in files]


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level JSON object.")
    return data


def _normalize_for_matching(text: str) -> str:
    text = text or ""
    # Remove role/appositive suffixes like " - viceprim-ministru, ..."
    text = re.sub(r"\s+-\s+.*$", "", text)
    text = re.sub(r"^\s*(domnul|doamna)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def _token_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text)
    parts = [p for p in cleaned.split() if p]
    return " ".join(sorted(parts))


def _extract_source_member_id(row: dict, chamber: str) -> str:
    value = str(row.get("id", "")).strip()
    if value:
        return value
    profile_url = str(row.get("profile_url", "")).strip()
    if chamber == "senator" and profile_url:
        parsed = urllib.parse.urlparse(profile_url)
        qs = urllib.parse.parse_qs(parsed.query)
        pid = qs.get("ParlamentarID", [])
        if pid and pid[0].strip():
            return pid[0].strip()
    if profile_url:
        return profile_url
    # Last fallback keeps deterministic identity.
    name = str(row.get("name", "")).strip()
    return hashlib.sha1(name.encode("utf-8")).hexdigest()


def _merge_speech_text(speech: dict) -> str:
    parts = []
    for key in ("text", "text2", "text3"):
        value = speech.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


def _build_intervention_id(stenogram_path: str, speech_index: int) -> str:
    path_hash = hashlib.sha1(stenogram_path.encode("utf-8")).hexdigest()[:12]
    return f"iv:{path_hash}:{speech_index}"


def _load_registry_members() -> tuple[list[dict], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    registries = [
        (Path("input/toti_deputatii.json"), "deputat"),
        (Path("input/toti_senatorii.json"), "senator"),
    ]
    members: list[dict] = []
    normalized_to_ids: dict[str, list[str]] = {}
    token_to_ids: dict[str, list[str]] = {}
    alias_to_ids: dict[str, list[str]] = {}

    for path, chamber in registries:
        payload = _read_json(path)
        rows = payload.get("members")
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected 'members' list.")
        for row in rows:
            if not isinstance(row, dict):
                continue
            source_member_id = _extract_source_member_id(row, chamber)
            name = str(row.get("name", "")).strip()
            if not source_member_id or not name:
                continue
            member_id = f"{chamber}_{source_member_id}"
            normalized_name = _normalize_for_matching(name)
            member = {
                "member_id": member_id,
                "source_member_id": source_member_id,
                "chamber": chamber,
                "name": name,
                "normalized_name": normalized_name,
                "party_id": str(row.get("party", "")).strip() or None,
                "profile_url": str(row.get("profile_url", "")).strip() or None,
                "circumscriptie": (
                    str(row.get("circumscriptie", "")).strip() if row.get("circumscriptie") is not None else None
                ),
            }
            members.append(member)
            normalized_to_ids.setdefault(normalized_name, []).append(member_id)
            token_to_ids.setdefault(_token_key(normalized_name), []).append(member_id)
            # Alias: "surname given-names" -> "given-names surname"
            name_parts = normalized_name.split()
            if len(name_parts) > 1:
                rotated = " ".join(name_parts[1:] + [name_parts[0]])
                alias_to_ids.setdefault(rotated, []).append(member_id)
    return members, normalized_to_ids, token_to_ids, alias_to_ids


def _extract_session_stats(path: Path, data: dict) -> SessionStats:
    session_id = str(data.get("session_id", ""))
    session_date = str(data.get("stenograma_date", ""))
    speeches = data.get("speeches", [])
    if not isinstance(speeches, list):
        raise ValueError(f"{path}: expected 'speeches' to be a list.")
    return SessionStats(
        path=path.as_posix(),
        session_id=session_id,
        stenograma_date=session_date,
        speeches_count=len(speeches),
    )


def _persist_run_data(
    conn: sqlite3.Connection,
    run_id: str,
    members: list[dict],
    interventions: list[dict],
    unmatched_counts: dict[tuple[str, str, str, str], int],
) -> tuple[int, int]:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        INSERT INTO runs (
            run_id,
            started_at,
            finished_at,
            status,
            sessions_processed,
            interventions_total,
            interventions_classified,
            unmatched_speakers
        )
        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'completed', 0, 0, 0, 0)
        ON CONFLICT(run_id) DO NOTHING
        """,
        (run_id,),
    )

    for member in members:
        conn.execute(
            """
            INSERT INTO members (
                member_id, source_member_id, chamber, name, normalized_name,
                party_id, profile_url, circumscriptie, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(member_id) DO UPDATE SET
                source_member_id = excluded.source_member_id,
                chamber = excluded.chamber,
                name = excluded.name,
                normalized_name = excluded.normalized_name,
                party_id = excluded.party_id,
                profile_url = excluded.profile_url,
                circumscriptie = excluded.circumscriptie,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                member["member_id"],
                member["source_member_id"],
                member["chamber"],
                member["name"],
                member["normalized_name"],
                member["party_id"],
                member["profile_url"],
                member["circumscriptie"],
            ),
        )

    for iv in interventions:
        conn.execute(
            """
            INSERT INTO interventions_raw (
                intervention_id, run_id, session_id, session_date, stenogram_path, speech_index,
                raw_speaker, normalized_speaker, member_id, text, text_hash, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(intervention_id) DO UPDATE SET
                run_id = excluded.run_id,
                session_id = excluded.session_id,
                session_date = excluded.session_date,
                stenogram_path = excluded.stenogram_path,
                speech_index = excluded.speech_index,
                raw_speaker = excluded.raw_speaker,
                normalized_speaker = excluded.normalized_speaker,
                member_id = excluded.member_id,
                text = excluded.text,
                text_hash = excluded.text_hash,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                iv["intervention_id"],
                run_id,
                iv["session_id"],
                iv["session_date"],
                iv["stenogram_path"],
                iv["speech_index"],
                iv["raw_speaker"],
                iv["normalized_speaker"],
                iv["member_id"],
                iv["text"],
                iv["text_hash"],
            ),
        )

        # Scaffolding stage: write placeholder analysis row for matched interventions.
        if iv["member_id"] is not None:
            conn.execute(
                """
                INSERT INTO intervention_analysis (
                    intervention_id,
                    run_id,
                    relevance_label,
                    topics_json,
                    confidence,
                    evidence_chunk_ids_json,
                    analysis_version,
                    updated_at
                )
                VALUES (?, ?, 'unknown', '[]', NULL, '[]', 'scaffold_v1', CURRENT_TIMESTAMP)
                ON CONFLICT(intervention_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    iv["intervention_id"],
                    run_id,
                ),
            )

    for (session_id, stenogram_path, raw_speaker, normalized_speaker), occurrences in unmatched_counts.items():
        conn.execute(
            """
            INSERT INTO unmatched_speakers (
                run_id, session_id, stenogram_path, raw_speaker, normalized_speaker, occurrences, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, session_id, normalized_speaker) DO UPDATE SET
                stenogram_path = excluded.stenogram_path,
                raw_speaker = excluded.raw_speaker,
                occurrences = unmatched_speakers.occurrences + excluded.occurrences,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                run_id,
                session_id,
                stenogram_path,
                raw_speaker,
                normalized_speaker,
                occurrences,
            ),
        )
    conn.commit()
    return len(interventions), sum(unmatched_counts.values())


def _build_summary_payload(
    run_id: str,
    sessions: list[SessionStats],
    interventions_total: int,
    matched_total: int,
    unmatched_total: int,
) -> dict:
    return {
        "run_id": run_id,
        "sessions_count": len(sessions),
        "speeches_total": sum(s.speeches_count for s in sessions),
        "interventions_total": interventions_total,
        "matched_interventions_total": matched_total,
        "unmatched_interventions_total": unmatched_total,
        "sessions": [
            {
                "path": s.path,
                "session_id": s.session_id,
                "session_date": s.stenograma_date,
                "speeches_count": s.speeches_count,
            }
            for s in sessions
        ],
    }


def _write_summary_file(run_id: str, payload: dict) -> Path:
    out_dir = Path("state/run_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_analysis_summary.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _store_summary_in_db(conn: sqlite3.Connection, payload: dict) -> None:
    conn.execute(
        """
        INSERT INTO run_outputs (
            run_id,
            sessions_count,
            speeches_total,
            summary_json,
            updated_at
        )
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(run_id) DO UPDATE SET
            sessions_count = excluded.sessions_count,
            speeches_total = excluded.speeches_total,
            summary_json = excluded.summary_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            str(payload["run_id"]),
            int(payload["sessions_count"]),
            int(payload["speeches_total"]),
            json.dumps(payload, ensure_ascii=True),
        ),
    )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze selected stenograms (speaker resolution slice).")
    parser.add_argument(
        "--run-id",
        default=os.environ.get("VOTEZ_RUN_ID"),
        help="Run ID (defaults to VOTEZ_RUN_ID env var)",
    )
    parser.add_argument(
        "--stenogram-list-path",
        default=os.environ.get("VOTEZ_STENOGRAM_LIST_PATH"),
        help="Path to selected stenograms JSON list (defaults to VOTEZ_STENOGRAM_LIST_PATH env var)",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB file for state (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    if not args.run_id:
        print("ERROR: Missing run ID. Provide --run-id or VOTEZ_RUN_ID.")
        return 1
    if not args.stenogram_list_path:
        print("ERROR: Missing stenogram list path. Provide --stenogram-list-path or VOTEZ_STENOGRAM_LIST_PATH.")
        return 1

    db_path = Path(args.db_path)
    init_db(db_path)

    list_path = Path(args.stenogram_list_path)
    if not list_path.exists():
        print(f"ERROR: Stenogram list file not found: {list_path}")
        return 1

    try:
        selected = _load_selected_paths(list_path)
        members, normalized_to_ids, token_to_ids, alias_to_ids = _load_registry_members()

        sessions: list[SessionStats] = []
        interventions: list[dict] = []
        unmatched_counts: dict[tuple[str, str, str, str], int] = {}

        for rel_path in selected:
            file_path = Path(rel_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Selected stenogram path not found: {file_path}")
            data = _read_json(file_path)
            stats = _extract_session_stats(file_path, data)
            sessions.append(stats)

            speeches = data.get("speeches", [])
            session_id = str(data.get("session_id", ""))
            session_date = str(data.get("stenograma_date", ""))
            stenogram_path = file_path.as_posix()
            for idx, speech in enumerate(speeches):
                if not isinstance(speech, dict):
                    continue
                raw_speaker = str(speech.get("speaker", "")).strip()
                normalized_speaker = _normalize_for_matching(raw_speaker)
                matched_ids = normalized_to_ids.get(normalized_speaker, [])
                if len(matched_ids) == 1:
                    member_id = matched_ids[0]
                else:
                    alias_ids = alias_to_ids.get(normalized_speaker, [])
                    if len(alias_ids) == 1:
                        member_id = alias_ids[0]
                    else:
                        token_ids = token_to_ids.get(_token_key(normalized_speaker), [])
                        member_id = token_ids[0] if len(token_ids) == 1 else None

                text = _merge_speech_text(speech)
                intervention_id = _build_intervention_id(stenogram_path, idx)
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                interventions.append(
                    {
                        "intervention_id": intervention_id,
                        "session_id": session_id,
                        "session_date": session_date,
                        "stenogram_path": stenogram_path,
                        "speech_index": idx,
                        "raw_speaker": raw_speaker,
                        "normalized_speaker": normalized_speaker,
                        "member_id": member_id,
                        "text": text,
                        "text_hash": text_hash,
                    }
                )
                if member_id is None and normalized_speaker:
                    key = (session_id, stenogram_path, raw_speaker, normalized_speaker)
                    unmatched_counts[key] = unmatched_counts.get(key, 0) + 1

        with sqlite3.connect(db_path) as conn:
            interventions_total, unmatched_total = _persist_run_data(
                conn=conn,
                run_id=args.run_id,
                members=members,
                interventions=interventions,
                unmatched_counts=unmatched_counts,
            )
            matched_total = interventions_total - unmatched_total
            payload = _build_summary_payload(
                run_id=args.run_id,
                sessions=sessions,
                interventions_total=interventions_total,
                matched_total=matched_total,
                unmatched_total=unmatched_total,
            )
            _store_summary_in_db(conn, payload)
        summary_path = _write_summary_file(args.run_id, payload)
    except (ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        "Analyzer finished successfully: "
        f"{payload['sessions_count']} session(s), "
        f"{payload['interventions_total']} intervention(s), "
        f"{payload['matched_interventions_total']} matched, "
        f"{payload['unmatched_interventions_total']} unmatched. "
        f"Summary file: {summary_path}; summary DB table: run_outputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
