#!/usr/bin/env python3
"""
Analyzer entrypoint used by run_pipeline.py.

Current slice:
- load selected stenograms
- normalize speakers
- resolve speakers to members using registry snapshots
- persist raw interventions + unmatched speakers
- build session-scoped RAG vector index (sentence-transformers + FAISS)
- retrieve evidence chunks via hybrid strategy (session_notes + neighbors + similarity)
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from init_db import DEFAULT_DB_PATH, init_db
from law_extractor import (
    build_session_law_index,
    extract_law_references,
    SessionLawIndex,
)
import rag_store


@dataclass(frozen=True)
class SessionStats:
    path: str
    session_id: str
    stenograma_date: str
    speeches_count: int


TOPIC_TAXONOMY: list[tuple[str, tuple[str, ...]]] = [
    ("costul vietii si inflatie", ("inflatie", "preturi", "scumpiri", "costul vietii")),
    ("echitate si simplitate fiscala", ("tax", "impozit", "fiscalitate", "tva")),
    ("eficienta cheltuielilor publice", ("cheltuieli publice", "eficienta bugetara", "risipa")),
    ("transparenta si responsabilitate bugetara", ("transparenta bugetara", "executie bugetara")),
    ("sustenabilitate fiscala", ("deficit", "datorie publica", "sustenabilitate fiscala")),
    ("sustenabilitatea pensiilor", ("pensie", "pensii", "sistem de pensii")),
    ("salarii si productivitate", ("salariu", "productivitate", "venituri salariale")),
    ("calitatea ocuparii si protectia muncii", ("contract de munca", "protectia muncii", "somaj")),
    ("formare profesionala si recalificare", ("formare profesionala", "recalificare", "ucenicie")),
    ("ocuparea tinerilor si primul loc de munca", ("tineri", "primul loc de munca", "internship")),
    ("educatie timpurie", ("educatie timpurie", "gradinita", "crese")),
    ("relevanta curriculumului scolar", ("curriculum", "programa scolara", "manuale")),
    ("recrutarea si retentia profesorilor", ("cadre didactice", "profesori", "cariera didactica")),
    ("infrastructura si siguranta scolara", ("infrastructura scolara", "siguranta in scoli")),
    ("competente digitale in educatie", ("competente digitale", "digitalizare in educatie")),
    ("calitate universitara si cercetare", ("universitate", "cercetare", "doctorat")),
    ("reducerea abandonului scolar", ("abandon scolar", "parasire timpurie")),
    ("acces la educatie rurala", ("educatie rurala", "transport scolar", "scoli rurale")),
    ("invatare pe tot parcursul vietii", ("invatare continua", "educatia adultilor")),
    ("sanatatea mintala a elevilor", ("consiliere scolara", "sanatate mintala elevi")),
    ("asistenta medicala primara", ("medicina de familie", "asistenta primara")),
    ("reforma managementului spitalicesc", ("spital", "management spitalicesc")),
    ("preventie si screening", ("preventie", "screening", "vaccinare")),
    ("acces si accesibilitate la medicamente", ("medicamente", "compensate", "farmacii")),
    ("retentia personalului medical", ("medici", "asistenti medicali", "exod medical")),
    ("pregatirea sistemului de urgenta", ("urgenta", "smurd", "upu")),
    ("servicii de sanatate mintala", ("psihiatrie", "psiholog", "sanatate mintala")),
    ("interoperabilitatea datelor medicale", ("dosar electronic", "interoperabilitate", "date medicale")),
    ("ingrijire pe termen lung si imbatranire", ("ingrijire pe termen lung", "varstnici")),
    ("sanatatea mamei si copilului", ("maternitate", "pediatrie", "sanatate materna")),
    ("securitate energetica si rezilienta retelei", ("securitate energetica", "sistem energetic", "retea electrica")),
    ("accesibilitatea energiei", ("facturi energie", "pret energie", "compensare energie")),
    ("tranzitie catre energie regenerabila", ("energie regenerabila", "eolian", "fotovoltaic")),
    ("competitivitate industriala si costuri energetice", ("costuri energetice industrie", "competitivitate industriala")),
    ("infrastructura de apa si rezilienta la seceta", ("apa", "canalizare", "seceta")),
    ("gestionarea deseurilor si economie circulara", ("deseuri", "reciclare", "economie circulara")),
    ("calitatea aerului urban", ("calitatea aerului", "poluare urbana")),
    ("adaptare climatica si pregatire pentru dezastre", ("adaptare climatica", "dezastre", "inundatii")),
    ("protectia padurilor si biodiversitate", ("paduri", "defrisari", "biodiversitate")),
    ("agricultura sustenabila si securitate alimentara", ("agricultura", "fermieri", "securitate alimentara")),
    ("mentenanta infrastructurii de transport", ("drumuri", "intretinere infrastructura", "autostrazi")),
    ("modernizare feroviara si capacitate de marfa", ("cale ferata", "feroviar", "transport marfa")),
    ("siguranta rutiera", ("siguranta rutiera", "accidente rutiere")),
    ("mobilitate urbana si transport public", ("transport public", "mobilitate urbana", "metrou")),
    ("accesibilitatea locuirii si urbanism", ("locuinte", "chirii", "urbanism")),
    ("stat de drept si eficienta justitiei", ("stat de drept", "justitie", "instante")),
    ("aplicarea legislatiei anticoruptie", ("coruptie", "dna", "integritate")),
    ("integritate in achizitii publice", ("achizitii publice", "licitatii", "seap")),
    ("simplificare administrativa si guvernare digitala", ("debirocratizare", "digitalizare", "ghiseu unic")),
    ("securitate nationala si infrastructura critica", ("securitate nationala", "infrastructura critica", "aparare")),
    ("procedura parlamentara si agenda", ("ordine de zi", "regulament", "procedura de vot", "motiune", "cenzura")),
    ("cadru constitutional", ("constitutie", "constitutional", "curtea constitutionala")),
    ("proces legislativ", ("proiect", "lege", "amendament", "ordonanta", "comisie", "articol")),
    ("dezbatere privind suspendarea presedintelui", ("suspend", "iohannis", "presedinte ilegitim")),
]

LAW_REFERENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpl[-\s]?x\s*(?:nr\.?\s*)?(\d+/\d{4})\b"), "plx {ref}"),
    (re.compile(r"\blegea?\s+nr\.?\s*(\d+/\d{4})\b"), "legea {ref}"),
    (re.compile(r"\boug\s+nr\.?\s*(\d+/\d{4})\b"), "oug {ref}"),
    (re.compile(r"\bog\s+nr\.?\s*(\d+/\d{4})\b"), "og {ref}"),
    (re.compile(r"\bhg\s+nr\.?\s*(\d+/\d{4})\b"), "hg {ref}"),
    (re.compile(r"\bordonanta(?:\s+de\s+urgenta)?\s+nr\.?\s*(\d+/\d{4})\b"), "ordonanta {ref}"),
]

PROCEDURAL_KEYWORDS = {
    "ordine de zi",
    "supun votului",
    "vot",
    "cvorum",
    "microfon",
    "cartele",
    "sedinta",
    "va rog",
    "declar deschisa",
    "regulament",
}

CONSTRUCTIVE_KEYWORDS = {
    "motiune",
    "cenzura",
    "proiect",
    "lege",
    "amendament",
    "articol",
    "ordonanta",
    "raport",
    "comisie",
    "buget",
}

NON_CONSTRUCTIVE_KEYWORDS = {
    "klaus iohannis",
    "presedinte ilegitim",
    "uzurpa functia",
    "impostor",
    "camarila",
    "acolitii",
    "dictatura",
    "tradare",
    "tradator",
    "ruinat tara",
}

RETRIEVAL_STOPWORDS = {
    "si",
    "sau",
    "in",
    "la",
    "cu",
    "de",
    "din",
    "pe",
    "ca",
    "este",
    "sunt",
    "fi",
    "fost",
    "va",
    "vot",
    "rog",
    "domnul",
    "doamna",
    "stima",
    "colegi",
    "sedinta",
    "plen",
}

SESSION_TOPIC_EARLY_SPEECHES_LIMIT = 30


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


def _extract_non_negative_int(row: dict, keys: tuple[str, ...]) -> int:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        return max(0, parsed)
    return 0


def _merge_speech_text(speech: dict) -> str:
    parts = []
    for key in ("text", "text2", "text3"):
        value = speech.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


def _analysis_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold()
    normalized = " ".join(normalized.split())
    return normalized


def _tokenize_for_retrieval(text: str) -> list[str]:
    normalized = _analysis_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    out = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in RETRIEVAL_STOPWORDS:
            continue
        out.append(tok)
    return out


def _build_session_chunks(
    session_id: str,
    run_id: str,
    stenogram_path: str,
    initial_notes: str,
    speeches: list[dict],
) -> list[dict]:
    chunks: list[dict] = []
    chunk_index = 0
    if initial_notes.strip():
        tokens = _tokenize_for_retrieval(initial_notes)
        chunks.append(
            {
                "chunk_id": f"ch:{session_id}:{chunk_index}",
                "run_id": run_id,
                "session_id": session_id,
                "stenogram_path": stenogram_path,
                "chunk_type": "session_notes",
                "chunk_index": chunk_index,
                "source_speech_index": None,
                "text": initial_notes.strip(),
                "tokens": tokens,
            }
        )
        chunk_index += 1

    for speech_idx, speech in enumerate(speeches):
        if not isinstance(speech, dict):
            continue
        speech_text = _merge_speech_text(speech)
        if len(_analysis_text(speech_text)) < 80:
            continue
        tokens = _tokenize_for_retrieval(speech_text)
        chunks.append(
            {
                "chunk_id": f"ch:{session_id}:{chunk_index}",
                "run_id": run_id,
                "session_id": session_id,
                "stenogram_path": stenogram_path,
                "chunk_type": "speech",
                "chunk_index": chunk_index,
                "source_speech_index": speech_idx,
                "text": speech_text,
                "tokens": tokens,
            }
        )
        chunk_index += 1
    return chunks


def _retrieve_evidence_chunk_ids(intervention_text: str, session_chunks: list[dict], top_k: int = 3) -> list[str]:
    query_tokens = set(_tokenize_for_retrieval(intervention_text))
    if not session_chunks:
        return []
    if not query_tokens:
        return [session_chunks[0]["chunk_id"]]

    scored: list[tuple[float, str]] = []
    for chunk in session_chunks:
        chunk_tokens = set(chunk.get("tokens", []))
        if not chunk_tokens:
            score = 0.0
        else:
            inter = len(query_tokens.intersection(chunk_tokens))
            union = len(query_tokens.union(chunk_tokens))
            score = inter / union if union else 0.0
        scored.append((score, chunk["chunk_id"]))

    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
    top = [chunk_id for score, chunk_id in ranked if score > 0.0][:top_k]
    if top:
        return top
    return [session_chunks[0]["chunk_id"]]


def _extract_topics(text: str, max_topics: int = 5) -> list[str]:
    normalized = _analysis_text(text)
    topics: list[str] = []

    # 1) Enrich with explicit law/bill references via comprehensive extractor.
    for ref in extract_law_references(text):
        topic = ref.canonical_id.lower()
        if topic not in topics:
            topics.append(topic)
        if len(topics) >= max_topics:
            return topics

    # 2) Add enriched taxonomy topics.
    for topic, keywords in TOPIC_TAXONOMY:
        if any(keyword in normalized for keyword in keywords) and topic not in topics:
            topics.append(topic)
        if len(topics) >= max_topics:
            break
    return topics


def _is_law_reference_topic(topic: str) -> bool:
    return topic.startswith(("plx ", "legea ", "oug ", "og ", "hg ", "ordonanta "))


def _topic_sort_key(item: tuple[str, float]) -> tuple[int, float, str]:
    topic, score = item
    law_priority = 1 if _is_law_reference_topic(topic) else 0
    return (law_priority, score, topic)


def _deterministic_analysis(
    text: str,
    intervention_topics: list[str],
    session_topics: list[str],
) -> tuple[str, list[str], float]:
    normalized = _analysis_text(text)
    session_topic_set = set(session_topics)
    matched_topics = [topic for topic in intervention_topics if topic in session_topic_set]

    if len(normalized) < 40:
        return "neutral", matched_topics, 0.45

    # Roll-call / attendance blocks are procedural.
    roll_call_hits = sum(1 for token in (" absent", " prezent", "nu votez") if token in normalized)
    if roll_call_hits >= 2:
        return "neutral", matched_topics, 0.90

    # "proiectul ordinii de zi" is procedural, not substantive legislation.
    adjusted_for_relevance = normalized
    adjusted_for_relevance = adjusted_for_relevance.replace("proiectul ordinii de zi", "ordinii de zi")
    adjusted_for_relevance = adjusted_for_relevance.replace("proiect de ordine de zi", "ordine de zi")

    non_constructive_hits = sum(1 for k in NON_CONSTRUCTIVE_KEYWORDS if k in normalized)
    constructive_hits = sum(1 for k in CONSTRUCTIVE_KEYWORDS if k in adjusted_for_relevance)
    procedural_hits = sum(1 for k in PROCEDURAL_KEYWORDS if k in normalized)
    has_topic_overlap = len(matched_topics) > 0

    # Purely rhetorical attacks with no substantive engagement.
    if non_constructive_hits > 0 and not has_topic_overlap and constructive_hits == 0 and procedural_hits <= 1:
        return "non_constructive", [], 0.80

    # No overlap means likely neutral/procedural in this baseline.
    if not has_topic_overlap:
        if procedural_hits > 0:
            return "neutral", [], 0.72
        return "neutral", [], 0.58

    # Overlap exists: evaluate likely substantive vs procedural.
    if constructive_hits > 0:
        strong_constructive_tokens = ("motiune", "cenzura", "ordonanta", "amendament", "articol")
        has_strong_constructive = any(t in adjusted_for_relevance for t in strong_constructive_tokens)
        if procedural_hits > 0 and not has_strong_constructive:
            return "neutral", matched_topics, 0.63
        return "constructive", matched_topics, 0.75

    # Purely procedural but on session topics.
    return "neutral", matched_topics, 0.67


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
                "bills_authored_total": _extract_non_negative_int(
                    row,
                    (
                        "bills_authored_total",
                        "bills_authored",
                        "total_bills_authored",
                        "initiated_bills_count",
                        "proiecte_initiate_total",
                        "proiecte_initiate",
                    ),
                ),
                "amendments_added_total": _extract_non_negative_int(
                    row,
                    (
                        "amendments_added_total",
                        "amendments_added",
                        "total_amendments_added",
                        "amendments_count",
                        "proposed_amendments_count",
                        "amendamente_adaugate_total",
                        "amendamente_adaugate",
                    ),
                ),
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
    session_chunks: list[dict],
    interventions: list[dict],
    unmatched_counts: dict[tuple[str, str, str, str], int],
    session_topics_map: dict[str, list[str]],
    session_to_stenogram_path: dict[str, str],
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
                party_id, bills_authored_total, amendments_added_total, profile_url, circumscriptie, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(member_id) DO UPDATE SET
                source_member_id = excluded.source_member_id,
                chamber = excluded.chamber,
                name = excluded.name,
                normalized_name = excluded.normalized_name,
                party_id = excluded.party_id,
                bills_authored_total = excluded.bills_authored_total,
                amendments_added_total = excluded.amendments_added_total,
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
                member["bills_authored_total"],
                member["amendments_added_total"],
                member["profile_url"],
                member["circumscriptie"],
            ),
        )

    for chunk in session_chunks:
        conn.execute(
            """
            INSERT INTO session_chunks (
                chunk_id, run_id, session_id, stenogram_path, chunk_type, chunk_index,
                source_speech_index, text, tokens_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                run_id = excluded.run_id,
                session_id = excluded.session_id,
                stenogram_path = excluded.stenogram_path,
                chunk_type = excluded.chunk_type,
                chunk_index = excluded.chunk_index,
                source_speech_index = excluded.source_speech_index,
                text = excluded.text,
                tokens_json = excluded.tokens_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                chunk["chunk_id"],
                chunk["run_id"],
                chunk["session_id"],
                chunk["stenogram_path"],
                chunk["chunk_type"],
                chunk["chunk_index"],
                chunk["source_speech_index"],
                chunk["text"],
                json.dumps(chunk["tokens"], ensure_ascii=True),
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

        # Deterministic baseline analysis (pre-RAG/LLM).
        if iv["member_id"] is not None:
            constructiveness_label, topics, confidence = _deterministic_analysis(
                iv["text"],
                iv.get("candidate_topics", []),
                session_topics_map.get(iv["session_id"], []),
            )
            conn.execute(
                """
                INSERT INTO intervention_analysis (
                    intervention_id,
                    run_id,
                    relevance_label,
                    relevance_source,
                    topics_json,
                    confidence,
                    evidence_chunk_ids_json,
                    analysis_version,
                    updated_at
                )
                VALUES (?, ?, ?, 'constructiveness_baseline_v1', ?, ?, ?, 'baseline_v1', CURRENT_TIMESTAMP)
                ON CONFLICT(intervention_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    relevance_label = excluded.relevance_label,
                    relevance_source = excluded.relevance_source,
                    topics_json = excluded.topics_json,
                    confidence = excluded.confidence,
                    evidence_chunk_ids_json = excluded.evidence_chunk_ids_json,
                    analysis_version = excluded.analysis_version,
                    updated_at = CURRENT_TIMESTAMP
                WHERE intervention_analysis.relevance_source = 'constructiveness_baseline_v1'
                """,
                (
                    iv["intervention_id"],
                    run_id,
                    constructiveness_label,
                    json.dumps(topics, ensure_ascii=True),
                    confidence,
                    json.dumps(iv.get("evidence_chunk_ids", []), ensure_ascii=True),
                ),
            )

    for session_id, topics in session_topics_map.items():
        conn.execute(
            """
            INSERT INTO session_topics (
                session_id, run_id, stenogram_path, topics_json, updated_at
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                run_id = excluded.run_id,
                stenogram_path = excluded.stenogram_path,
                topics_json = excluded.topics_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                session_id,
                run_id,
                session_to_stenogram_path.get(session_id, ""),
                json.dumps(topics, ensure_ascii=True),
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
    session_topic_counters: dict[str, Counter[str]],
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
                "top_topics": [
                    {"topic": topic, "count": count}
                    for topic, count in sorted(
                        session_topic_counters.get(s.session_id, Counter()).items(),
                        key=lambda x: _topic_sort_key((x[0], x[1])),
                        reverse=True,
                    )[:10]
                ],
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
        session_chunks: list[dict] = []
        session_chunks_by_session: dict[str, list[dict]] = {}
        interventions: list[dict] = []
        unmatched_counts: dict[tuple[str, str, str, str], int] = {}
        session_topic_counters: dict[str, Counter[str]] = defaultdict(Counter)
        session_topic_scores: dict[str, dict[str, float]] = defaultdict(dict)
        session_to_stenogram_path: dict[str, str] = {}

        n_selected = len(selected)
        print(f"Baseline: processing {n_selected} stenogram(s)...")

        for file_idx, rel_path in enumerate(selected, 1):
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
            session_to_stenogram_path[session_id] = stenogram_path
            initial_notes = str(data.get("initial_notes", "")).strip()

            print(
                f"\n[{file_idx}/{n_selected}] {file_path.name}"
                f"  session={session_id}  date={session_date}"
                f"  speeches={len(speeches)}"
            )

            built_chunks = _build_session_chunks(
                session_id=session_id,
                run_id=args.run_id,
                stenogram_path=stenogram_path,
                initial_notes=initial_notes,
                speeches=speeches,
            )
            session_chunks.extend(built_chunks)
            session_chunks_by_session[session_id] = built_chunks

            print(f"  Building RAG index: {len(built_chunks)} chunk(s)...")
            rag_store.build_session_index(session_id, built_chunks)
            print(f"  RAG index ready.")

            if initial_notes:
                for topic in _extract_topics(initial_notes, max_topics=12):
                    session_topic_counters[session_id][topic] += 1
                    session_topic_scores[session_id][topic] = session_topic_scores[session_id].get(topic, 0.0) + 4.0

            session_matched = 0
            session_unmatched = 0

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
                extracted_topics = _extract_topics(text)
                is_substantial = len(_analysis_text(text)) >= 160
                if idx < SESSION_TOPIC_EARLY_SPEECHES_LIMIT and is_substantial:
                    for topic in extracted_topics:
                        session_topic_counters[session_id][topic] += 1
                        # Session topics must be independent of each queried intervention;
                        # only early substantial speeches shape the session context.
                        session_topic_scores[session_id][topic] = session_topic_scores[session_id].get(topic, 0.0) + 2.5

                intervention_id = _build_intervention_id(stenogram_path, idx)
                retrieved = rag_store.retrieve_chunks(
                    session_id=session_id,
                    intervention_text=text,
                    intervention_speech_index=idx,
                    top_k=rag_store.DEFAULT_TOP_K,
                )
                evidence_chunk_ids = [r.chunk_id for r in retrieved]
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

                if member_id:
                    session_matched += 1
                else:
                    session_unmatched += 1

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
                        "candidate_topics": extracted_topics,
                        "evidence_chunk_ids": evidence_chunk_ids,
                    }
                )
                if member_id is None and normalized_speaker:
                    key = (session_id, stenogram_path, raw_speaker, normalized_speaker)
                    unmatched_counts[key] = unmatched_counts.get(key, 0) + 1

            print(
                f"  Done: {len(speeches)} speeches,"
                f" {session_matched} matched, {session_unmatched} unmatched."
            )

        # Build per-session law reference index and persist as JSON artifact.
        session_law_indices: dict[str, SessionLawIndex] = {}
        for session_id, stenogram_path in session_to_stenogram_path.items():
            file_path = Path(stenogram_path)
            if not file_path.exists():
                continue
            data = _read_json(file_path)
            initial_notes = str(data.get("initial_notes", "")).strip()
            speeches = data.get("speeches", [])
            law_idx = build_session_law_index(session_id, initial_notes, speeches)
            session_law_indices[session_id] = law_idx
            if law_idx.all_law_ids:
                print(f"  Session {session_id}: {len(law_idx.all_law_ids)} law reference(s) extracted")
                for lid in law_idx.all_law_ids:
                    speech_idxs = law_idx.law_to_speeches.get(lid, [])
                    print(f"    - {lid} (speeches: {speech_idxs})")

        # Persist law indices as JSON for downstream LLM use.
        law_index_dir = Path("state/law_indices")
        law_index_dir.mkdir(parents=True, exist_ok=True)
        for session_id, law_idx in session_law_indices.items():
            law_index_path = law_index_dir / f"{session_id}_law_index.json"
            law_index_path.write_text(
                json.dumps({
                    "session_id": session_id,
                    "law_to_speeches": law_idx.law_to_speeches,
                    "speech_to_laws": law_idx.speech_to_laws,
                    "all_law_ids": law_idx.all_law_ids,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        session_topics_map: dict[str, list[str]] = {}
        for session_id, counter in session_topic_counters.items():
            scored = []
            for topic, count in counter.items():
                score = session_topic_scores[session_id].get(topic, 0.0) + (count * 0.1)
                scored.append((topic, score))
            top = [topic for topic, _score in sorted(scored, key=_topic_sort_key, reverse=True)[:20]]
            session_topics_map[session_id] = top

        print(f"\nSaving to DB: {len(members)} members, {len(session_chunks)} chunks, {len(interventions)} interventions...")
        with sqlite3.connect(db_path) as conn:
            interventions_total, unmatched_total = _persist_run_data(
                conn=conn,
                run_id=args.run_id,
                members=members,
                session_chunks=session_chunks,
                interventions=interventions,
                unmatched_counts=unmatched_counts,
                session_topics_map=session_topics_map,
                session_to_stenogram_path=session_to_stenogram_path,
            )
            matched_total = interventions_total - unmatched_total
            print(
                f"  DB saved: {interventions_total} interventions total, "
                f"{matched_total} matched, {unmatched_total} unmatched."
            )
            payload = _build_summary_payload(
                run_id=args.run_id,
                sessions=sessions,
                interventions_total=interventions_total,
                matched_total=matched_total,
                unmatched_total=unmatched_total,
                session_topic_counters=session_topic_counters,
            )
            _store_summary_in_db(conn, payload)
            print(f"  Run summary saved to DB table: run_outputs")
        summary_path = _write_summary_file(args.run_id, payload)
        print(f"  Run summary written to: {summary_path}")
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
