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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from init_db import DEFAULT_DB_PATH, init_db


DEFAULT_TAXONOMY_CONFIG_PATH = Path("config/topic_taxonomy.json")


def _map_label(label: str) -> str:
    if label in {"constructive", "neutral", "non_constructive"}:
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


def _top_directions(counter: Counter[str], limit: int = 20) -> list[dict]:
    filtered = Counter({k: v for k, v in counter.items() if k != "altele"})
    ranked = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))
    return [{"topic": topic, "count": count} for topic, count in ranked[:limit]]


_TOPIC_STOPWORDS = {
    "si", "sau", "de", "din", "la", "cu", "pentru", "in", "pe", "ale", "al", "ai", "a",
    "un", "o", "unei", "unui", "lui", "lor", "prin", "privind", "despre", "asupra",
    "modificarea", "modificari", "lege", "legii", "proiect", "proiectul", "propunere",
    "legislativa", "parlament", "parlamentara", "romania", "roman", "national",
}

_TOKEN_EQUIVALENTS = {
    "fiscal": "fiscalitate",
    "fiscala": "fiscalitate",
    "fiscale": "fiscalitate",
    "taxe": "taxe",
    "taxa": "taxe",
    "taxelor": "taxe",
    "impozit": "impozite",
    "impozite": "impozite",
    "impozitelor": "impozite",
    "educatie": "educatie",
    "invatamant": "educatie",
    "scoala": "educatie",
    "scoli": "educatie",
    "elevi": "educatie",
    "studenti": "educatie",
    "sanatate": "sanatate",
    "spital": "sanatate",
    "spitale": "sanatate",
    "medicamente": "sanatate",
    "drum": "infrastructura",
    "drumuri": "infrastructura",
    "autostrada": "infrastructura",
    "autostrazi": "infrastructura",
    "feroviar": "infrastructura",
    "transport": "infrastructura",
    "energie": "energie",
    "energetic": "energie",
    "energetica": "energie",
    "agricultura": "agricultura",
    "agricol": "agricultura",
    "agricole": "agricultura",
    "fermieri": "agricultura",
    "pensie": "pensii",
    "pensii": "pensii",
    "pensiilor": "pensii",
    "militar": "aparare",
    "militare": "aparare",
    "aparare": "aparare",
    "securitate": "securitate",
    "siguranta": "securitate",
    "sigurantei": "securitate",
    "suveranism": "suveranitate",
    "suveranist": "suveranitate",
    "suveranista": "suveranitate",
    "suveraniste": "suveranitate",
    "constitutie": "constitutional",
    "constitutional": "constitutional",
    "constitutionala": "constitutional",
    "neconstitutionalitate": "constitutional",
    "ccr": "constitutional",
    "procedura": "procedural",
    "procedurala": "procedural",
    "vot": "procedural",
    "sedinta": "procedural",
    "plen": "procedural",
    "ordine": "procedural",
    "agenda": "procedural",
    "rusia": "rusia_ucraina",
    "ucraina": "rusia_ucraina",
    "nato": "securitate_externa",
    "csat": "securitate_externa",
    "apararii": "aparare",
    "drone": "aparare",
    "buget": "buget",
    "bugetar": "buget",
    "bugetara": "buget",
    "bugetului": "buget",
}

_DIRECTION_RULES: list[tuple[str, set[str], set[str]]] = [
    (
        "politica externa si securitate",
        {"rusia_ucraina", "securitate_externa", "geopolitica", "aparare", "securitate", "suveranitate"},
        {
            "rusia", "ucraina", "nato", "razboi", "geopolitica", "aparare", "securitate", "suveranitate",
            "csat", "uniunea europeana", "ue", "sua", "diaspora", "sanctiuni", "pace",
            "politica externa", "grupuri de prietenie", "paza obiective",
        },
    ),
    (
        "procedura parlamentara",
        {"procedural"},
        {
            "procedura", "vot", "ordine", "sedinta", "plen", "cvorum", "respingere", "regulament",
            "ora prim-ministrului", "retrimitere la comisie", "lucru in paralel", "amendament",
            "amendamente", "comisie",
        },
    ),
    (
        "economie si fiscalitate",
        {"fiscalitate", "taxe", "impozite", "buget", "investitii", "pnrr", "fonduri"},
        {
            "fiscal", "tax", "impoz", "buget", "austeritate", "investitii", "pnrr", "fonduri",
            "finante", "cheltuieli", "datorie publica", "privatizare", "imm", "multinationale",
            "salariu minim", "saracie", "industrie", "servicii publice", "tva",
            "pensii", "pensiile speciale", "pensii de serviciu", "reconversie profesionala",
            "valea jiului", "minerit",
        },
    ),
    (
        "energie si mediu",
        {"energie"},
        {
            "energie", "facturi energie", "certificat verde", "certificat emisii", "piata energiei",
            "pret energie", "energie termica", "energie regenerabila", "carbon", "apele romane",
            "salrom", "resurse nationale", "certificate verzi", "hidroelectrica",
        },
    ),
    (
        "infrastructura si transport",
        {"infrastructura"},
        {"drum", "autostr", "transport", "feroviar", "aeroport", "port", "control spatiu aerian", "tulcea", "risc seismic"},
    ),
    (
        "agricultura si dezvoltare rurala",
        {"agricultura"},
        {"agricultura", "fermier", "irig", "rural", "fond cinegetic", "ferme de familie", "despagubiri"},
    ),
    (
        "sanatate",
        {"sanatate"},
        {"sanatate", "spital", "medicament", "medical", "reforma sanatatii", "salarizare medici", "profesia medic"},
    ),
    (
        "educatie",
        {"educatie"},
        {"educatie", "invatamant", "scoala", "elev", "student", "univers", "burse", "curriculum"},
    ),
    (
        "justitie si constitutional",
        {"constitutional"},
        {
            "constitut", "constitutional", "ccr", "justit", "stat de drept", "prescriptie penala",
            "coruptie", "drepturi fundamentale", "abuzuri", "retrocedari", "proprietate privata",
            "retrocedari ilegale", "cod civil", "retrocedare", "contraventii", "csm", "diicot",
            "anpc", "antifrauda", "cna",
        },
    ),
    (
        "digitalizare si tehnologie",
        {"digitalizare"},
        {"digital", "cibern", "ai", "tehnolog", "dezinformare", "propaganda", "inteligenta artificiala", "eurohpc"},
    ),
    (
        "media si comunicare publica",
        {"tvr", "radiodifuziune"},
        {"tvr", "radiodifuziune", "libertatea de exprimare"},
    ),
    (
        "administratie publica si guvernare",
        {"guvern", "transparenta"},
        {
            "guvern", "minister", "administr", "transparent", "numiri politice", "finantare electorala",
            "demisie presedinte", "alegeri prezidentiale", "critica presedinte", "vicepremieri",
            "curtea de conturi", "ans", "numire presedinte ca", "proiect de tara",
            "transfer imobil", "transfer proprietate", "transfer autoritate", "lege organica",
            "claritate legislativa", "responsabilitate institutionala", "proprietate", "critica politica",
            "comisia europeana", "usr", "prelungire program",
        },
    ),
    (
        "cultura si patrimoniu",
        {"patrimoniu"},
        {"cultura", "patrimoniu", "brancusi", "muze", "festival", "salina", "valori nationale", "identitate nationala"},
    ),
    (
        "social si drepturi civile",
        {"minoritati"},
        {
            "minoritati", "violenta domestica", "democratie", "extremism", "unitate nationala",
            "romani diaspora", "pasapoarte", "permise auto", "masuri represive", "putin", "federatia rusa",
            "basarabia", "holodomor", "istoria evreilor", "corneliu vadim tudor",
            "cumintenia pamantului",
        },
    ),
    (
        "proces legislativ si reforme",
        set(),
        {"pl-x", "oug ", "og ", "hg ", "legea nr", "phcd", "cod fiscal", "reglementare"},
    ),
    (
        "justitie si constitutional",
        set(),
        {"ejtn"},
    ),
]


def _load_taxonomy_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _apply_taxonomy_config(path: Path) -> bool:
    data = _load_taxonomy_config(path)
    if data is None:
        return False

    stopwords_raw = data.get("topic_stopwords", [])
    equivalents_raw = data.get("token_equivalents", {})
    rules_raw = data.get("direction_rules", [])

    if not isinstance(stopwords_raw, list) or not isinstance(equivalents_raw, dict) or not isinstance(rules_raw, list):
        return False

    stopwords: set[str] = set()
    for item in stopwords_raw:
        if isinstance(item, str) and item.strip():
            stopwords.add(item.strip())

    equivalents: dict[str, str] = {}
    for k, v in equivalents_raw.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            equivalents[k.strip()] = v.strip()

    rules: list[tuple[str, set[str], set[str]]] = []
    for item in rules_raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        mapped_tokens = item.get("mapped_tokens", [])
        raw_roots = item.get("raw_roots", [])
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(mapped_tokens, list) or not isinstance(raw_roots, list):
            continue
        mapped_set = {str(x).strip() for x in mapped_tokens if isinstance(x, str) and str(x).strip()}
        roots_set = {str(x).strip() for x in raw_roots if isinstance(x, str) and str(x).strip()}
        rules.append((label.strip(), mapped_set, roots_set))

    if not stopwords or not equivalents or not rules:
        return False

    global _TOPIC_STOPWORDS
    global _TOKEN_EQUIVALENTS
    global _DIRECTION_RULES
    _TOPIC_STOPWORDS = stopwords
    _TOKEN_EQUIVALENTS = equivalents
    _DIRECTION_RULES = rules
    return True


def _normalize_topic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    return re.sub(r"\s+", " ", normalized)


def _extract_law_id(text: str) -> str | None:
    patterns = [
        r"\bpl[-\s]?x\s*(\d+/\d{4})\b",
        r"\boug\s*(\d+/\d{4})\b",
        r"\bog\s*(\d+/\d{4})\b",
        r"\bhg\s*(\d+/\d{4})\b",
        r"\blegea?\s*nr\.?\s*(\d+/\d{4})\b",
        r"\blegea?\s*(\d+/\d{4})\b",
        r"\bphcd\s*(\d+/\d{4})\b",
    ]
    labels = ["PL-x", "OUG", "OG", "HG", "Legea nr.", "Legea", "PHCD"]
    normalized = _normalize_topic_text(text)
    for label, pattern in zip(labels, patterns):
        m = re.search(pattern, normalized, flags=re.IGNORECASE)
        if m:
            return f"{label} {m.group(1)}"
    return None


def _tokenize_topic(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", _normalize_topic_text(text))
    out: list[str] = []
    for tok in raw_tokens:
        mapped = _TOKEN_EQUIVALENTS.get(tok, tok)
        if len(mapped) < 3:
            continue
        if mapped in _TOPIC_STOPWORDS:
            continue
        out.append(mapped)
    return out


def _topic_key(text: str) -> str:
    law_id = _extract_law_id(text)
    if law_id:
        return f"law:{law_id.lower()}"
    tokens = sorted(set(_tokenize_topic(text)))
    if tokens:
        return "kw:" + "|".join(tokens[:6])
    return "raw:" + _normalize_topic_text(text)


def _topic_direction(topic: str) -> str:
    if _extract_law_id(topic):
        return "proces legislativ si reforme"
    normalized = _normalize_topic_text(topic)
    tokens = set(_tokenize_topic(topic))
    for label, mapped_tokens, raw_roots in _DIRECTION_RULES:
        if tokens.intersection(mapped_tokens):
            return label
        if any(root in normalized for root in raw_roots):
            return label
    return "altele"


_LAW_ID_PATTERNS: list[tuple[str, str]] = [
    ("PL-x", r"\bpl[-\s]?x\s*(\d{1,4})(?:/(\d{4}))?\b"),
    ("OUG", r"\boug\s*(\d{1,4})(?:/(\d{4}))?\b"),
    ("OG", r"\bog\s*(\d{1,4})(?:/(\d{4}))?\b"),
    ("HG", r"\bhg\s*(\d{1,4})(?:/(\d{4}))?\b"),
    ("Legea nr.", r"\blegea?\s*nr\.?\s*(\d{1,4})(?:/(\d{4}))?\b"),
]

_AUTHORSHIP_CONTEXT_PATTERNS = [
    re.compile(r"\bam initiat\b"),
    re.compile(r"\bam depus\b"),
    re.compile(r"\bam semnat\b"),
    re.compile(r"\bsunt(?:em)?\s+(?:co)?initiator(?:ii|i)?\b"),
    re.compile(r"\b(?:co)?initiator(?:ii|i)?\s+sunt(?:em)?\b"),
    re.compile(r"\binitiativa\s+(?:mea|noastra)\b"),
    re.compile(r"\bpropunerea\s+(?:mea|noastra)\s+legislativa\b"),
    re.compile(r"\bproiectul\s+(?:meu|nostru)\s+de\s+lege\b"),
]

_AMENDMENT_CONTEXT_PATTERNS = [
    re.compile(r"\bam depus\s+amendament"),
    re.compile(r"\bam depus\s+amendamente"),
    re.compile(r"\bam propus\s+amendament"),
    re.compile(r"\bam propus\s+amendamente"),
    re.compile(r"\bam introdus\s+amendament"),
    re.compile(r"\bam formulat\s+amendament"),
    re.compile(r"\bamendamentul\s+(?:meu|nostru)\b"),
    re.compile(r"\bamendamentele\s+(?:mele|noastre)\b"),
]

_AMENDMENT_NUMBER_PATTERN = re.compile(r"\bamendament(?:ul|ele)?(?:\s+nr\.?)?\s*(\d{1,5}(?:/\d{2,4})?)\b")


def _extract_all_law_ids(text: str) -> set[str]:
    normalized = _normalize_topic_text(text)
    out: set[str] = set()
    for label, pattern in _LAW_ID_PATTERNS:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            number = str(match.group(1)).strip()
            year = str(match.group(2)).strip() if match.group(2) else ""
            if not number:
                continue
            identifier = f"{number}/{year}" if year else number
            out.add(f"{label} {identifier}")
    return out


def _extract_legislation_contributions(text: str) -> tuple[set[str], set[str], int, int]:
    """
    Returns:
    - authored_bill_ids (deduplicated by identifier)
    - amendment_ids (deduplicated by amendment number + optional bill id)
    - generic_authored_bill_events (fallback events without explicit bill id)
    - generic_amendment_events (fallback events without explicit amendment id)
    """
    normalized = _normalize_topic_text(text)
    law_ids = _extract_all_law_ids(normalized)

    has_authorship_context = any(p.search(normalized) for p in _AUTHORSHIP_CONTEXT_PATTERNS)
    has_amendment_context = any(p.search(normalized) for p in _AMENDMENT_CONTEXT_PATTERNS)

    authored_bill_ids: set[str] = set()
    generic_authored_bill_events = 0
    if has_authorship_context:
        authored_bill_ids.update(law_ids)
        has_generic_bill_reference = re.search(r"\b(proiect(?:ul)?\s+de\s+lege|propunere(?:a)?\s+legislativa)\b", normalized)
        if not authored_bill_ids and has_generic_bill_reference:
            generic_authored_bill_events = 1

    amendment_ids: set[str] = set()
    generic_amendment_events = 0
    if has_amendment_context:
        law_context = sorted(law_ids)[0] if law_ids else ""
        for match in _AMENDMENT_NUMBER_PATTERN.finditer(normalized):
            amendment_no = str(match.group(1)).strip()
            if not amendment_no:
                continue
            if law_context:
                amendment_ids.add(f"{law_context}|amendament {amendment_no}")
            else:
                amendment_ids.add(f"amendament {amendment_no}")
        if not amendment_ids:
            generic_amendment_events = 1

    return authored_bill_ids, amendment_ids, generic_authored_bill_events, generic_amendment_events


class TopicCanonicalizer:
    def __init__(self) -> None:
        self._key_counts: Counter[str] = Counter()
        self._key_alias_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._resolved: dict[str, str] = {}

    def add(self, topic: str, count: int = 1) -> None:
        if not isinstance(topic, str):
            return
        value = topic.strip()
        if not value:
            return
        key = _topic_key(value)
        self._key_counts[key] += count
        self._key_alias_counts[key][value] += count

    def resolve(self) -> None:
        resolved: dict[str, str] = {}
        for key in self._key_counts:
            if key.startswith("law:"):
                law_label = key.replace("law:", "", 1).strip()
                resolved[key] = law_label.upper().replace("NR.", "nr.")
                continue
            aliases = self._key_alias_counts[key]
            ranked = sorted(aliases.items(), key=lambda x: (-x[1], x[0].lower()))
            resolved[key] = ranked[0][0]
        self._resolved = resolved

    def canonical(self, topic: str) -> str:
        key = _topic_key(topic)
        if not self._resolved:
            self.resolve()
        return self._resolved.get(key, topic.strip())

    def top_topics(self, limit: int = 20) -> list[dict]:
        if not self._resolved:
            self.resolve()
        ranked = sorted(
            ((self._resolved.get(k, k), c) for k, c in self._key_counts.items()),
            key=lambda x: (-x[1], x[0].lower()),
        )
        return [{"topic": topic, "count": count} for topic, count in ranked[:limit]]

    def top_topics_with_aliases(self, limit: int = 100, alias_limit: int = 5) -> list[dict]:
        if not self._resolved:
            self.resolve()
        ranked_keys = sorted(
            self._key_counts.items(),
            key=lambda x: (-x[1], self._resolved.get(x[0], x[0])),
        )
        out: list[dict] = []
        for key, count in ranked_keys[:limit]:
            canonical = self._resolved.get(key, key)
            aliases = sorted(
                self._key_alias_counts[key].items(),
                key=lambda x: (-x[1], x[0].lower()),
            )[:alias_limit]
            out.append(
                {
                    "topic": canonical,
                    "count": count,
                    "aliases": [{"topic": a, "count": c} for a, c in aliases],
                }
            )
        return out


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


def _export_session_topics(conn: sqlite3.Connection, topics_dir: Path, session_links: dict[str, str]) -> int:
    """Write one JSON file per session to outputs/session_topics/."""
    topics_dir.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        """
        SELECT st.session_id, st.topics_json, st.topics_source, st.updated_at,
               MIN(sc.stenogram_path) AS stenogram_path
        FROM session_topics st
        JOIN session_chunks sc ON sc.session_id = st.session_id
        GROUP BY st.session_id
        ORDER BY st.session_id
        """
    ).fetchall()
    written = 0
    for row in rows:
        session_id, topics_json_raw, topics_source, updated_at, stenogram_path = row
        try:
            topics = json.loads(topics_json_raw or "[]")
        except json.JSONDecodeError:
            topics = []
        stenogram_name = Path(stenogram_path).stem if stenogram_path else session_id
        out = {
            "session_id": session_id,
            "stenogram": stenogram_name,
            "source_url": session_links.get(str(session_id), ""),
            "topics_source": topics_source,
            "updated_at": updated_at,
            "topics": topics,
        }
        out_file = topics_dir / f"topics_for_{stenogram_name}.json"
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    return written


def export_outputs(db_path: Path, output_dir: Path, taxonomy_config_path: Path = DEFAULT_TAXONOMY_CONFIG_PATH) -> tuple[int, int]:
    loaded_taxonomy = _apply_taxonomy_config(taxonomy_config_path)
    print(
        f"Export taxonomy: {'loaded' if loaded_taxonomy else 'using built-in defaults'} "
        f"({taxonomy_config_path})"
    )
    init_db(db_path)
    print(f"Export: loading data from {db_path}...")
    session_links = _load_session_links()
    print(f"  Loaded {len(session_links)} session source links.")
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
                ia.confidence,
                ia.reasoning
            FROM interventions_raw iv
            JOIN members m ON m.member_id = iv.member_id
            LEFT JOIN intervention_analysis ia ON ia.intervention_id = iv.intervention_id
            WHERE iv.member_id IS NOT NULL
            ORDER BY iv.member_id, iv.session_date, iv.session_id, iv.speech_index
            """
        ).fetchall()

    print(f"  Loaded {len(rows)} intervention rows for {len({r[0] for r in rows})} member(s).")
    global_topic_model = TopicCanonicalizer()
    global_direction_counter: Counter[str] = Counter()
    for row in rows:
        topics = _safe_topics(row[8])
        for topic in topics:
            global_topic_model.add(topic)
            global_direction_counter[_topic_direction(topic)] += 1
    global_topic_model.resolve()

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
            constructiveness_label_raw,
            topics_json,
            confidence,
            reasoning,
        ) = row
        constructiveness_label = _map_label(constructiveness_label_raw)
        topics_raw = _safe_topics(topics_json)
        topics = [global_topic_model.canonical(t) for t in topics_raw]
        confidence_value = float(confidence) if confidence is not None else 0.0
        stenogram_name = Path(stenogram_path).name

        if member_id not in member_data:
            member_data[member_id] = {
                "member_id": member_id,
                "name": member_name,
                "party_id": party_id,
                "party_name": party_id,
                "authored_bill_ids": set(),
                "amendment_ids": set(),
                "generic_bills_authored_events": 0,
                "generic_amendments_added_events": 0,
                "counts": {"constructive": 0, "neutral": 0, "non_constructive": 0},
                "topics_counter": Counter(),
                "directions_counter": Counter(),
                "interventions": {"constructive": [], "neutral": [], "non_constructive": []},
            }

        md = member_data[member_id]
        authored_bill_ids, amendment_ids, generic_bill_events, generic_amendment_events = _extract_legislation_contributions(
            text or ""
        )
        md["authored_bill_ids"].update(authored_bill_ids)
        md["amendment_ids"].update(amendment_ids)
        md["generic_bills_authored_events"] += generic_bill_events
        md["generic_amendments_added_events"] += generic_amendment_events
        md["counts"][constructiveness_label] += 1
        md["topics_counter"].update(topics)
        md["directions_counter"].update(_topic_direction(t) for t in topics)
        md["interventions"][constructiveness_label].append(
            {
                "session_id": session_id,
                "session_date": session_date,
                "text": text or "",
                "topics": topics,
                "topics_raw": topics_raw,
                "confidence": confidence_value,
                "reasoning": reasoning or "",
                "stenogram_name": stenogram_name,
                "stenogram_link": session_links.get(str(session_id), ""),
            }
        )

    # Session topics — one file per stenogram, always overwrite.
    topics_dir = output_dir / "session_topics"
    print(f"  Exporting session topics to {topics_dir}...")
    with sqlite3.connect(db_path) as topics_conn:
        n_topics = _export_session_topics(topics_conn, topics_dir, session_links)
    print(f"  Written: {n_topics} session topic file(s) → {topics_dir}")

    # Global canonical topic stats across all interventions.
    global_topics_dir = output_dir / "topics"
    _clear_json_files(global_topics_dir)
    global_topics = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "all_interventions",
        "total_distinct_canonical_topics": len(global_topic_model.top_topics_with_aliases(limit=100000)),
        "top_directions": _top_directions(global_direction_counter, limit=50),
        "other_topics_count": int(global_direction_counter.get("altele", 0)),
        "top_topics": global_topic_model.top_topics_with_aliases(limit=200, alias_limit=8),
    }
    global_topics_path = global_topics_dir / "interventions_topics_index.json"
    global_topics_path.write_text(
        json.dumps(global_topics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Written: {global_topics_path}")

    members_dir = output_dir / "members"
    parties_dir = output_dir / "parties"
    print(f"  Clearing output dirs: {members_dir}, {parties_dir}")
    _clear_json_files(members_dir)
    _clear_json_files(parties_dir)

    print(f"  Writing {len(member_data)} member file(s)...")
    members_index = []
    for member_id in sorted(member_data.keys()):
        md = member_data[member_id]
        counts = md["counts"]
        bills_authored_total = len(md["authored_bill_ids"]) + int(md["generic_bills_authored_events"])
        amendments_added_total = len(md["amendment_ids"]) + int(md["generic_amendments_added_events"])
        interventions_total = counts["constructive"] + counts["neutral"] + counts["non_constructive"]
        top_topics = _top_topics(md["topics_counter"])
        top_directions = _top_directions(md["directions_counter"])
        other_topics_count = int(md["directions_counter"].get("altele", 0))
        members_index.append(
            {
                "member_id": md["member_id"],
                "name": md["name"],
                "party_id": md["party_id"],
                "party_name": md["party_name"],
                "interventions_total": interventions_total,
                "bills_authored_total": bills_authored_total,
                "amendments_added_total": amendments_added_total,
                "constructive_count": counts["constructive"],
                "neutral_count": counts["neutral"],
                "non_constructive_count": counts["non_constructive"],
                "top_topics": top_topics,
                "top_directions": top_directions,
                "other_topics_count": other_topics_count,
            }
        )

        member_detail = {
            "member_id": md["member_id"],
            "name": md["name"],
            "party_id": md["party_id"],
            "party_name": md["party_name"],
            "stats": {
                "interventions_total": interventions_total,
                "bills_authored_total": bills_authored_total,
                "amendments_added_total": amendments_added_total,
                "constructive_count": counts["constructive"],
                "neutral_count": counts["neutral"],
                "non_constructive_count": counts["non_constructive"],
            },
            "top_topics": top_topics,
            "top_directions": top_directions,
            "other_topics_count": other_topics_count,
            "interventions": md["interventions"],
        }
        member_name_slug = _slugify_name(md["name"])
        member_file = members_dir / f"interventions_{member_id}_{member_name_slug}.json"
        member_file.write_text(
            json.dumps(member_detail, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(
            f"    {member_file.name}  "
            f"total={interventions_total}  "
            f"constructive={counts['constructive']}  "
            f"neutral={counts['neutral']}  "
            f"non_constructive={counts['non_constructive']}"
        )

    members_index = sorted(members_index, key=lambda x: (-x["interventions_total"], x["member_id"]))
    (members_dir / "interventions_index.json").write_text(
        json.dumps(members_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"  Written: {members_dir / 'interventions_index.json'}")

    party_to_members: dict[str, list[dict]] = defaultdict(list)
    for member_entry in members_index:
        party_key = member_entry["party_id"] if member_entry["party_id"] else "unknown"
        party_to_members[party_key].append(member_entry)

    print(f"  Writing {len(party_to_members)} party file(s)...")
    parties_index = []
    for party_id in sorted(party_to_members.keys()):
        members = party_to_members[party_id]
        party_name = party_id if party_id != "unknown" else "Unknown"

        counts = {
            "constructive": sum(m["constructive_count"] for m in members),
            "neutral": sum(m["neutral_count"] for m in members),
            "non_constructive": sum(m["non_constructive_count"] for m in members),
        }
        bills_authored_total = sum(int(m.get("bills_authored_total", 0)) for m in members)
        amendments_added_total = sum(int(m.get("amendments_added_total", 0)) for m in members)
        interventions_total = counts["constructive"] + counts["neutral"] + counts["non_constructive"]

        topic_counter: Counter[str] = Counter()
        direction_counter: Counter[str] = Counter()
        for m in members:
            for topic in m["top_topics"]:
                topic_counter[topic["topic"]] += int(topic["count"])
            for direction in m.get("top_directions", []):
                direction_counter[direction["topic"]] += int(direction["count"])
        top_topics = _top_topics(topic_counter)
        top_directions = _top_directions(direction_counter)
        other_topics_count = int(sum(int(m.get("other_topics_count", 0)) for m in members))

        party_index_entry = {
            "party_id": party_id,
            "party_name": party_name,
            "members_count": len(members),
            "interventions_total": interventions_total,
            "bills_authored_total": bills_authored_total,
            "amendments_added_total": amendments_added_total,
            "constructive_count": counts["constructive"],
            "neutral_count": counts["neutral"],
            "non_constructive_count": counts["non_constructive"],
            "top_topics": top_topics,
            "top_directions": top_directions,
            "other_topics_count": other_topics_count,
        }
        parties_index.append(party_index_entry)

        party_detail = {
            "party_id": party_id,
            "party_name": party_name,
            "stats": {
                "members_count": len(members),
                "interventions_total": interventions_total,
                "bills_authored_total": bills_authored_total,
                "amendments_added_total": amendments_added_total,
                "constructive_count": counts["constructive"],
                "neutral_count": counts["neutral"],
                "non_constructive_count": counts["non_constructive"],
            },
            "top_topics": top_topics,
            "top_directions": top_directions,
            "other_topics_count": other_topics_count,
            "members": [
                {
                    "member_id": m["member_id"],
                    "name": m["name"],
                    "interventions_total": m["interventions_total"],
                    "bills_authored_total": int(m.get("bills_authored_total", 0)),
                    "amendments_added_total": int(m.get("amendments_added_total", 0)),
                    "constructive_count": m["constructive_count"],
                    "neutral_count": m["neutral_count"],
                    "non_constructive_count": m["non_constructive_count"],
                    "top_topics": m["top_topics"],
                    "top_directions": m.get("top_directions", []),
                    "other_topics_count": int(m.get("other_topics_count", 0)),
                }
                for m in sorted(members, key=lambda x: (-x["interventions_total"], x["member_id"]))
            ],
        }
        party_file = parties_dir / f"interventions_{party_id}.json"
        party_file.write_text(
            json.dumps(party_detail, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(
            f"    {party_file.name}  "
            f"members={len(members)}  "
            f"total={interventions_total}  "
            f"constructive={counts['constructive']}  "
            f"neutral={counts['neutral']}  "
            f"non_constructive={counts['non_constructive']}"
        )

    parties_index = sorted(parties_index, key=lambda x: (-x["interventions_total"], x["party_id"]))
    (parties_dir / "interventions_index.json").write_text(
        json.dumps(parties_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"  Written: {parties_dir / 'interventions_index.json'}")
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
    parser.add_argument(
        "--taxonomy-config",
        default=str(DEFAULT_TAXONOMY_CONFIG_PATH),
        help=f"Path to topic taxonomy JSON config (default: {DEFAULT_TAXONOMY_CONFIG_PATH})",
    )
    args = parser.parse_args()

    members_count, parties_count = export_outputs(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
        taxonomy_config_path=Path(args.taxonomy_config),
    )
    print(
        "Export completed: "
        f"{members_count} members, {parties_count} parties. "
        f"Output root: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
