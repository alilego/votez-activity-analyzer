#!/usr/bin/env python3
"""
Crawl CDEP deputy profile activity links and persist the linked records to SQLite.

The script starts from input/toti_deputatii.json, opens each deputy profile_url,
discovers the activity pages from the profile table, then stores list records for:

- Propuneri legislative initiate
- Proiecte de hotarare initiate
- Intrebari si interpelari
- Motiuni
- Declaratii politice depuse in scris

By default it only adds new crawler data. Pass --update-existing to refresh
existing crawler-owned rows from newly parsed pages. It never writes to the
pipeline's non-crawler data tables such as members, interventions, or runs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import io
import importlib.util
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from export_activity import (
    DEFAULT_ACTIVITY_OUTPUT_DIR,
    export_activity_snapshots,
)
from init_db import DEFAULT_DB_PATH


DEFAULT_INPUT_PATH = Path("input/toti_deputatii.json")
DEFAULT_LAW_INITIATOR_PDF_DIR = Path("outputs/pdfs/law_initiators")

# Inițiatori blocks are often on an early cover sheet but may appear on page 2+ of the PDF.
DEFAULT_LAW_INITIATOR_OCR_PAGES = 10
DEFAULT_USER_AGENT = "votez-activity-analyzer/1.0 (+https://github.com)"
DEFAULT_OCR_LANGUAGE = "ron+eng"

ACTIVITY_LABELS = {
    "legislative_proposals": "Propuneri legislative initiate",
    "decision_projects": "Proiecte de hotarare initiate",
    "questions": "Intrebari si interpelari",
    "motions": "Motiuni",
    "political_declarations": "Declaratii politice depuse in scris",
}

STANDARD_PROFILE_ACTIVITY_PAGES = {
    "legislative_proposals": "2",
    "decision_projects": "32",
    "questions": "3",
    "motions": "11",
}

VOID_TAGS = {"br", "hr", "img", "input", "link", "meta", "param"}

ACTIVITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dep_act_member_activity_crawl (
    member_id TEXT PRIMARY KEY,
    profile_url TEXT NOT NULL,
    legislative_proposals_url TEXT,
    legislative_proposals_count INTEGER,
    promulgated_laws_count INTEGER,
    decision_projects_url TEXT,
    decision_projects_count INTEGER,
    questions_url TEXT,
    questions_count INTEGER,
    motions_url TEXT,
    motions_count INTEGER,
    political_declarations_url TEXT,
    political_declarations_count INTEGER,
    legislative_proposals_stored INTEGER NOT NULL DEFAULT 0,
    decision_projects_stored INTEGER NOT NULL DEFAULT 0,
    questions_stored INTEGER NOT NULL DEFAULT 0,
    motions_stored INTEGER NOT NULL DEFAULT 0,
    political_declarations_stored INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    crawled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES members(member_id)
);

CREATE TABLE IF NOT EXISTS dep_act_laws (
    law_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL UNIQUE,
    identifier TEXT,
    adopted_law_identifier TEXT,
    motive_pdf_url TEXT,
    initiators_text TEXT,
    initiators_extracted_at TEXT,
    initiators_parse_error TEXT,
    initiators_source TEXT,
    title TEXT NOT NULL,
    details_text TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dep_act_member_laws (
    member_id TEXT NOT NULL,
    law_id TEXT NOT NULL,
    is_initiator INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES members(member_id),
    FOREIGN KEY (law_id) REFERENCES dep_act_laws(law_id),
    PRIMARY KEY (member_id, law_id)
);

CREATE TABLE IF NOT EXISTS dep_act_decision_projects (
    decision_project_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL UNIQUE,
    identifier TEXT,
    title TEXT NOT NULL,
    details_text TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dep_act_member_decision_projects (
    member_id TEXT NOT NULL,
    decision_project_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES members(member_id),
    FOREIGN KEY (decision_project_id) REFERENCES dep_act_decision_projects(decision_project_id),
    PRIMARY KEY (member_id, decision_project_id)
);

CREATE TABLE IF NOT EXISTS dep_act_questions_interpellations (
    question_id TEXT PRIMARY KEY,
    member_id TEXT NOT NULL,
    source_url TEXT NOT NULL UNIQUE,
    identifier TEXT,
    recipient TEXT,
    text TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES members(member_id)
);

CREATE TABLE IF NOT EXISTS dep_act_motions (
    motion_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    details_text TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dep_act_member_motions (
    member_id TEXT NOT NULL,
    motion_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES members(member_id),
    FOREIGN KEY (motion_id) REFERENCES dep_act_motions(motion_id),
    PRIMARY KEY (member_id, motion_id)
);

CREATE TABLE IF NOT EXISTS dep_act_political_declarations (
    political_declaration_id TEXT PRIMARY KEY,
    member_id TEXT NOT NULL,
    source_url TEXT NOT NULL UNIQUE,
    text_url TEXT,
    title TEXT NOT NULL,
    full_text TEXT NOT NULL,
    details_text TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES members(member_id)
);

CREATE INDEX IF NOT EXISTS idx_member_laws_law_id
    ON dep_act_member_laws(law_id);
CREATE INDEX IF NOT EXISTS idx_member_decision_projects_project_id
    ON dep_act_member_decision_projects(decision_project_id);
CREATE INDEX IF NOT EXISTS idx_member_motions_motion_id
    ON dep_act_member_motions(motion_id);
CREATE INDEX IF NOT EXISTS idx_political_declarations_member_id
    ON dep_act_political_declarations(member_id);
"""

ACTIVITY_MIGRATIONS = (
    "ALTER TABLE dep_act_laws ADD COLUMN adopted_law_identifier TEXT",
    "ALTER TABLE dep_act_laws ADD COLUMN motive_pdf_url TEXT",
    "ALTER TABLE dep_act_laws ADD COLUMN initiators_text TEXT",
    "ALTER TABLE dep_act_laws ADD COLUMN initiators_extracted_at TEXT",
    "ALTER TABLE dep_act_laws ADD COLUMN initiators_parse_error TEXT",
    "ALTER TABLE dep_act_laws ADD COLUMN initiators_source TEXT",
    "ALTER TABLE dep_act_member_laws ADD COLUMN is_initiator INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE dep_act_member_activity_crawl ADD COLUMN political_declarations_url TEXT",
    "ALTER TABLE dep_act_member_activity_crawl ADD COLUMN political_declarations_count INTEGER",
    "ALTER TABLE dep_act_member_activity_crawl ADD COLUMN political_declarations_stored INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE dep_act_questions_interpellations ADD COLUMN member_id TEXT",
    "ALTER TABLE dep_act_questions_interpellations ADD COLUMN identifier TEXT",
    "ALTER TABLE dep_act_questions_interpellations ADD COLUMN recipient TEXT",
)

LEGACY_ACTIVITY_TABLE_RENAMES = (
    ("member_activity_crawl", "dep_act_member_activity_crawl"),
    ("laws", "dep_act_laws"),
    ("member_laws", "dep_act_member_laws"),
    ("decision_projects", "dep_act_decision_projects"),
    ("member_decision_projects", "dep_act_member_decision_projects"),
    ("questions_interpellations", "dep_act_questions_interpellations"),
    ("motions", "dep_act_motions"),
    ("member_motions", "dep_act_member_motions"),
    ("political_declarations", "dep_act_political_declarations"),
)


ACTIVITY_JOIN_TABLE_FOREIGN_KEYS = (
    (
        "dep_act_member_laws",
        {"members", "dep_act_laws"},
        """
        CREATE TABLE {table} (
            member_id TEXT NOT NULL,
            law_id TEXT NOT NULL,
            is_initiator INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (law_id) REFERENCES dep_act_laws(law_id),
            PRIMARY KEY (member_id, law_id)
        )
        """,
        ("member_id", "law_id", "is_initiator", "created_at"),
    ),
    (
        "dep_act_member_decision_projects",
        {"members", "dep_act_decision_projects"},
        """
        CREATE TABLE {table} (
            member_id TEXT NOT NULL,
            decision_project_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (decision_project_id) REFERENCES dep_act_decision_projects(decision_project_id),
            PRIMARY KEY (member_id, decision_project_id)
        )
        """,
        ("member_id", "decision_project_id", "created_at"),
    ),
    (
        "dep_act_member_motions",
        {"members", "dep_act_motions"},
        """
        CREATE TABLE {table} (
            member_id TEXT NOT NULL,
            motion_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (member_id) REFERENCES members(member_id),
            FOREIGN KEY (motion_id) REFERENCES dep_act_motions(motion_id),
            PRIMARY KEY (member_id, motion_id)
        )
        """,
        ("member_id", "motion_id", "created_at"),
    ),
)


@dataclass
class Link:
    href: str
    text: str = ""


@dataclass
class Cell:
    text_parts: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _clean_text(" ".join(self.text_parts))


@dataclass
class Row:
    cells: list[Cell] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _clean_text(" ".join(cell.text for cell in self.cells))

    @property
    def links(self) -> list[Link]:
        out: list[Link] = []
        for cell in self.cells:
            out.extend(cell.links)
        return out


@dataclass
class ActivityLink:
    url: str | None = None
    count: int | None = None
    promulgated_count: int | None = None
    raw_text: str = ""


@dataclass
class ListingRecord:
    record_id: str
    source_url: str
    identifier: str | None
    title: str
    details_text: str
    columns: list[str]
    adopted_law_identifier: str | None = None
    recipient: str | None = None


@dataclass
class PoliticalDeclarationRecord:
    declaration_id: str
    member_id: str
    source_url: str
    text_url: str | None
    title: str
    full_text: str
    details_text: str
    columns: list[str]


@dataclass
class StoreResult:
    seen: int = 0
    stored: int = 0
    associated: int = 0


class TableExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[Row] = []
        self._current_row: Row | None = None
        self._current_cell: Cell | None = None
        self._current_link: Link | None = None
        self._row_stack: list[tuple[Row, Cell | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag == "tr":
            self._finish_link()
            if self._current_row is not None:
                self._row_stack.append((self._current_row, self._current_cell))
            self._current_row = Row()
            self._current_cell = None
        elif tag in {"td", "th"} and self._current_row is not None:
            self._finish_link()
            self._current_cell = Cell()
            self._current_row.cells.append(self._current_cell)
        elif tag == "a" and self._current_cell is not None:
            self._finish_link()
            self._current_link = Link(href=attrs_dict.get("href", ""), text="")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a":
            self._finish_link()
        elif tag in {"td", "th"}:
            self._finish_link()
            self._current_cell = None
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._current_cell is None:
            return
        self._current_cell.text_parts.append(data)
        if self._current_link is not None:
            self._current_link.text += data

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in VOID_TAGS and self._current_cell is not None:
            self._current_cell.text_parts.append(" ")

    def _finish_link(self) -> None:
        if self._current_link is None:
            return
        self._current_link.text = _clean_text(self._current_link.text)
        if self._current_link.href and self._current_cell is not None:
            self._current_cell.links.append(self._current_link)
        self._current_link = None

    def _finish_row(self) -> None:
        self._finish_link()
        completed = self._current_row
        if completed is not None and completed.cells:
            self.rows.append(completed)
        if self._row_stack:
            parent_row, parent_cell = self._row_stack.pop()
            if completed is not None and parent_cell is not None:
                nested_text = completed.text
                if nested_text:
                    parent_cell.text_parts.append(nested_text)
                parent_cell.links.extend(completed.links)
            self._current_row = parent_row
            self._current_cell = parent_cell
            return
        self._current_row = None
        self._current_cell = None


class PageTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[Link] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._current_link: Link | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            attrs_dict = {k.lower(): (v or "") for k, v in attrs}
            self._current_link = Link(href=attrs_dict.get("href", ""), text="")
        elif tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._current_link is not None:
            self._current_link.text = _clean_text(self._current_link.text)
            if self._current_link.href:
                self.links.append(self._current_link)
            self._current_link = None
        elif tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data)
        if self._in_title:
            self.title_parts.append(data)
        if self._current_link is not None:
            self._current_link.text += data

    @property
    def text(self) -> str:
        return _clean_text(" ".join(self.parts))

    @property
    def title(self) -> str:
        return _clean_text(" ".join(self.title_parts))


class ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._current_parts: list[str] | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "p":
            self._finish_paragraph()
            self._current_parts = []
        elif tag == "br" and self._current_parts is not None:
            self._current_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "p" and self._current_parts is not None:
            self._finish_paragraph()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._current_parts is None:
            return
        self._current_parts.append(data)

    def close(self) -> None:
        self._finish_paragraph()
        super().close()

    def _finish_paragraph(self) -> None:
        if self._current_parts is None:
            return
        text = _clean_text(" ".join(self._current_parts))
        if text:
            self.paragraphs.append(text)
        self._current_parts = None


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _fold(value: str) -> str:
    text = _clean_text(value).replace("ţ", "ț").replace("Ţ", "Ț")
    text = text.replace("ş", "ș").replace("Ş", "Ș")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def _parse_rows(html_text: str) -> list[Row]:
    parser = TableExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.rows


def _first_int(value: str) -> int | None:
    match = re.search(r"\b(\d+)\b", value or "")
    return int(match.group(1)) if match else None


def _promulgated_count(value: str) -> int | None:
    folded = _fold(value)
    match = re.search(r"din\s+care\s+(\d+)\s+promulgate", folded)
    return int(match.group(1)) if match else None


def _activity_label_key(value: str) -> str:
    return _fold(value).strip(" :-–—")


def parse_profile_activity(html_text: str, base_url: str) -> dict[str, ActivityLink]:
    rows = _parse_rows(html_text)
    out = {key: ActivityLink() for key in ACTIVITY_LABELS}
    folded_labels = {key: _activity_label_key(label) for key, label in ACTIVITY_LABELS.items()}

    for row in rows:
        for index, cell in enumerate(row.cells):
            folded_cell = _activity_label_key(cell.text)
            for key, folded_label in folded_labels.items():
                if folded_label != folded_cell:
                    continue
                value_cells = row.cells[index + 1 :] if index + 1 < len(row.cells) else []
                value_text = _clean_text(" ".join(c.text for c in value_cells))
                links: list[Link] = []
                for c in value_cells:
                    links.extend(c.links)
                if not links:
                    links = row.links
                url = _first_http_link(links, base_url)
                if out[key].url and not url:
                    continue
                out[key] = ActivityLink(
                    url=url,
                    count=_first_int(value_text),
                    promulgated_count=(
                        _promulgated_count(value_text)
                        if key == "legislative_proposals"
                        else None
                    ),
                    raw_text=value_text,
                )
    return out


def _first_http_link(links: list[Link], base_url: str) -> str | None:
    for link in links:
        href = (link.href or "").strip()
        if not href or href.startswith("#"):
            continue
        folded_href = href.casefold()
        if folded_href.startswith(("javascript:", "mailto:")):
            continue
        return urllib.parse.urljoin(base_url, href)
    return None


def _standard_profile_activity_url(profile_url: str, activity_key: str) -> str | None:
    page = STANDARD_PROFILE_ACTIVITY_PAGES.get(activity_key)
    if not page:
        return None
    parsed = urllib.parse.urlparse(profile_url)
    if not parsed.path.endswith("structura2015.mp"):
        return None
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if not query.get("idm"):
        return None
    query["pag"] = [page]
    query.setdefault("cam", ["2"])
    query.setdefault("leg", ["2024"])
    query.setdefault("idl", ["1"])
    query.setdefault("prn", ["0"])
    query.setdefault("par", [""])
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
    )


LAW_IDENTIFIER_RE = re.compile(
    r"\b(PL[\s\-–—]*x|BPI|BP|B|L)\s*(?:nr\.?)?[\s\-–—]*(\d{1,5}/\d{4})\b",
    re.IGNORECASE,
)
DECISION_IDENTIFIER_RE = re.compile(
    r"\b(?:PHCD|PHC|HOT[AĂ]R[AÂ]RE(?:A)?(?:\s+CAMEREI\s+DEPUTA[ȚT]ILOR)?)\s*(?:nr\.?)?\s*(\d{1,5}/\d{4})\b",
    re.IGNORECASE,
)
QUESTION_IDENTIFIER_RE = re.compile(
    r"\bnr\.?\s*([0-9]+[A-Z]?/[0-9]{2}-[0-9]{2}-[0-9]{4})\b",
    re.IGNORECASE,
)
ADOPTED_LAW_IDENTIFIER_RE = re.compile(
    r"\bLege(?:a)?\s*(?:nr\.?)?\s*(\d{1,5}/\d{4})\b",
    re.IGNORECASE,
)


def _extract_identifier(kind: str, text: str) -> str | None:
    if kind == "laws":
        match = LAW_IDENTIFIER_RE.search(text)
        if match:
            prefix = _fold(match.group(1))
            token = match.group(2)
            if prefix.startswith("pl"):
                return f"PL-x {token}"
            return f"{prefix.upper()}{token}"
    if kind == "decision_projects":
        match = DECISION_IDENTIFIER_RE.search(text)
        if match:
            return _clean_text(text[match.start() : match.end()])
    if kind == "questions":
        match = QUESTION_IDENTIFIER_RE.search(text)
        if match:
            return f"nr.{match.group(1)}"
    return None


def _extract_adopted_law_identifier(text: str) -> str | None:
    match = ADOPTED_LAW_IDENTIFIER_RE.search(text)
    if not match:
        return None
    return f"Lege {match.group(1)}"


def _is_law_metadata_title(value: str, identifier: str | None) -> bool:
    text = _clean_text(value)
    if not text:
        return True
    folded = _fold(text)
    if identifier and folded == _fold(identifier):
        return True
    if LAW_IDENTIFIER_RE.fullmatch(text):
        return True
    if re.fullmatch(r"\d+\.?", text):
        return True
    if re.fullmatch(r"\d{1,5}/\d{2}\.\d{2}\.\d{4}", text):
        return True
    return False


def _normalize_law_listing_fields(
    details_text: str,
    title: str,
) -> tuple[str | None, str, str]:
    identifier = _extract_identifier("laws", details_text)
    normalized_details = _clean_text(details_text)
    if identifier:
        match = LAW_IDENTIFIER_RE.search(normalized_details)
        if match:
            normalized_details = _clean_text(
                normalized_details[match.end() :].strip(" .:-–—")
            )
    normalized_title = _clean_text(title)
    if normalized_details and _is_law_metadata_title(normalized_title, identifier):
        normalized_title = normalized_details
    elif not normalized_title:
        normalized_title = normalized_details

    return identifier, normalized_title, normalized_details


def _normalize_question_listing_fields(
    details_text: str,
    title: str,
) -> tuple[str | None, str, str]:
    identifier = _extract_identifier("questions", details_text)
    normalized_details = _clean_text(details_text)
    if identifier:
        match = QUESTION_IDENTIFIER_RE.search(normalized_details)
        if match:
            normalized_details = _clean_text(
                normalized_details[match.end() :].strip(" .:-–—")
            )

    normalized_title = _clean_text(title)
    if not normalized_title or _fold(normalized_title) in {
        "intrebarea",
        "întrebarea",
        "interpelarea",
        _fold(identifier or ""),
    } or (identifier and _fold(identifier) in _fold(normalized_title)):
        normalized_title = normalized_details

    return identifier, normalized_title, normalized_details


def _law_stage_column_index(row: Row) -> int | None:
    for index, cell in enumerate(row.cells):
        folded = _fold(cell.text).rstrip(":")
        if folded == "stadiu":
            return index
    return None


def _extract_adopted_law_identifier_from_columns(
    columns: list[str],
    stage_column_index: int | None,
) -> tuple[str | None, int | None]:
    if stage_column_index is not None and stage_column_index < len(columns):
        candidate_indices = [stage_column_index]
    elif len(columns) >= 5:
        candidate_indices = [len(columns) - 1]
    else:
        candidate_indices = []

    for index in candidate_indices:
        identifier = _extract_adopted_law_identifier(columns[index])
        if identifier:
            return identifier, index
    return None, None


def _law_details_without_stage_column(
    columns: list[str],
    fallback: str,
    stage_column_index: int | None,
) -> str:
    if stage_column_index is None or stage_column_index >= len(columns):
        return fallback
    return _clean_text(
        " ".join(
            column
            for index, column in enumerate(columns)
            if index != stage_column_index
        )
    )


def parse_question_detail_recipient(html_text: str) -> str | None:
    values: list[str] = []
    collecting = False
    for row in _parse_rows(html_text):
        if len(row.cells) < 2:
            continue
        label = _fold(row.cells[0].text).rstrip(":")
        if label in {"destinatar", "destinatari"}:
            collecting = True
            value = _clean_text(" ".join(cell.text for cell in row.cells[1:]))
            if value:
                values.append(value)
            continue
        if collecting:
            if label:
                break
            value = _clean_text(" ".join(cell.text for cell in row.cells[1:]))
            if value:
                values.append(value)
    return " | ".join(values) if values else None


def _extract_political_declaration_title(value: str) -> str:
    text = _clean_text(value)
    quote_match = re.search(r"[\"„“]([^\"”„“]+)[\"”]", text)
    if quote_match:
        return _clean_text(quote_match.group(1).strip(" ;:"))

    folded = _fold(text)
    marker_match = re.search(r"\bdeclarat(?:ie|ia)\s+politic(?:a|e)\b", folded)
    if not marker_match:
        return text

    title = text[marker_match.end() :]
    title = re.sub(
        r"^\s*(?:referitoare\s+la|privind|despre|cu\s+tema|:|-)\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s*;?\s*în\s+scris\s*$", "", title, flags=re.IGNORECASE)
    return _strip_wrapping_quotes(title.strip(" ;:-"))


def _query_value(url: str, keys: tuple[str, ...]) -> str | None:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    lowered = {key.casefold(): value for key, value in query.items()}
    for key in keys:
        values = lowered.get(key.casefold())
        if values and values[0].strip():
            return values[0].strip()
    return None


def _stable_record_id(kind: str, source_url: str, text: str) -> str:
    if kind == "laws":
        value = _query_value(source_url, ("idp", "id", "nr"))
        if value:
            return f"law:cdep:{value}"
        identifier = _extract_identifier(kind, text)
        if identifier:
            return "law:" + re.sub(r"[^a-z0-9]+", "_", _fold(identifier)).strip("_")
    elif kind == "decision_projects":
        value = _query_value(source_url, ("idp", "idh", "id", "nr"))
        if value:
            return f"decision_project:cdep:{value}"
    elif kind == "questions":
        value = _query_value(source_url, ("idi", "idint", "id", "nr"))
        if value:
            return f"question:cdep:{value}"
        identifier = _extract_identifier(kind, text)
        if identifier:
            return "question:" + re.sub(r"[^a-z0-9]+", "_", _fold(identifier)).strip("_")
    elif kind == "motions":
        value = _query_value(source_url, ("idm", "id", "idmotiune", "idmo", "nr"))
        if value:
            return f"motion:cdep:{value}"
    elif kind == "political_declarations":
        ids = _query_value(source_url, ("ids",))
        idm = _query_value(source_url, ("idm",))
        if ids and idm:
            safe_idm = re.sub(r"[^0-9a-z]+", "_", _fold(idm)).strip("_")
            return f"political_declaration:cdep:{ids}:{safe_idm}"
        value = _query_value(source_url, ("ids", "id", "idv", "idp", "id_decl", "nr"))
        if value:
            return f"political_declaration:cdep:{value}"

    digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]
    return f"{kind}:url:{digest}"


def _is_political_declaration_detail_url(url: str) -> bool:
    folded_url = _fold(url)
    if "stenograma_scris" not in folded_url:
        return False
    ids = _query_value(url, ("ids",))
    idm = _query_value(url, ("idm",))
    return bool(ids and idm and "," in idm)


def _record_link_score(kind: str, url: str, link_text: str, row_text: str) -> int:
    haystack = _fold(" ".join([url, link_text, row_text]))
    score = 0
    if kind == "laws":
        if "proiect" in haystack:
            score += 4
        if "pl-x" in haystack or "pl x" in haystack:
            score += 4
        if LAW_IDENTIFIER_RE.search(row_text):
            score += 3
    elif kind == "decision_projects":
        if "hotarare" in haystack:
            score += 4
        if "phcd" in haystack or "phc" in haystack:
            score += 3
        if "proiect" in haystack:
            score += 2
    elif kind == "questions":
        if "interpel" in haystack:
            score += 4
        if "intrebar" in haystack:
            score += 4
    elif kind == "motions":
        if "motiun" in haystack:
            score += 5
    elif kind == "political_declarations":
        if not _is_political_declaration_detail_url(url):
            return 0
        score += 10
        if "declarat" in haystack and "politic" in haystack:
            score += 6
        elif "declarat" in haystack:
            score += 3
        if not _query_value(url, ("sir", "nav")):
            score += 2
    if re.search(r"\d", url):
        score += 1
    return score


def _select_record_link(kind: str, row: Row, base_url: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    row_text = row.text
    for link in row.links:
        href = (link.href or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.casefold().startswith(("javascript:", "mailto:")):
            continue
        url = urllib.parse.urljoin(base_url, href)
        if kind == "laws":
            # Fix 8 (part 1): inject the missing `cam=2` on CDEP law URLs at
            # ingestion time so every downstream step (including a fresh
            # hydration run) stores a directly-fetchable URL.
            url = _normalize_law_source_url(url)
        score = _record_link_score(kind, url, link.text, row_text)
        if score > 0:
            candidates.append((score, url))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


SKIP_ROW_PATTERNS = (
    "camera deputatilor",
    "senatul romaniei",
    "contact",
    "harta site",
    "curriculum vitae",
    "declaratii de avere",
    "declaratii de interese",
    "luari de cuvant",
    "voturi electronice",
)


def _is_probable_data_row(kind: str, row: Row, source_url: str) -> bool:
    text = row.text
    folded = _fold(text)
    if len(text) < 8 or not row.links:
        return False
    if any(pattern in folded for pattern in SKIP_ROW_PATTERNS):
        return False
    if not re.search(r"\d", text) and kind not in {"motions", "political_declarations"}:
        return False
    link = _select_record_link(kind, row, source_url)
    return link is not None


def parse_listing_records(html_text: str, source_url: str, kind: str) -> list[ListingRecord]:
    rows = _parse_rows(html_text)
    records: list[ListingRecord] = []
    seen_urls: set[str] = set()
    law_stage_index: int | None = None
    law_stage_from_right: int | None = None
    for row in rows:
        if kind == "laws":
            detected_stage_index = _law_stage_column_index(row)
            if detected_stage_index is not None:
                law_stage_index = detected_stage_index
                law_stage_from_right = len(row.cells) - detected_stage_index - 1
        if not _is_probable_data_row(kind, row, source_url):
            continue
        record_url = _select_record_link(kind, row, source_url)
        if not record_url or record_url in seen_urls:
            continue
        seen_urls.add(record_url)
        columns = [cell.text for cell in row.cells if cell.text]
        details_text = row.text
        record_id = _stable_record_id(kind, record_url, details_text)
        link_text = ""
        for link in row.links:
            if urllib.parse.urljoin(source_url, link.href) == record_url:
                link_text = link.text
                break
        title = _clean_text(link_text) or (columns[1] if len(columns) > 1 else details_text)
        identifier = _extract_identifier(kind, details_text)
        adopted_law_identifier = None
        if kind == "laws":
            row_stage_index = law_stage_index
            if law_stage_from_right is not None and law_stage_from_right < len(columns):
                row_stage_index = len(columns) - law_stage_from_right - 1
            adopted_law_identifier, row_stage_index = (
                _extract_adopted_law_identifier_from_columns(
                    columns,
                    row_stage_index,
                )
            )
            if (
                adopted_law_identifier is None
                and law_stage_from_right is not None
                and law_stage_from_right < len(columns)
            ):
                row_stage_index = len(columns) - law_stage_from_right - 1
            details_text = _law_details_without_stage_column(
                columns,
                details_text,
                row_stage_index,
            )
            identifier, title, details_text = _normalize_law_listing_fields(
                details_text,
                title,
            )
        elif kind == "questions":
            identifier, title, details_text = _normalize_question_listing_fields(
                details_text,
                title,
            )
        elif kind == "political_declarations":
            title = _extract_political_declaration_title(details_text) or title
        records.append(
            ListingRecord(
                record_id=record_id,
                source_url=record_url,
                identifier=identifier,
                title=title,
                details_text=details_text,
                columns=columns,
                adopted_law_identifier=adopted_law_identifier,
            )
        )
    return records


def _page_text_and_links(html_text: str) -> tuple[str, str, list[Link]]:
    parser = PageTextExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.text, parser.title, parser.links


def _paragraphs_from_html(html_text: str) -> list[str]:
    parser = ParagraphExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.paragraphs


def _political_declaration_content_html(html_text: str) -> str:
    match = re.search(
        r"<!--\s*START=.*?-->(.*?)(?:<!--\s*END\s*-->|$)",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else html_text


def _strip_wrapping_quotes(value: str) -> str:
    text = _clean_text(value)
    while len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "“" and text[-1] == "”")
        or (text[0] == "„" and text[-1] == "”")
    ):
        text = _clean_text(text[1:-1])
    return text


def parse_political_declaration_text_page(html_text: str) -> tuple[str, str]:
    content_html = _political_declaration_content_html(html_text)
    paragraphs = [
        text
        for text in (_strip_wrapping_quotes(p) for p in _paragraphs_from_html(content_html))
        if text
    ]
    content_paragraphs: list[str] = []
    for paragraph in paragraphs:
        folded = _fold(paragraph)
        if re.search(r"\b(domnul|doamna)\b", folded) and paragraph.endswith(":"):
            continue
        content_paragraphs.append(paragraph)

    if content_paragraphs:
        title = content_paragraphs[0]
        body = "\n\n".join(content_paragraphs[1:]).strip()
        return title, body or title

    full_text, page_title, _ = _page_text_and_links(html_text)
    return _strip_wrapping_quotes(page_title), full_text


def _select_political_declaration_text_link(html_text: str, base_url: str) -> str | None:
    rows = _parse_rows(html_text)
    candidates: list[tuple[int, str]] = []
    for row in rows:
        row_folded = _fold(row.text)
        for link in row.links:
            href = (link.href or "").strip()
            if not href or href.startswith("#"):
                continue
            if href.casefold().startswith(("javascript:", "mailto:")):
                continue
            url = urllib.parse.urljoin(base_url, href)
            haystack = _fold(" ".join([row.text, link.text, url]))
            score = 0
            if "stenograma_scris" in haystack:
                score += 10
            if "text" in haystack:
                score += 4
            if "declarat" in haystack:
                score += 3
            if "politic" in haystack:
                score += 2
            if "afis" in haystack or "vizualiz" in haystack or "html" in haystack:
                score += 1
            if row_folded and ("declarat" in row_folded or "text" in row_folded):
                score += 1
            if score:
                candidates.append((score, url))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    _, _, links = _page_text_and_links(html_text)
    for link in links:
        haystack = _fold(" ".join([link.text, link.href]))
        if "declarat" in haystack and ("text" in haystack or "politic" in haystack):
            return urllib.parse.urljoin(base_url, link.href)
    return None


def parse_political_declaration_detail(
    detail_html: str,
    detail_url: str,
    *,
    fetcher: Fetcher | None = None,
) -> tuple[str | None, str, str]:
    if _is_political_declaration_detail_url(detail_url):
        title, full_text = parse_political_declaration_text_page(detail_html)
        return detail_url, title, full_text

    text_url = _select_political_declaration_text_link(detail_html, detail_url)
    source_html = detail_html
    if text_url and fetcher is not None and text_url != detail_url:
        source_html = fetcher.fetch(text_url)
    title, full_text = parse_political_declaration_text_page(source_html)
    return text_url or detail_url, title, full_text


def _debug_preview(value: str, *, limit: int = 220) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _is_bad_political_declaration_text(value: str) -> bool:
    folded = _fold(value)
    return (
        not folded
        or folded == "copyright © camera deputatilor"
        or "copyright © camera deputatilor" in folded
    )


def _print_political_declaration_listing_debug(
    *,
    member_id: str,
    listing_url: str,
    listing_html: str,
    records: list[ListingRecord],
) -> None:
    print(f"[political_declarations debug] member_id={member_id}")
    print(f"[political_declarations debug] listing_url={listing_url}")
    print(f"[political_declarations debug] listing_html_len={len(listing_html)}")
    print(f"[political_declarations debug] listing_records={len(records)}")
    for index, record in enumerate(records, start=1):
        print(
            "[political_declarations debug] "
            f"listing_record[{index}] id={record.record_id}"
        )
        print(
            "[political_declarations debug] "
            f"listing_record[{index}] source_url={record.source_url}"
        )
        print(
            "[political_declarations debug] "
            f"listing_record[{index}] title={_debug_preview(record.title)}"
        )
        print(
            "[political_declarations debug] "
            f"listing_record[{index}] details={_debug_preview(record.details_text)}"
        )


def _print_political_declaration_detail_debug(
    *,
    index: int,
    record: ListingRecord,
    detail_html: str,
    text_url: str | None,
    detail_title: str,
    full_text: str,
) -> None:
    paragraphs = _paragraphs_from_html(detail_html)
    page_text, page_title, links = _page_text_and_links(detail_html)
    print(
        "[political_declarations debug] "
        f"detail[{index}] source_url={record.source_url}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] is_individual_url={_is_political_declaration_detail_url(record.source_url)}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] html_len={len(detail_html)} contains_start={'START=' in detail_html} "
        f"contains_p_tag={'<p' in detail_html.lower()} contains_copyright={'Copyright' in detail_html}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] html_head_preview={_debug_preview(detail_html[:700], limit=500)}"
    )
    start_index = detail_html.find("START=")
    if start_index >= 0:
        snippet_start = max(0, start_index - 220)
        snippet_end = min(len(detail_html), start_index + 700)
        print(
            "[political_declarations debug] "
            f"detail[{index}] html_start_marker_preview="
            f"{_debug_preview(detail_html[snippet_start:snippet_end], limit=700)}"
        )
    print(
        "[political_declarations debug] "
        f"detail[{index}] page_title={_debug_preview(page_title)}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] page_text_preview={_debug_preview(page_text)}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] paragraph_count={len(paragraphs)}"
    )
    for paragraph_index, paragraph in enumerate(paragraphs[:8], start=1):
        print(
            "[political_declarations debug] "
            f"detail[{index}] paragraph[{paragraph_index}]={_debug_preview(paragraph)}"
        )
    print(
        "[political_declarations debug] "
        f"detail[{index}] link_count={len(links)}"
    )
    for link_index, link in enumerate(links[:12], start=1):
        print(
            "[political_declarations debug] "
            f"detail[{index}] link[{link_index}] href={link.href} text={_debug_preview(link.text, limit=120)}"
        )
    print(
        "[political_declarations debug] "
        f"detail[{index}] parsed_text_url={text_url or '-'}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] parsed_title={_debug_preview(detail_title)}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] parsed_full_text_len={len(full_text)}"
    )
    print(
        "[political_declarations debug] "
        f"detail[{index}] parsed_full_text_preview={_debug_preview(full_text)}"
    )
    if _is_bad_political_declaration_text(full_text):
        print(
            "[political_declarations debug] "
            f"detail[{index}] WARNING parsed full_text looks like footer/empty text"
        )


def _print_questions_listing_debug(
    *,
    member_id: str,
    listing_url: str,
    listing_html: str,
    records: list[ListingRecord],
) -> None:
    print(f"[questions debug] member_id={member_id}")
    print(f"[questions debug] listing_url={listing_url}")
    print(f"[questions debug] listing_html_len={len(listing_html)}")
    print(f"[questions debug] listing_records={len(records)}")
    for index, record in enumerate(records, start=1):
        print(f"[questions debug] listing_record[{index}] id={record.record_id}")
        print(f"[questions debug] listing_record[{index}] source_url={record.source_url}")
        print(f"[questions debug] listing_record[{index}] identifier={record.identifier or '-'}")
        print(f"[questions debug] listing_record[{index}] title={_debug_preview(record.title)}")
        print(
            f"[questions debug] listing_record[{index}] details="
            f"{_debug_preview(record.details_text)}"
        )


def _print_question_detail_debug(
    *,
    index: int,
    record: ListingRecord,
    detail_html: str,
    recipient: str | None,
) -> None:
    rows = _parse_rows(detail_html)
    page_text, page_title, links = _page_text_and_links(detail_html)
    print(f"[questions debug] detail[{index}] source_url={record.source_url}")
    print(f"[questions debug] detail[{index}] html_len={len(detail_html)}")
    print(f"[questions debug] detail[{index}] html_head_preview={_debug_preview(detail_html[:700], limit=500)}")
    print(f"[questions debug] detail[{index}] page_title={_debug_preview(page_title)}")
    print(f"[questions debug] detail[{index}] page_text_preview={_debug_preview(page_text)}")
    print(f"[questions debug] detail[{index}] row_count={len(rows)}")
    for row_index, row in enumerate(rows[:16], start=1):
        cells = " | ".join(_debug_preview(cell.text, limit=120) for cell in row.cells)
        print(f"[questions debug] detail[{index}] row[{row_index}]={cells}")
    print(f"[questions debug] detail[{index}] link_count={len(links)}")
    for link_index, link in enumerate(links[:12], start=1):
        print(
            f"[questions debug] detail[{index}] link[{link_index}] "
            f"href={link.href} text={_debug_preview(link.text, limit=120)}"
        )
    print(f"[questions debug] detail[{index}] parsed_recipient={recipient or '-'}")
    if not recipient:
        print(f"[questions debug] detail[{index}] WARNING no Destinatar row parsed")


def _print_motions_listing_debug(
    *,
    member_id: str,
    listing_url: str | None,
    listing_html: str,
    records: list[ListingRecord],
) -> None:
    print(f"[motions debug] member_id={member_id}")
    print(f"[motions debug] listing_url={listing_url or '-'}")
    print(f"[motions debug] listing_html_len={len(listing_html)}")
    print(f"[motions debug] listing_records={len(records)}")
    rows = _parse_rows(listing_html)
    print(f"[motions debug] listing_row_count={len(rows)}")
    for row_index, row in enumerate(rows[:24], start=1):
        cells = " | ".join(_debug_preview(cell.text, limit=140) for cell in row.cells)
        links = " ; ".join(
            f"{link.href} :: {_debug_preview(link.text, limit=80)}"
            for link in row.links[:4]
        )
        print(f"[motions debug] row[{row_index}]={cells}")
        if links:
            print(f"[motions debug] row[{row_index}] links={links}")
    for index, record in enumerate(records, start=1):
        print(f"[motions debug] listing_record[{index}] id={record.record_id}")
        print(f"[motions debug] listing_record[{index}] source_url={record.source_url}")
        print(f"[motions debug] listing_record[{index}] title={_debug_preview(record.title)}")
        print(
            f"[motions debug] listing_record[{index}] details="
            f"{_debug_preview(record.details_text)}"
        )


def _print_profile_activity_debug(
    *,
    member_id: str,
    profile_url: str,
    profile_html: str,
    activity: dict[str, ActivityLink],
) -> None:
    print(f"[profile debug] member_id={member_id}")
    print(f"[profile debug] profile_url={profile_url}")
    print(f"[profile debug] profile_html_len={len(profile_html)}")
    rows = _parse_rows(profile_html)
    activity_words = (
        "propuneri",
        "hotarare",
        "hotarare",
        "intreb",
        "interpel",
        "motiun",
        "declar",
    )
    matched = [
        row
        for row in rows
        if any(word in _fold(row.text) for word in activity_words)
    ]
    print(f"[profile debug] parsed_rows={len(rows)} activity_like_rows={len(matched)}")
    for row_index, row in enumerate(matched[:40], start=1):
        cells = " | ".join(_debug_preview(cell.text, limit=160) for cell in row.cells)
        links = " ; ".join(
            f"{link.href} :: {_debug_preview(link.text, limit=80)}"
            for link in row.links[:6]
        )
        print(f"[profile debug] row[{row_index}]={cells}")
        if links:
            print(f"[profile debug] row[{row_index}] links={links}")
    for key, link in activity.items():
        print(
            f"[profile debug] activity[{key}] count={link.count} "
            f"url={link.url or '-'} raw_text={_debug_preview(link.raw_text)}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level JSON object.")
    return data


def _normalize_for_matching(text: str) -> str:
    text = re.sub(r"\s+-\s+.*$", "", text or "")
    text = re.sub(r"^\s*(domnul|doamna)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def _load_deputies(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("members")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected 'members' list.")
    deputies = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_member_id = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        profile_url = str(row.get("profile_url", "")).strip()
        if not source_member_id or not name or not profile_url:
            continue
        deputies.append(
            {
                "member_id": f"deputat_{source_member_id}",
                "source_member_id": source_member_id,
                "chamber": "deputat",
                "name": name,
                "normalized_name": _normalize_for_matching(name),
                "party_id": str(row.get("party", "")).strip() or None,
                "profile_url": profile_url,
                "circumscriptie": (
                    str(row.get("circumscriptie", "")).strip()
                    if row.get("circumscriptie") is not None
                    else None
                ),
            }
        )
    return deputies


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def rename_legacy_activity_tables(conn: sqlite3.Connection) -> None:
    for old_name, new_name in LEGACY_ACTIVITY_TABLE_RENAMES:
        if not _table_exists(conn, old_name) or _table_exists(conn, new_name):
            continue
        conn.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")


def repair_activity_join_foreign_keys(conn: sqlite3.Connection) -> None:
    for table, expected_targets, create_sql, columns in ACTIVITY_JOIN_TABLE_FOREIGN_KEYS:
        if not _table_exists(conn, table):
            continue
        fk_targets = {
            row[2]
            for row in conn.execute(
                f"PRAGMA foreign_key_list({_quote_identifier(table)})"
            )
        }
        if fk_targets == expected_targets:
            continue
        temp_table = f"__rebuild_{table}"
        quoted_temp = _quote_identifier(temp_table)
        quoted_table = _quote_identifier(table)
        existing_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({quoted_table})")
        }
        copy_columns = [column for column in columns if column in existing_columns]
        column_sql = ", ".join(_quote_identifier(column) for column in copy_columns)
        conn.execute(f"DROP TABLE IF EXISTS {quoted_temp}")
        conn.execute(create_sql.format(table=quoted_temp))
        if copy_columns:
            conn.execute(
                f"INSERT INTO {quoted_temp} ({column_sql}) "
                f"SELECT {column_sql} FROM {quoted_table}"
            )
        conn.execute(f"DROP TABLE {quoted_table}")
        conn.execute(f"ALTER TABLE {quoted_temp} RENAME TO {quoted_table}")


def ensure_activity_schema(conn: sqlite3.Connection) -> None:
    """Create only the tables owned by this crawler."""
    rename_legacy_activity_tables(conn)
    repair_activity_join_foreign_keys(conn)
    conn.executescript(ACTIVITY_SCHEMA_SQL)
    for migration in ACTIVITY_MIGRATIONS:
        try:
            conn.execute(migration)
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute(
            """
            UPDATE dep_act_questions_interpellations
            SET member_id = (
                SELECT member_id
                FROM member_questions_interpellations mq
                WHERE mq.question_id = dep_act_questions_interpellations.question_id
                LIMIT 1
            )
            WHERE member_id IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM member_questions_interpellations mq
                  WHERE mq.question_id = dep_act_questions_interpellations.question_id
              )
            """
        )
    except sqlite3.OperationalError:
        pass
    conn.execute("DROP TABLE IF EXISTS member_questions_interpellations")
    try:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_questions_interpellations_member_id
                ON dep_act_questions_interpellations(member_id)
            """
        )
    except sqlite3.OperationalError:
        pass


def require_members_present(
    conn: sqlite3.Connection,
    selected: list[dict[str, Any]],
) -> None:
    if not selected:
        return
    wanted = [member["member_id"] for member in selected]
    placeholders = ",".join("?" for _ in wanted)
    try:
        existing = {
            row[0]
            for row in conn.execute(
                f"SELECT member_id FROM members WHERE member_id IN ({placeholders})",
                wanted,
            ).fetchall()
        }
    except sqlite3.OperationalError as exc:
        if "no such table: members" in str(exc):
            raise ValueError(
                "The members table does not exist. Run the normal pipeline/member "
                "import first; crawl_deputy_activity.py does not write non-crawler tables."
            ) from exc
        raise

    missing = [member_id for member_id in wanted if member_id not in existing]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            "Missing member rows in members table: "
            f"{preview}{suffix}. Run the normal pipeline/member import first; "
            "crawl_deputy_activity.py does not write non-crawler tables."
        )


class TransientFetchError(RuntimeError):
    """
    Network-layer failure (DNS, refused, timed out, 5xx …). Separate from
    logical errors so the hydrator can avoid poisoning `initiators_parse_error`
    on a transient outage: the next run will see `initiators_text IS NULL`
    and retry the same law automatically.
    """


def _is_transient_network_exception(exc: BaseException) -> bool:
    if isinstance(exc, TransientFetchError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code is not None and 500 <= exc.code < 600
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            return True
        if isinstance(reason, OSError):
            return True
        text = str(reason).lower()
        return (
            "refused" in text
            or "timed out" in text
            or "timeout" in text
            or "temporarily" in text
            or "reset" in text
            or "unreachable" in text
            or "name or service not known" in text
            or "nodename nor servname" in text
        )
    return False


def _normalize_law_source_url(url: str) -> str:
    """
    Ensure a stored law `source_url` is directly fetchable.

    Historically, deputy-profile activity pages linked to CDEP law pages with
    href values that sometimes lost the `cam=` query parameter along the way
    (e.g. `upl_pck2015.proiect?idp=22204`). Without `cam=`, the CDEP handler
    serves an error page rather than the proiect fișă. All `law:cdep:*` rows
    are crawled from Camera Deputaților profiles so `cam=2` is the correct
    default.
    """
    if not url:
        return url
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return url
    host = (parsed.hostname or "").casefold()
    if not (host == "cdep.ro" or host.endswith(".cdep.ro")):
        return url
    if "upl_pck2015.proiect" not in parsed.path:
        return url
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if any(key.casefold() == "cam" for key, _ in query_pairs):
        return url
    query_pairs = [("cam", "2"), *query_pairs]
    new_query = urllib.parse.urlencode(query_pairs)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _safe_filename_component(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _law_initiator_pdf_cache_path(
    pdf_dir: Path,
    *,
    law_id: str,
    identifier: str | None,
) -> Path:
    cache_key = _safe_filename_component(identifier or law_id)
    return pdf_dir / f"initiators_{cache_key}.pdf"


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.stem + ".",
        suffix=path.suffix + ".tmp",
        delete=False,
    ) as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


class Fetcher:
    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 30,
        retries: int = 2,
        sleep_seconds: float = 0.5,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def fetch(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(min(self.sleep_seconds * (attempt + 1), 5.0))
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml",
                    },
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    charset = response.headers.get_content_charset() or "utf-8"
                    return body.decode(charset, errors="replace")
            except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def fetch_bytes(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(min(self.sleep_seconds * (attempt + 1), 5.0))
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/pdf,*/*",
                    },
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error


def validate_law_initiator_ocr_dependencies() -> None:
    missing: list[str] = []
    if shutil.which("tesseract") is None:
        missing.append("Tesseract OCR binary (`tesseract`)")
    if importlib.util.find_spec("pypdfium2") is None:
        missing.append("Python package `pypdfium2`")
    if importlib.util.find_spec("PIL") is None:
        missing.append("Python package `Pillow`")
    if missing:
        raise ValueError(
            "Law initiator OCR requires local dependencies: "
            + ", ".join(missing)
            + ". Install Python deps with `pip3 install -r requirements.txt` "
            "and install Tesseract locally, for example `brew install tesseract` "
            "on macOS. Romanian OCR is best with the `ron` language pack."
        )


def parse_motive_pdf_url(html_text: str, source_url: str) -> str | None:
    for row in _parse_rows(html_text):
        if "expunerea de motive" not in _fold(row.text):
            continue
        for link in row.links:
            href = (link.href or "").strip()
            if ".pdf" in href.casefold():
                return urllib.parse.urljoin(source_url, href)
    return None


def _hostname_is_senat_ro(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").casefold()
    return host == "senat.ro" or host.endswith(".senat.ro")


def _senat_abs_url(base: str, href: str) -> str:
    """Join a senat fișă href to the base URL, normalizing backslash relatives.

    The senat.ro fișă exposes the same PDF via two hrefs: an absolute
    path with forward slashes (`/legis/PDF/2026/26b134AD.PDF`) and a
    relative path with backslashes (`PDF\\2026\\26b134AD.pdf`). The
    latter is not a valid URL, so normalize backslashes to forward
    slashes and anchor under `/legis/` when the path starts with `PDF/`.
    """
    raw = html.unescape((href or "").strip())
    if "\\" in raw:
        normalized = raw.replace("\\", "/")
        if normalized.lower().startswith("pdf/"):
            normalized = "/legis/" + normalized
        return urllib.parse.urljoin(base, normalized)
    return urllib.parse.urljoin(base, raw)


def parse_senat_expunere_motive_pdf_url(html_text: str, source_url: str) -> str | None:
    """
    Senat fișă: *Expunerea de motive* uses EM in the filename,
    e.g. /legis/PDF/2026/26b131EM.PDF?nocache=true
    """
    base = source_url.strip()

    def abs_url(href: str) -> str:
        return _senat_abs_url(base, href)

    for row in _parse_rows(html_text):
        folded = _fold(row.text)
        if "expunere" in folded and "motive" in folded:
            for link in row.links:
                href = (link.href or "").strip()
                hc = href.casefold()
                if "em.pdf" in hc and "/legis/" in hc:
                    return abs_url(href)
            for link in row.links:
                href = (link.href or "").strip()
                ht = _fold(link.text)
                if ".pdf" in href.casefold() and "expunere" in ht and "motive" in ht:
                    return abs_url(href)
        for link in row.links:
            href = (link.href or "").strip()
            if not href or ".pdf" not in href.casefold():
                continue
            hc = href.casefold()
            link_fold = _fold(link.text)
            if "em.pdf" in hc and "/legis/" in hc:
                if "expunere" in link_fold and "motive" in link_fold:
                    return abs_url(href)

    for row in _parse_rows(html_text):
        for link in row.links:
            href = (link.href or "").strip()
            hc = href.casefold()
            if "em.pdf" in hc and "/legis/" in hc:
                return abs_url(href)

    match = re.search(
        r"""href\s*=\s*["']([^"']*?/legis/PDF/[^"']*?EM\.PDF[^"']*)["']""",
        html_text,
        flags=re.IGNORECASE,
    )
    if match:
        return abs_url(match.group(1))
    return None


def parse_senat_adresa_inaintare_pdf_url(
    html_text: str, source_url: str
) -> str | None:
    """
    Senat fișă: *adresă de înaintare a inițiativei legislative pentru dezbatere*
    (filename uses the `AD` code, e.g. `/legis/PDF/2026/26b134AD.PDF`).

    This document is the single best source for law initiators on senat.ro:
    it is a short (usually 1-page) cover letter addressed from the submitting
    chamber's Birou permanent to the other chamber, ending with
    `Inițiator,\nDeputat/Senator <NAME>` (or `Inițiatori,\nDeputat <NAME1>,\n
    Deputat <NAME2>` for multi-initiator bills). Because it is a formal
    cover letter, it does **not** embed the full list of susținători the EM
    PDF often mixes in, which is why the extractor wrongly promoted co-
    signers to initiators (see `_fuzzy_member_matches_initiator_lines`). We
    therefore prefer AD.PDF over EM.PDF whenever the senat law page exposes
    one.
    """
    base = source_url.strip()

    def abs_url(href: str) -> str:
        return _senat_abs_url(base, href)

    def is_ad_pdf_href(href: str) -> bool:
        hc = href.casefold()
        if ".pdf" not in hc:
            return False
        return bool(
            re.search(r"/legis/pdf/\d{4}/\d{2}b\d+ad\.pdf", hc)
            or re.search(r"pdf[\\/]\d{4}[\\/]\d{2}b\d+ad\.pdf", hc)
        )

    def label_looks_like_adresa(text: str) -> bool:
        f = _fold(text)
        return (
            "adresa" in f
            and ("inaintare" in f or "inainta" in f)
            and "initiativ" in f
            and "legislativ" in f
            and "dezbatere" in f
        )

    for row in _parse_rows(html_text):
        if label_looks_like_adresa(row.text):
            for link in row.links:
                href = (link.href or "").strip()
                if ".pdf" in href.casefold() and is_ad_pdf_href(href):
                    return abs_url(href)
            for link in row.links:
                href = (link.href or "").strip()
                if ".pdf" in href.casefold():
                    return abs_url(href)
        for link in row.links:
            href = (link.href or "").strip()
            if not href or ".pdf" not in href.casefold():
                continue
            if label_looks_like_adresa(link.text) and is_ad_pdf_href(href):
                return abs_url(href)

    for m in re.finditer(
        r"""<a\s+[^>]*href\s*=\s*["']([^"']+\.pdf[^"']*)["'][^>]*>(.*?)</a>""",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = m.group(1)
        inner = re.sub(r"<[^>]+>", " ", m.group(2))
        if is_ad_pdf_href(href) and label_looks_like_adresa(inner):
            return abs_url(href)

    match = re.search(
        r"""href\s*=\s*["']([^"']*?(?:/legis/PDF/|PDF[\\/])\d{4}[\\/]\d{2}b\d+AD\.PDF[^"']*)["']""",
        html_text,
        flags=re.IGNORECASE,
    )
    if match:
        return abs_url(match.group(1))
    return None


def parse_senat_forma_initiatorului_pdf_url(
    html_text: str, source_url: str
) -> str | None:
    """
    Senat fișă: *Forma inițiatorului* PDF (usually `<YY>b<NNN>FG.PDF` or similar).

    Only used as a last-resort fallback when there is no EM.PDF row on the
    page: the signature of the initiator sits at the bottom of this document
    too, and OCRing only the **last** pages of it (see `_extract_pdf_text_ocr`
    with `from_tail=True`) keeps the extractor focused on the signature and
    not on the full proposal body.
    """
    base = source_url.strip()

    def abs_url(href: str) -> str:
        return _senat_abs_url(base, href)

    for row in _parse_rows(html_text):
        folded = _fold(row.text)
        if "forma" in folded and "initiator" in folded:
            for link in row.links:
                href = (link.href or "").strip()
                if ".pdf" in href.casefold():
                    return abs_url(href)

    for row in _parse_rows(html_text):
        for link in row.links:
            href = (link.href or "").strip()
            link_fold = _fold(link.text)
            if ".pdf" in href.casefold() and "forma" in link_fold and "initiator" in link_fold:
                return abs_url(href)
    return None


def list_senat_law_initiator_pdf_urls(html_text: str, source_url: str) -> list[str]:
    """
    Senat: prefer *adresa de înaintare a inițiativei legislative pentru
    dezbatere* (AD.PDF), then *Expunerea de motive* (EM.PDF), finally
    *Forma inițiatorului* (FG.PDF).

    Returns the list of all candidate PDFs in priority order so the caller
    can try AD first and fall back to EM when AD yields no usable
    initiator section (some AD cover letters are just "vă înaintăm spre
    dezbatere" without naming the initiator). The per-PDF attempt loop in
    `hydrate_law_initiators_for_records` stops on the first PDF that
    matches at least one known member.
    """
    urls: list[str] = []
    ad = parse_senat_adresa_inaintare_pdf_url(html_text, source_url)
    if ad:
        urls.append(ad)
    em = parse_senat_expunere_motive_pdf_url(html_text, source_url)
    if em and em not in urls:
        urls.append(em)
    if not urls:
        fallback = parse_motive_pdf_url(html_text, source_url)
        if fallback:
            urls.append(fallback)
    if not urls:
        forma = parse_senat_forma_initiatorului_pdf_url(html_text, source_url)
        if forma:
            urls.append(forma)
    return urls


def parse_law_initiator_pdf_url(html_text: str, source_url: str) -> str | None:
    """
    Primary PDF URL stored on dep_act_laws.motive_pdf_url: *Expunerea de motive*
    (CDEP table link; Senat EM.PDF or same label via table parse).
    """
    if _hostname_is_senat_ro(source_url):
        urls = list_senat_law_initiator_pdf_urls(html_text, source_url)
        if urls:
            return urls[0]
    return parse_motive_pdf_url(html_text, source_url)


def _extract_pdf_text_direct(pdf_bytes: bytes) -> str:
    if importlib.util.find_spec("pypdf") is None:
        return ""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _run_tesseract(image_path: Path, language: str) -> str:
    language_candidates = [language]
    if language != "eng":
        language_candidates.append("eng")
    last_error = ""
    for candidate in language_candidates:
        completed = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "-l",
                candidate,
                "--psm",
                "6",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return completed.stdout
        last_error = completed.stderr.strip() or completed.stdout.strip()
    raise RuntimeError(last_error or "Tesseract OCR failed")


def _extract_pdf_text_ocr(
    pdf_bytes: bytes,
    *,
    language: str = DEFAULT_OCR_LANGUAGE,
    max_pages: int = DEFAULT_LAW_INITIATOR_OCR_PAGES,
    from_tail: bool = True,
) -> str:
    """
    OCR up to `max_pages` pages. When `from_tail=True` (default), OCRs the
    last N pages, because the initiator signature sits at the end of the
    *Expunerea de motive* / *Forma inițiatorului* PDF; OCRing the tail
    avoids drowning the signature in body text that would otherwise
    produce false-positive token matches against member names.
    """
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "document.pdf"
        pdf_path.write_bytes(pdf_bytes)
        document = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(document)
        if total_pages <= 0:
            return ""
        effective = min(total_pages, max(1, max_pages))
        if from_tail:
            page_range = range(total_pages - effective, total_pages)
        else:
            page_range = range(effective)
        texts: list[str] = []
        for index in page_range:
            page = document[index]
            image = page.render(scale=3).to_pil()
            image_path = tmp_path / f"page_{index + 1}.png"
            image.save(image_path)
            texts.append(_run_tesseract(image_path, language))
        return "\n".join(texts)



def extract_pdf_text_local(
    pdf_bytes: bytes,
    *,
    language: str = DEFAULT_OCR_LANGUAGE,
    max_pages: int = DEFAULT_LAW_INITIATOR_OCR_PAGES,
) -> str:
    direct_text = _extract_pdf_text_direct(pdf_bytes)
    cleaned = _clean_text(direct_text)
    if len(cleaned) < 80:
        return _extract_pdf_text_ocr(
            pdf_bytes,
            language=language,
            max_pages=max_pages,
            from_tail=True,
        )
    if _embedded_pdf_missing_initiator_name_signals(direct_text):
        return _extract_pdf_text_ocr(
            pdf_bytes,
            language=language,
            max_pages=max_pages,
            from_tail=True,
        )
    return direct_text


_INITIATOR_HEADER_RE = re.compile(
    r"(?iu)\bini(?:ț|t)iator(?:ilor|ii|i)?\b|\binitiator\b",
)

# Header for the supporters / signatures block; initiator names may appear just above it.
_SUSTINATORI_HEADER_RE = re.compile(
    r"(?iu)\bsus(?:ț|t)in(?:ă|â|a|ã)t(?:or|ori|orii|orilor)\b"
    r"|\bsustinator(?:i|ii|ilor)?\b",
)

_INITIATOR_BEFORE_SUSTINATORI_NOTE = (
    "[Inițiatori identificați în textul dinaintea tabelului susținători (posibil OCR manuscris):]"
)


def _initiator_name_signals_in_text(chunk: str) -> bool:
    """Heuristic: Camera table rows or Senat-style 'Deputat PARTY Nume' lines."""
    if re.search(
        r"(?iu)\b(deputat|senator)\s+\S+\s+[^\W\d_]\S*",
        chunk,
    ):
        return True
    # Printed lists use capitalized surnames; avoid matching OCR junk like "3 rrfu4".
    if re.search(
        r"(?u)(?<![0-9])\d{1,3}\s+(?-i:[A-Z])(?:[^\W\d_]|-){3,}",
        chunk,
    ):
        return True
    return False


def _embedded_pdf_missing_initiator_name_signals(direct_text: str) -> bool:
    """
    True when the PDF text layer is unlikely to contain a usable initiator block:
    header present but no name lines after it, or an Expunere de motive-style doc
    with body text embedded but no initiator header at all (header often image-only).
    """
    cleaned = _clean_text(direct_text)
    if len(cleaned) < 80:
        return True
    m = _INITIATOR_HEADER_RE.search(cleaned) or re.search(
        r"(?i)\binitiatori\b",
        cleaned,
    )
    if m:
        window = cleaned[m.end() : m.end() + 2600]
        if _initiator_name_signals_in_text(window):
            return False
        return True
    if len(cleaned) >= 500 and re.search(
        r"(?i)\bexpunere\s+a?\s*de\s+motive\b",
        cleaned,
    ):
        return True
    return False


# Party / group tokens at end of printed “Inițiatori” table rows (Camera Deputaților).
_PARTY_TOKENS_FOLDED = frozenset(
    {
        "psd",
        "pnl",
        "usr",
        "udmr",
        "aur",
        "per",
        "pmp",
        "plus",
        "reper",
        "forta",
        "fortã",
        "independent",
        "independenti",
        "neafiliat",
        "neafiliati",
        "minoritati",
        "minoritate",
    }
)

_INITIATOR_LINE_STOPWORDS_FOLDED = frozenset(
    _fold(
        w,
    )
    for w in (
        "expunere",
        "motive",
        "motivare",
        "semnatura",
        "semnaturi",
        "camera",
        "deputatilor",
        "senatului",
        "parlamentul",
        "romaniei",
        "hotarare",
        "lege",
        "proiect",
        "propunere",
        "legislativa",
        "nr",
        "pentru",
        "privind",
        "normativ",
        "fisa",
        "fișa",
        "act",
        "crt",
        "numele",
        "prenumele",
        "grupul",
        "parlamentar",
        "initiativa",
        "prezenta",
        "pagina",
        "pagină",
    )
)

_RE_NAMEISH_TOKEN = re.compile(
    r"(?u)^[^\W\d_](?:[^\W\d_]|[.\-]){0,29}$",
)

_HANDWRITING_SYMBOLS = frozenset("|{}[]()®©*+=<>/`´^~•_\\§@#%")

_HANDWRITING_CONSONANT_RUN = re.compile(
    r"(?i)[bcdfghjkmnpqstvwxz]{5,}",
)


def _case_chaos_in_word(tok: str) -> bool:
    """Handwritten OCR often alternates random case within a single token."""
    letters = [c for c in tok if c.isalpha()]
    if len(letters) < 5:
        return False
    flips = sum(
        1
        for i in range(len(letters) - 1)
        if letters[i].islower() != letters[i + 1].islower()
    )
    return flips >= 5


def _handwriting_like_ocr(text: str) -> bool:
    """True if this looks like cursive/signature OCR or noise, not printed form text."""
    t = _clean_text(text)
    if not t or len(t) > 100:
        return True
    sym = sum(1 for c in t if c in _HANDWRITING_SYMBOLS)
    if sym >= 2:
        return True
    if re.fullmatch(r"\d+\s*", t):
        return True
    # Digit glued to letters (common in messy OCR); skip lines that start with table index.
    if not re.match(r"^\d+\s+", t) and re.search(
        r"(?<=[^\W\d_])\d|(?<=\d)(?=[^\W\d_])",
        t,
    ):
        return True
    letters_only = "".join(c for c in t if c.isalpha())
    if len(letters_only) >= 8:
        vowels = sum(1 for c in _fold(letters_only) if c in "aeiou")
        if vowels / len(letters_only) < 0.12:
            return True
    if _HANDWRITING_CONSONANT_RUN.search(_fold(t)):
        return True
    return any(len(w) >= 5 and _case_chaos_in_word(w) for w in t.split())


def _tokens_romanian_printed_shape(cleaned: str) -> bool:
    """Printed lists use title-case / uppercase tokens, not cursive blobs."""
    tokens = cleaned.split()
    if not (2 <= len(tokens) <= 6):
        return False
    for tok in tokens:
        for part in tok.split("-"):
            if not part:
                return False
            if not part[0].isupper():
                return False
    return True


def _segment_looks_printed_typed(segment: str) -> bool:
    """Heuristic: likely machine-printed text rather than handwritten OCR noise."""
    if len(segment) < 3:
        return False
    letters = sum(1 for c in segment if c.isalpha())
    if letters < max(4, int(len(segment) * 0.5)):
        return False
    junk = sum(1 for c in segment if c in "|{}[]®©`´^~_•*")
    return junk <= 2


def _parse_initiator_table_numbered_row(line: str) -> str | None:
    """Parse rows like '1 Ciunt Ionel PSD' (printed names in initiator tables)."""
    raw = _clean_text(line)
    if not raw:
        return None
    tokens = raw.split()
    if len(tokens) < 3:
        return None
    if not tokens[0].isdigit():
        return None
    if _fold(tokens[1]) in ("deputat", "senator"):
        return None
    party_idx: int | None = None
    for i in range(len(tokens) - 1, 0, -1):
        if _fold(tokens[i]) in _PARTY_TOKENS_FOLDED:
            party_idx = i
            break
    if party_idx is None or party_idx < 2:
        return None
    name = " ".join(tokens[1:party_idx])
    if not _segment_looks_printed_typed(name):
        return None
    if _handwriting_like_ocr(name):
        return None
    return name


def _strip_leading_chamber_party_prefix(s: str) -> str:
    """Senat EM lines: 'Deputat AUR Nume Prenume' / 'Senator USR …'."""
    tokens = s.split()
    if len(tokens) < 3:
        return s
    if _fold(tokens[0]) not in ("deputat", "senator"):
        return s
    if _fold(tokens[1]) not in _PARTY_TOKENS_FOLDED:
        return s
    return " ".join(tokens[2:])


def _strip_initiator_line_roles(line: str) -> str:
    s = _clean_text(line).strip().lstrip(",;").strip()
    # Header glue from "INITIATORI:" when OCR merges the block into one line.
    s = re.sub(r"^[\s:.;,]+", "", s)
    s = re.sub(r"(?iu)^(?:inițiatori|initiatori|initiator)(?:\s*[:,])?\s*", "", s)
    s = s.split(",")[0].strip()
    s = re.sub(r"^\s*[\-|•.:]+\s*", "", s)
    s = re.sub(r"^\s*\d+[.)]\s*", "", s)
    s = re.sub(
        r"(?iu)^\d{1,3}\s+(?=(?:deputat|senator)\b)",
        "",
        s,
    )
    s = _strip_leading_chamber_party_prefix(s)
    # Trailing ", deputat …" / " popescu deputat …" (after name is already isolated).
    s = re.sub(
        r"(?iu)(?:\s*,\s*|\s+)(?:deputat|senator)\b.*$",
        "",
        s,
    ).strip()
    s = re.sub(r"(?iu)\s+[_|].*$", "", s)
    s = re.sub(r"(?iu)\s+\w\s*$", "", s)
    return s.strip()


def _line_looks_like_printed_deputy_name(line: str) -> bool:
    cleaned = _strip_initiator_line_roles(line)
    if not cleaned:
        return False
    folded_line = _fold(cleaned)
    for stop in _INITIATOR_LINE_STOPWORDS_FOLDED:
        if stop in folded_line:
            return False
    tokens = cleaned.split()
    if not (2 <= len(tokens) <= 6):
        return False
    letters = sum(c.isalpha() for c in cleaned)
    if letters < len(cleaned) * 0.55:
        return False
    digit_runs = len(re.findall(r"\d", cleaned))
    if digit_runs > 1:
        return False
    for t in tokens:
        if _fold(t) in _PARTY_TOKENS_FOLDED:
            return False
        if not _RE_NAMEISH_TOKEN.match(t):
            return False
    if _handwriting_like_ocr(cleaned):
        return False
    if not _tokens_romanian_printed_shape(cleaned):
        return False
    return True


def _split_ocr_merged_chamber_lines(line: str) -> list[str]:
    """Turn one OCR line with several 'Deputat/Senator PARTY …' entries into fragments."""
    raw = _clean_text(line)
    if not raw:
        return []
    parts = re.split(
        r"(?iu)(?<=[^\W\d_])\s+(?=(?:deputat|senator)\s+)",
        raw,
    )
    return [p.strip() for p in parts if p.strip()]


def _inject_newlines_before_numbered_table_rows(text: str) -> str:
    """OCR often yields one long line; table rows start with a short row index + name."""
    s = _clean_text(text).strip()
    if not s:
        return s
    # Guaranteed space before row indexes so (?<= ) is fixed-width for Python re.
    s = re.sub(
        r"(?<= )(?<![0-9])(\d{1,3})(?![0-9])\s+(?=[^\W\d_])",
        r"\n\1 ",
        f" {s} ",
    )
    return s.strip()


def _extract_printed_initiator_lines_from_block(body: str) -> list[str]:
    """Collect typed initiators only; handwritten supporter signatures must not match."""
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in _inject_newlines_before_numbered_table_rows(body).splitlines():
        line = _clean_text(raw_line)
        if not line:
            continue
        fragments = _split_ocr_merged_chamber_lines(line)
        for fragment in fragments:
            sub_parts = [p.strip() for p in re.split(r"\|", fragment) if p.strip()]
            pieces = sub_parts if len(sub_parts) > 1 else [fragment]
            for piece in pieces:
                from_row = _parse_initiator_table_numbered_row(piece)
                if from_row:
                    key = _fold(from_row)
                    if key not in seen:
                        seen.add(key)
                        out.append(from_row)
                    continue
                candidate = _strip_initiator_line_roles(piece)
                if not candidate or _handwriting_like_ocr(candidate):
                    continue
                if _line_looks_like_printed_deputy_name(candidate):
                    key = _fold(candidate)
                    if key not in seen:
                        seen.add(key)
                        out.append(candidate)
    return out


def _slice_before_sustinatori_header(text: str) -> str | None:
    """End initiator zone before susținători words or 'Lista susținătorilor' (OCR may mangle letters)."""
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    cuts: list[int] = []
    m = _SUSTINATORI_HEADER_RE.search(cleaned)
    if m:
        cuts.append(m.start())
    m2 = re.search(r"(?i)\blista\s+sust", cleaned)
    if m2:
        cuts.append(m2.start())
    if not cuts:
        return None
    return cleaned[: min(cuts)].rstrip()


def _parse_numbered_row_name_pre_sustinatori(line: str) -> str | None:
    """Number + name + party row; name may look handwritten — only used above susținători table."""
    raw = _clean_text(line)
    if not raw:
        return None
    tokens = raw.split()
    if len(tokens) < 3 or not tokens[0].isdigit():
        return None
    if _fold(tokens[1]) in ("deputat", "senator"):
        return None
    party_idx: int | None = None
    for i in range(len(tokens) - 1, 0, -1):
        if _fold(tokens[i]) in _PARTY_TOKENS_FOLDED:
            party_idx = i
            break
    if party_idx is None or party_idx < 2:
        return None
    name = " ".join(tokens[1:party_idx])
    letters = sum(c.isalpha() for c in name)
    if letters < 4 or letters < len(name) * 0.35:
        return None
    sym = sum(1 for c in name if c in _HANDWRITING_SYMBOLS)
    if sym >= 3:
        return None
    if _HANDWRITING_CONSONANT_RUN.search(_fold(name)):
        return None
    return name


def _line_looks_like_initiator_name_before_sustinatori(line: str) -> bool:
    """
    Name-shaped line in the zone before the susținători table.
    Allows OCR that would be rejected as handwriting elsewhere.
    """
    candidate = _strip_initiator_line_roles(line)
    if not candidate:
        return False
    folded_line = _fold(candidate)
    if "sustinator" in folded_line:
        return False
    for stop in _INITIATOR_LINE_STOPWORDS_FOLDED:
        if stop in folded_line:
            return False
    tokens = candidate.split()
    # Narrower than generic initiator lines: pre-susținători zone is usually a short name, not a sentence.
    if not (2 <= len(tokens) <= 6):
        return False
    letters = sum(c.isalpha() for c in candidate)
    if letters < max(5, int(len(candidate) * 0.4)):
        return False
    if len(candidate) > 120:
        return False
    sym = sum(1 for c in candidate if c in _HANDWRITING_SYMBOLS)
    if sym >= 4:
        return False
    if _HANDWRITING_CONSONANT_RUN.search(_fold(candidate)):
        return False
    digit_glued = not re.match(r"^\d+\s+", candidate) and re.search(
        r"(?<=[^\W\d_])\d|(?<=\d)(?=[^\W\d_])",
        candidate,
    )
    if digit_glued:
        return False
    for t in tokens:
        if _fold(t) in _PARTY_TOKENS_FOLDED:
            return False
        if sum(c.isalpha() for c in t) < 2:
            return False
    return True


def _pre_sustinatori_name_chunks(line: str) -> list[str]:
    """Split narrative + trailing name on '. ' so OCR paragraphs stay one line."""
    s = _clean_text(line)
    if not s:
        return []
    parts = re.split(r"\.\s+", s)
    return [p.strip() for p in parts if p.strip()]


def _extract_handwritten_initiators_before_sustinatori(
    block: str,
    *,
    exclude_fold: set[str],
) -> list[str]:
    """Lines strictly above *susținători*; relaxed rules so manual script OCR can match."""
    out: list[str] = []
    seen: set[str] = set(exclude_fold)
    blob = _inject_newlines_before_numbered_table_rows(block)
    for raw_line in blob.splitlines():
        chunks = _pre_sustinatori_name_chunks(raw_line)
        if not chunks:
            chunks = [_clean_text(raw_line)]
        for line in chunks:
            if not line:
                continue
            fold_line = _fold(line)
            if "sustinator" in fold_line:
                continue
            from_row = _parse_numbered_row_name_pre_sustinatori(line)
            if from_row:
                key = _fold(from_row)
                if key not in seen:
                    seen.add(key)
                    out.append(from_row)
                continue
            if _line_looks_like_initiator_name_before_sustinatori(line):
                cand = _strip_initiator_line_roles(line)
                key = _fold(cand)
                if key not in seen:
                    seen.add(key)
                    out.append(cand)
    return out


_PROSE_INITIATOR_PATTERNS = (
    re.compile(r"(?i)[îi]n\s+numele\s+ini[țt]?iatorilor[;:,]?\s*([^\n]+)"),
    re.compile(r"(?iu)\binitiator(?:ilor|ii|i)?\s*:\s*([^\n]+)"),
)

# Stop the prose-captured tail at the first token that clearly belongs to the
# next section (table headers, page footers, body-copy words). This keeps
# lines such as "Initiator: Cezar-Mihail Drăgoescu" intact while cutting off
# OCR-merged junk like "Cezar-Mihail Drăgoescu Lista iniSatorilor ...".
_PROSE_INITIATOR_TAIL_STOP_RE = re.compile(
    r"(?iu)\b("
    r"lista|propunere|propunerii|nr\b|pagina|fi[sș]a|expunere|motiv"
    r"|semn(?:ă|a)tur|circumscrip|grupul|parlamentar|camera|senatului"
    r"|romaniei|proiect|legislativ|comp\]?letar|for\s+bookmark"
    r")",
)

_NUMELE_INITIATORILOR_RE = re.compile(
    r"(?iu)(?:\bnumele\s+ini(?:ț|t)iatorilor\b"
    r"|\bnume[lL]e\s+ini(?:ț|t)?iatori(?:ilor|lor|ii)\b)",
)

_PROSE_IN_NUMELE_INITIATORILOR_RE = re.compile(
    r"(?iu)\b[îi]n\s+numele\s+ini(?:ț|t)iatorilor\b",
)

# Signature block at the very end of the *Expunerea de motive*:
#
#     Inițiator,
#     Deputat Silviu-Octavian Gurlui
#
# or on a single OCR line "Inițiator, Deputat Silviu-Octavian Gurlui". Required
# anchor when the PDF has no printed "Inițiatori:" header / table at all.
_SIGNATURE_INITIATOR_THEN_DEPUTAT_RE = re.compile(
    r"(?iu)\bini(?:ț|t)?iator[ii]?[,:.]?\s*[\n\r]?\s*"
    r"(?:Deputat|Senator)\s+[^\n\r]{3,80}",
)

# Signature block without the "Inițiator" keyword, of the form:
#
#     Deputat Raisa Enachi
#     Circumscripția electorală nr. 39 Vaslui
#
# The deputy's circumscription line is a near-unique marker of the EM
# signature block, so we anchor on it directly. Newlines are collapsed to
# spaces by `_clean_text` before matching, so we only require whitespace.
_SIGNATURE_DEPUTAT_CIRCUMSCRIPTIE_RE = re.compile(
    r"(?iu)\b(?:Deputat|Senator)\s+[^|;,]{3,80}?\s+"
    r"Circumscrip(?:ț|t)i(?:a|e)\s+electoral",
)

# Extractor helper: pull NAME out of "Deputat NAME" / "Senator NAME" in a
# signature block, but only when NAME is **not** followed by an all-caps
# SURNAME pattern (that shape is handled by _extract_surname_first_initiator_lines)
# and when NAME contains no party token (that shape is already handled by
# _extract_printed_initiator_lines_from_block via _strip_leading_chamber_party_prefix).
_SIGNATURE_DEPUTAT_NAME_RE = re.compile(
    r"(?iu)\b(?:Deputat|Senator)\s+([^\n\r,;|]{3,80}?)(?=\s*(?:[\n\r]|$|\|))",
)


def _initiator_header_tail_is_support_letter_prose(
    cleaned_full: str, match_end: int
) -> bool:
    """
    True when text right after a would-be initiator *word* header is body copy such as
    *Inițiatorii își exprimă susținerea pentru prezentul proiect…* — not signatures.
    """
    tail = _clean_text(cleaned_full[match_end : match_end + 240])
    if not tail:
        return False
    fold = _fold(tail)
    if "exprim" in fold:
        return True
    if "prezentul" in fold and "proiect" in fold:
        return True
    if "propunerii" in fold and "legislat" in fold:
        return True
    raw_l = tail.lower()
    if "ultsi" in raw_l and ("exprim" in raw_l or "sust" in fold):
        return True
    return False


def _initiator_header_followed_by_lista_without_name_signals(
    cleaned_full: str, match_end: int
) -> bool:
    """OCR may repeat 'în numele inițiatorilor' before Lista; skip if no deputat/senator line."""
    tail = _clean_text(cleaned_full[match_end : match_end + 220])
    f = _fold(tail)
    if "lista" not in f or "sust" not in f:
        return False
    return not _initiator_name_signals_in_text(tail[:200])


def _find_initiator_anchor(
    cleaned_full: str,
) -> tuple[re.Match[str] | None, str]:
    """
    Locate where the initiator block starts in the OCR text and report the
    provenance label that matched. Provenance is used to populate
    `dep_act_laws.initiators_source` and to help downstream debugging.

    Preference order:
      1. 'În numele inițiatorilor' prose header
      2. 'numele inițiatorilor' header (without 'În')
      3. 'Inițiatori:' / 'Initiator:' colon header
      4. Signature block: 'Inițiator,\\nDeputat NAME'
      5. Signature block: 'Deputat NAME\\nCircumscripția electorală ...'
      6. Generic 'Inițiator' word header followed by plausible name signals

    The 4–5 anchors are new: they catch short Expunere de motive signature
    blocks (one initiator per law) that have no list-of-initiators header.
    """
    for prose_in in reversed(list(_PROSE_IN_NUMELE_INITIATORILOR_RE.finditer(cleaned_full))):
        if _initiator_header_tail_is_support_letter_prose(cleaned_full, prose_in.end()):
            continue
        if _initiator_header_followed_by_lista_without_name_signals(
            cleaned_full, prose_in.end()
        ):
            continue
        return prose_in, "em_in_numele_initiatorilor"
    for nm in reversed(list(_NUMELE_INITIATORILOR_RE.finditer(cleaned_full))):
        if _initiator_header_tail_is_support_letter_prose(cleaned_full, nm.end()):
            continue
        if _initiator_header_followed_by_lista_without_name_signals(cleaned_full, nm.end()):
            continue
        return nm, "em_numele_initiatorilor"
    for rx in (
        re.compile(r"(?iu)\bini(?:ț|t)iatori\s*:"),
        re.compile(r"(?iu)\binitiator\s*:"),
    ):
        m = rx.search(cleaned_full)
        if m:
            return m, "em_initiatori_colon"
    # Signature-block anchors (new). Prefer the very last occurrence so the
    # match points at the signature at the bottom of the document, not at a
    # "Deputat NAME" mention in the body.
    sig1_matches = list(_SIGNATURE_INITIATOR_THEN_DEPUTAT_RE.finditer(cleaned_full))
    if sig1_matches:
        return sig1_matches[-1], "em_signature_initiator_deputat"
    sig2_matches = list(_SIGNATURE_DEPUTAT_CIRCUMSCRIPTIE_RE.finditer(cleaned_full))
    if sig2_matches:
        return sig2_matches[-1], "em_signature_deputat_circumscriptie"
    best_with_signals: re.Match[str] | None = None
    best_sig_start = -1
    fallback: re.Match[str] | None = None
    fallback_start = -1
    for cand in _INITIATOR_HEADER_RE.finditer(cleaned_full):
        start = cand.start()
        prefix = cleaned_full[max(0, start - 8) : start]
        if re.search(r"(?i)numele\s+$", prefix):
            continue
        if _initiator_header_tail_is_support_letter_prose(cleaned_full, cand.end()):
            continue
        if _initiator_header_followed_by_lista_without_name_signals(cleaned_full, cand.end()):
            continue
        window = cleaned_full[cand.end() : cand.end() + 1400]
        if _initiator_name_signals_in_text(window):
            if start > best_sig_start:
                best_with_signals = cand
                best_sig_start = start
        else:
            if start > fallback_start:
                fallback = cand
                fallback_start = start
    if best_with_signals:
        return best_with_signals, "em_initiator_word_with_names"
    if fallback:
        return fallback, "em_initiator_word"
    return None, ""


def _find_initiator_header_match(cleaned_full: str) -> re.Match[str] | None:
    """Back-compat shim that returns only the match (drops provenance label)."""
    match, _ = _find_initiator_anchor(cleaned_full)
    return match


def _repair_ocr_glued_caps_surname_before_given(line: str) -> str:
    """Insert a space when OCR glues ALL-CAPS surname to a capitalised given name."""
    caps = "A-Z\u0102\u00c2\u00ce\u0218\u021a\u015e\u0162"
    lower = "a-z\u0103\u00e2\u00ee\u0219\u021b\u015f\u0163"
    pat = rf"(?u)([{caps}]{{2,}})([{caps}][{lower}][^\s;|]*)"
    return re.sub(pat, r"\1 \2", line)


def _line_looks_like_surname_first_initiator(line: str) -> bool:
    """Senat block: SURNAME Given-Names (surname often all caps), e.g. RUS Vasile-Ciprian."""
    raw = _clean_text(line)
    if not raw or _handwriting_like_ocr(raw):
        return False
    fold_line = _fold(raw)
    if "pagina" in fold_line:
        return False
    if re.search(r"\d\s*din\s*\d", fold_line):
        return False
    tokens = raw.split()
    if not (2 <= len(tokens) <= 8):
        return False
    chamberish = sum(1 for t in tokens if _fold(t) in ("deputat", "senator"))
    if chamberish >=2 or (chamberish == 1 and len(tokens) <= 2):
        return False
    if _fold(tokens[0]) in ("deputat", "senator"):
        tokens = tokens[1:]
        if len(tokens) < 2:
            return False
    for stop in _INITIATOR_LINE_STOPWORDS_FOLDED:
        if stop in fold_line:
            return False
    if any(_fold(t) in _PARTY_TOKENS_FOLDED for t in tokens):
        return False
    sur = "".join(c for c in tokens[0] if c.isalpha())
    if len(sur) < 2 or not sur.isupper():
        return False
    for t in tokens[1:]:
        if not _RE_NAMEISH_TOKEN.match(t):
            return False
    letters = sum(c.isalpha() for c in raw)
    if letters < max(5, int(len(raw) * 0.5)):
        return False
    return True


def _extract_surname_first_initiator_lines(block: str) -> list[str]:
    """Names from 'RUS Vasile-Ciprian ; BERESCU Monica-Elena' / two-column OCR glue."""
    out: list[str] = []
    seen: set[str] = set()
    blob = _inject_newlines_before_numbered_table_rows(block)
    for raw_line in blob.splitlines():
        line = _repair_ocr_glued_caps_surname_before_given(_clean_text(raw_line))
        for piece in re.split(r"\s*;\s*", line):
            frag = _clean_text(piece).lstrip(",; ")
            if not frag:
                continue
            if not _line_looks_like_surname_first_initiator(frag):
                continue
            key = _fold(frag)
            if key in seen:
                continue
            seen.add(key)
            out.append(frag)
    return out


def _truncate_prose_chunk_at_stop(chunk: str) -> str:
    """
    OCR often glues the signature line and the next-page header onto a single
    line, e.g. "Cezar-Mihail Drăgoescu Lista iniSatornor la propunerea ...".
    Truncate at the first word that clearly belongs to the next section so
    downstream name-shape checks see only the signature fragment.
    """
    cleaned = _clean_text(chunk)
    if not cleaned:
        return cleaned
    stop = _PROSE_INITIATOR_TAIL_STOP_RE.search(cleaned)
    if not stop:
        return cleaned
    head = cleaned[: stop.start()].strip().rstrip(",;.")
    return head


def _extract_prose_initiator_names(section: str) -> list[str]:
    """Typed intro lines such as 'În numele initiatorilor, Nume …'."""
    out: list[str] = []
    seen: set[str] = set()
    for pat in _PROSE_INITIATOR_PATTERNS:
        for m in pat.finditer(section):
            raw_chunk = m.group(1).strip()
            truncated = _truncate_prose_chunk_at_stop(raw_chunk)
            if not truncated:
                continue
            cleaned = _strip_initiator_line_roles(truncated)
            if not cleaned or _handwriting_like_ocr(cleaned):
                continue
            if _line_looks_like_printed_deputy_name(truncated):
                key = _fold(cleaned)
                if key not in seen:
                    seen.add(key)
                    out.append(cleaned)
    return out


def _extract_signature_deputat_names(section: str) -> list[str]:
    """
    Extract 'NAME' from `Deputat NAME` / `Senator NAME` signature lines, used
    when the EM ends with a single-initiator signature block like:

        Inițiator,
        Deputat Silviu-Octavian Gurlui

    or

        Deputat Raisa Enachi
        Circumscripția electorală nr. 39 Vaslui

    Deliberately ignores the party-prefixed variant (`Deputat AUR Name`, which
    `_extract_printed_initiator_lines_from_block` already handles) and the
    all-caps surname-first variant (`Deputat BERESCU Monica`, handled by
    `_extract_surname_first_initiator_lines`), to avoid duplicate output.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _SIGNATURE_DEPUTAT_NAME_RE.finditer(section):
        candidate = _clean_text(m.group(1))
        candidate = re.sub(r"^\s*\d+[.)]?\s*", "", candidate)
        if not candidate or _handwriting_like_ocr(candidate):
            continue
        tokens = candidate.split()
        if not (2 <= len(tokens) <= 6):
            continue
        if any(_fold(t) in _PARTY_TOKENS_FOLDED for t in tokens):
            continue
        alpha_first = "".join(c for c in tokens[0] if c.isalpha())
        if len(alpha_first) >= 3 and alpha_first.isupper():
            continue
        for stop_word in _INITIATOR_LINE_STOPWORDS_FOLDED:
            if stop_word in _fold(candidate):
                candidate = ""
                break
        if not candidate:
            continue
        if not _tokens_romanian_printed_shape(candidate):
            continue
        key = _fold(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _normalize_romanian_diacritics(text: str) -> str:
    """
    Replace the cedilla forms (`ţ`/`Ţ`, `ş`/`Ş`) with the modern comma-below
    forms (`ț`/`Ț`, `ș`/`Ș`). All anchor regexes target the modern forms, and
    this is a pure 1:1 character substitution so match offsets remain valid
    in the original `cleaned_full` string.
    """
    if not text:
        return text
    return (
        text.replace("ţ", "ț")
        .replace("Ţ", "Ț")
        .replace("ş", "ș")
        .replace("Ş", "Ș")
    )


def extract_initiators_section_with_source(text: str) -> tuple[str, str]:
    """
    Return (snippet, source_label).

    When no initiator anchor is found anywhere in the PDF text, we return
    `("", "none")` instead of the first few thousand characters of body:
    matching member names against Expunerea-de-motive body text produces
    rampant false positives (a 4000-char body typically token-matches
    10+ member names by accident). An empty snippet causes
    `match_initiator_member_ids` to short-circuit to no matches.
    """
    cleaned_full = _normalize_romanian_diacritics(_clean_text(text))
    match, source = _find_initiator_anchor(cleaned_full)
    if not match:
        return "", "none"
    span_end = min(len(cleaned_full), match.start() + 15000)
    section = cleaned_full[match.start() : span_end]
    end_match = re.search(
        r"(?i)\bexpunere(?:a)?\s+de\s+motive\b|\bmotivare\b",
        section[40:],
    )
    if end_match:
        section = section[: 40 + end_match.start()]
    header_end = match.end() - match.start()
    body = section[header_end:] if header_end < len(section) else ""
    pre_sustinatori = _slice_before_sustinatori_header(body)
    body_for_printed_table = (
        pre_sustinatori if pre_sustinatori is not None else body
    )
    table_and_lines = _extract_printed_initiator_lines_from_block(
        body_for_printed_table
    )
    prose = _extract_prose_initiator_names(section)
    surname_first = _extract_surname_first_initiator_lines(body_for_printed_table)
    # New extractor for single-signature EM blocks: the whole matched section
    # *is* the signature line in these cases, so we also scan it (in addition
    # to the pre-susținători slice) to pick up "Deputat NAME".
    signature_names = _extract_signature_deputat_names(section)
    merged: list[str] = []
    seen_fold: set[str] = set()
    for label in table_and_lines + prose + surname_first + signature_names:
        k = _fold(label)
        if k in seen_fold:
            continue
        seen_fold.add(k)
        merged.append(label)
    handwritten_before_sus: list[str] = []
    if pre_sustinatori is not None:
        handwritten_before_sus = _extract_handwritten_initiators_before_sustinatori(
            pre_sustinatori,
            exclude_fold=set(seen_fold),
        )
        for label in handwritten_before_sus:
            seen_fold.add(_fold(label))

    header = match.group(0).strip()
    lines_out: list[str] = []
    if merged:
        lines_out.append(header)
        lines_out.extend(merged)
    if handwritten_before_sus:
        lines_out.append(_INITIATOR_BEFORE_SUSTINATORI_NOTE)
        lines_out.extend(handwritten_before_sus)
    if lines_out:
        if not merged and handwritten_before_sus:
            lines_out.insert(0, header)
        return (
            "\n".join(_clean_text(line) for line in lines_out if _clean_text(line)),
            source,
        )
    # Anchor found but none of the per-shape extractors produced a clean name
    # line. Fall back to the raw section text (already trimmed to the anchor
    # → "Expunere de motive" window); matching against this narrow slice is
    # still safe because it never contains body prose — Fix 1 only forbids
    # falling back to body text when **no anchor** is found at all.
    return _clean_text(section), source


def extract_initiators_section(text: str) -> str:
    """Back-compat wrapper: return only the snippet string."""
    snippet, _ = extract_initiators_section_with_source(text)
    return snippet


def _name_tokens(value: str) -> set[str]:
    ignored = {"deputat", "deputati", "senator", "senatori", "domnul", "doamna"}
    ignored |= _PARTY_TOKENS_FOLDED
    tokens = set(re.findall(r"[a-z0-9]+", _fold(value)))
    return {token for token in tokens if len(token) > 1 and token not in ignored}


_INITIATOR_LOG_HEADER_LINE_RE = re.compile(
    r"(?iu)^(ini(?:ț|t)iator(?:ilor|ii|i)?|initiator)\s*:?\s*$"
    r"|^în\s+numele\s+ini(?:ț|t)iatorilor[,]?\s*$"
    r"|^numele\s+ini(?:ț|t)iatorilor[,]?\s*$",
)


def _initiator_snippet_log_name_lines(initiators_text: str) -> list[str]:
    """Non-header, non-note lines from extract_initiators_section output (for logging)."""
    out: list[str] = []
    note_fold = _fold(_INITIATOR_BEFORE_SUSTINATORI_NOTE)
    for raw in (initiators_text or "").splitlines():
        line = _clean_text(raw)
        if not line:
            continue
        if line.startswith("["):
            continue
        if _fold(line) == note_fold:
            continue
        if _INITIATOR_LOG_HEADER_LINE_RE.match(line):
            continue
        out.append(line)
    return out


def _truncate_log_fragment(value: str, max_len: int = 220) -> str:
    s = _clean_text(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def load_deputy_member_match_rows(
    conn: sqlite3.Connection,
) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2] or ""))
        for row in conn.execute(
            """
            SELECT member_id, name, normalized_name
            FROM members
            WHERE chamber = 'deputat'
            """
        ).fetchall()
    ]


def load_senator_member_match_rows(
    conn: sqlite3.Connection,
) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2] or ""))
        for row in conn.execute(
            """
            SELECT member_id, name, normalized_name
            FROM members
            WHERE chamber = 'senator'
            """
        ).fetchall()
    ]


def _law_initiator_match_log_suffix(
    initiators_text: str,
    *,
    matched_member_ids: list[str],
    deputy_rows: list[tuple[str, str, str]],
    senator_rows: list[tuple[str, str, str]],
) -> str:
    """Chamber breakdown when matches exist; otherwise explain zero matches."""
    dep_set = {r[0] for r in deputy_rows}
    sen_set = {r[0] for r in senator_rows}
    if matched_member_ids:
        n_dep = sum(1 for m in matched_member_ids if m in dep_set)
        n_sen = sum(1 for m in matched_member_ids if m in sen_set)
        return f" ({n_dep} deputati, {n_sen} senatori)"
    stripped = _clean_text(initiators_text)
    if not stripped:
        return " [why: no initiator snippet extracted from PDF]"
    name_lines = _initiator_snippet_log_name_lines(initiators_text)
    token_n = len(_name_tokens(initiators_text))
    if not name_lines:
        if token_n < 2:
            return (
                " [why: no initiator names found in extracted text "
                f"(name-like tokens={token_n}; likely OCR/layout or wrong section)]"
            )
        return (
            " [why: no initiator name lines after header in snippet "
            f"(name-like tokens={token_n}; text may be narrative or header-only)]"
        )
    preview = "; ".join(_truncate_log_fragment(line, 72) for line in name_lines[:8])
    preview = _truncate_log_fragment(preview, 220)
    return (
        " [why: extracted initiator name(s) not matched to any deputat/senator "
        f"in members table: {preview}]"
    )


def _fuzzy_member_matches_initiator_lines(
    lines_fold: list[str], variant_fold: str
) -> bool:
    """OCR-tolerant match for a member name against initiator snippet lines."""
    if not variant_fold or len(variant_fold) < 5:
        return False
    blob = " ".join(lines_fold)
    if variant_fold in blob:
        return True
    for lf in lines_fold:
        if len(lf) < 6:
            continue
        sm = difflib.SequenceMatcher(None, variant_fold, lf)
        if len(variant_fold) >= 12 and len(lf) >= 12 and sm.ratio() >= 0.78:
            return True
        if min(len(variant_fold), len(lf)) >= 8 and sm.ratio() >= 0.84:
            return True
    # Split on whitespace AND on hyphens, separately tracking the
    # member's surname (first whitespace-separated token). Romanian
    # compound given names like `Remus-Gabriel`, `Doru-Lucian`,
    # `Augustin-Florin` frequently appear in the OCR text with only one
    # half captured (e.g. `Remus Negoi` for `NEGOI Eugen-Remus`,
    # `Lapusan Remus` for `Lăpuşan Remus-Gabriel`). Treating the
    # compound as separate sub-tokens lets us require a majority of
    # pieces to match. Tracking the surname separately lets us reject
    # matches where the surname is absent — otherwise two different
    # members sharing a compound given name (e.g. `Lăpuşan Remus-
    # Gabriel` and `Mihalcea Remus-Gabriel`) would both match the same
    # line "lapusan remus-gabriel".
    def _keep(sub: str) -> bool:
        return (
            len(sub) >= 4
            and sub not in _PARTY_TOKENS_FOLDED
            and sub not in ("deputat", "senator", "domnul", "doamna")
        )

    def _keep_surname(sub: str) -> bool:
        # Accept short surnames (e.g. "Bem") that the generic length
        # filter would drop. We still filter party/role words because
        # a surname token can never be one of those. Without this, any
        # member whose surname has fewer than 4 letters would fall
        # through the per-line matcher and match on their given name
        # alone — e.g. `BEM Cristian` would match an AD signature
        # `Șipoș Cristian`, because `bem` is filtered, leaving only
        # `cristian`, and the surname gate trivially passes with an
        # empty surname set.
        return (
            len(sub) >= 2
            and sub not in _PARTY_TOKENS_FOLDED
            and sub not in ("deputat", "senator", "domnul", "doamna")
        )

    whitespace_tokens = variant_fold.split()
    if not whitespace_tokens:
        return False
    surname_subparts = [s for s in whitespace_tokens[0].split("-") if _keep_surname(s)]
    given_subparts: list[str] = []
    for token in whitespace_tokens[1:]:
        given_subparts.extend(s for s in token.split("-") if _keep(s))
    parts = list(surname_subparts) + given_subparts
    if not parts:
        return False
    surname_set = set(surname_subparts)
    # For 1- and 2-subtoken names every piece must match. For 3+
    # subtokens allow one missing piece to tolerate OCR-dropped middle
    # names / compound-half drops (e.g. `NEGOI Eugen-Remus` → `Remus
    # Negoi`: hits `negoi` + `remus`, `eugen` missing, 2/3). Being
    # strict for 2-subtoken names is what prevents the shared-given-
    # name false positives (`Nassar/Cușnir Rodica` vs `Rodica Plopeanu`).
    required = len(parts) if len(parts) <= 2 else len(parts) - 1
    # Count hits **per line** (one person per line in a multi-initiator
    # list): this prevents sub-tokens from different people on
    # different lines from combining into a spurious match. Require the
    # surname specifically to be one of the matched sub-tokens;
    # otherwise members sharing only a compound given name with the
    # real initiator would still pass.
    def _token_fuzzy_hit(piece: str, line: str) -> bool:
        if piece in line:
            return True
        for w in re.findall(r"[a-zăâîșț0-9-]+", line):
            if len(w) < 4:
                continue
            if difflib.SequenceMatcher(None, piece, w).ratio() >= 0.88:
                return True
        return False

    for lf in lines_fold:
        line_hits = 0
        surname_matched = not surname_set
        for p in parts:
            if _token_fuzzy_hit(p, lf):
                line_hits += 1
                if p in surname_set:
                    surname_matched = True
        if line_hits >= required and surname_matched:
            return True
    return False


def match_initiator_member_ids(
    initiators_text: str,
    member_rows: list[tuple[str, str, str]],
) -> list[str]:
    # Fix 1 short-circuit: if the extractor returned nothing, never fall back
    # to matching member names against raw PDF body text — that path turns
    # every deputy whose given+family names happen to appear anywhere in the
    # Expunere de motive into an "initiator".
    if not _clean_text(initiators_text):
        return []
    text_tokens = _name_tokens(initiators_text)
    folded_text = _fold(initiators_text)
    matched: list[str] = []
    matched_ids: set[str] = set()
    for member_id, name, normalized_name in member_rows:
        variants = [name, normalized_name]
        token_sets = [_name_tokens(variant) for variant in variants if variant]
        if any(tokens and tokens <= text_tokens for tokens in token_sets):
            matched.append(member_id)
            matched_ids.add(member_id)
            continue
        if any(_fold(variant) and _fold(variant) in folded_text for variant in variants):
            matched.append(member_id)
            matched_ids.add(member_id)
    lines_fold = [_fold(ln) for ln in _initiator_snippet_log_name_lines(initiators_text)]
    for member_id, name, normalized_name in member_rows:
        if member_id in matched_ids:
            continue
        for variant in (name, normalized_name):
            vf = _fold(variant) if variant else ""
            if vf and _fuzzy_member_matches_initiator_lines(lines_fold, vf):
                matched.append(member_id)
                matched_ids.add(member_id)
                break
    return matched


def _member_fuzzy_variants(name: str, normalized_name: str) -> list[str]:
    """Build name variants used by the single-signature fuzzy fallback."""
    source = _fold(normalized_name) or _fold(name)
    if not source:
        return []
    parts = source.split()
    if not parts:
        return []
    variants: set[str] = {source}
    if len(parts) >= 1:
        variants.add(parts[0])
        variants.add(parts[-1])
    if len(parts) >= 2:
        variants.add(" ".join(parts[:2]))
        variants.add(" ".join(parts[-2:]))
        variants.add(f"{parts[0]}-{parts[1]}")
        variants.add(f"{parts[-2]}-{parts[-1]}")
    return [v for v in variants if v and len(v) >= 5]


def _signature_window_token_clusters(window: str) -> list[str]:
    """Return 1-, 2- and 3-token clusters from the folded signature window."""
    folded = _fold(window)
    if not folded:
        return []
    tokens = re.findall(r"[a-z0-9ăâîșț-]+", folded)
    clusters: set[str] = set()
    for token in tokens:
        if len(token) >= 4:
            clusters.add(token)
    for i in range(len(tokens) - 1):
        clusters.add(f"{tokens[i]} {tokens[i + 1]}")
    for i in range(len(tokens) - 2):
        clusters.add(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")
    return [c for c in clusters if len(c) >= 5]


def single_signature_fuzzy_match(
    signature_window: str,
    member_rows: list[tuple[str, str, str]],
    *,
    threshold: float = 0.85,
) -> list[str]:
    """
    Fix 6: one-and-only-one fuzzy fallback for short single-initiator signature
    windows. Intended for snippets of ≲300 characters extracted from around an
    `Inițiator:` anchor where the normal extractors (which require a full
    printed name) produced nothing — e.g. because OCR dropped the surname or
    mangled the given name into `Cezar-rmail`.

    We compare member name variants (full, given-only, surname-only,
    two-token shapes, and a hyphen-joined given-name shape) against every
    1-/2-/3-token cluster of the signature window, using difflib ratios.
    We only return a match when **exactly one** member clears the threshold;
    otherwise we return nothing so we never turn a noisy OCR blob into a
    plausible but incorrect initiator attribution.
    """
    window = _clean_text(signature_window)
    if not window or len(window) > 1200:
        return []
    clusters = _signature_window_token_clusters(window)
    if not clusters:
        return []

    strong: list[tuple[str, float]] = []
    for member_id, name, normalized_name in member_rows:
        variants = _member_fuzzy_variants(name, normalized_name)
        if not variants:
            continue
        best = 0.0
        for variant in variants:
            for cluster in clusters:
                ratio = difflib.SequenceMatcher(None, variant, cluster).ratio()
                if ratio > best:
                    best = ratio
                    if best >= 0.999:
                        break
            if best >= 0.999:
                break
        if best >= threshold:
            strong.append((member_id, best))

    if len(strong) != 1:
        return []
    return [strong[0][0]]


def _record_law_initiator_error(
    conn: sqlite3.Connection,
    *,
    law_id: str,
    motive_pdf_url: str | None,
    error: str,
) -> None:
    conn.execute(
        """
        UPDATE dep_act_laws
        SET motive_pdf_url = COALESCE(?, motive_pdf_url),
            initiators_parse_error = ?,
            initiators_extracted_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE law_id = ?
        """,
        (motive_pdf_url, error, law_id),
    )


def _extract_and_match_law_initiators_from_pdf_bytes(
    pdf_bytes: bytes,
    *,
    combined_member_rows: list[tuple[str, str, str]],
    ocr_language: str,
    ocr_max_pages: int,
) -> tuple[str, str, list[str], str | None]:
    text_part = extract_pdf_text_local(
        pdf_bytes,
        language=ocr_language,
        max_pages=ocr_max_pages,
    )
    snippet, src = extract_initiators_section_with_source(text_part)
    ids = match_initiator_member_ids(snippet, combined_member_rows)
    fuzzy_id: str | None = None
    if not ids and src != "none":
        cleaned_full = _normalize_romanian_diacritics(_clean_text(text_part))
        anchor_m, _ = _find_initiator_anchor(cleaned_full)
        if anchor_m is not None:
            window_start = anchor_m.start()
            window = cleaned_full[window_start : window_start + 300]
            fuzzy_hits = single_signature_fuzzy_match(
                window,
                combined_member_rows,
            )
            if fuzzy_hits:
                ids = fuzzy_hits
                fuzzy_id = fuzzy_hits[0]
                src = src + "+fuzzy_single_signature"
                if not snippet:
                    snippet = f"[fuzzy fallback on signature window]\n{window}"
    return snippet, src, ids, fuzzy_id


def hydrate_law_initiators_for_records(
    conn: sqlite3.Connection,
    *,
    records: list[ListingRecord],
    fetcher: Fetcher,
    member_rows: list[tuple[str, str, str]],
    force: bool = False,
    skip_if_initiator_linked: bool = True,
    hydrated_law_ids: set[str] | None = None,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ocr_max_pages: int = DEFAULT_LAW_INITIATOR_OCR_PAGES,
    debug: bool = False,
    progress_prefix: str = "",
    commit_each: bool = False,
    senator_member_rows: list[tuple[str, str, str]] | None = None,
    law_initiator_pdf_dir: Path = DEFAULT_LAW_INITIATOR_PDF_DIR,
    per_fetch_sleep_seconds: float = 0.0,
    consecutive_failure_pause_after: int = 0,
    consecutive_failure_pause_seconds: float = 0.0,
    consecutive_failure_abort_after: int = 0,
    sleep_fn: Any | None = None,
) -> StoreResult:
    # `sleep_fn` defaults to `time.sleep`; it is injectable so tests can
    # avoid real waits while still verifying throttle/pause behaviour.
    sleeper = sleep_fn if sleep_fn is not None else time.sleep
    result = StoreResult(seen=len(records))
    prefix = f"{progress_prefix} " if progress_prefix else ""
    if senator_member_rows is None:
        senator_member_rows = load_senator_member_match_rows(conn)
    consecutive_transient_failures = 0
    abort_reason: str | None = None
    for index, record in enumerate(records, start=1):
        if index > 1:
            print(flush=True)
        log_context = f"{record.record_id} page={record.source_url}"
        row = conn.execute(
            """
            SELECT law_id, motive_pdf_url, initiators_text, initiators_parse_error
            FROM dep_act_laws
            WHERE law_id = ? OR source_url = ?
            ORDER BY CASE WHEN source_url = ? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (record.record_id, record.source_url, record.source_url),
        ).fetchone()
        if not row:
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} not stored yet, skipping",
                flush=True,
            )
            continue
        law_id = str(row[0])
        log_context = f"{law_id} page={record.source_url}"
        if hydrated_law_ids is not None and law_id in hydrated_law_ids:
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} already checked in this run, skipping",
                flush=True,
            )
            continue
        if skip_if_initiator_linked and conn.execute(
            """
            SELECT 1 FROM dep_act_member_laws
            WHERE law_id = ? AND is_initiator = 1
            LIMIT 1
            """,
            (law_id,),
        ).fetchone():
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} skipping (already has is_initiator=1 in dep_act_member_laws)",
                flush=True,
            )
            if hydrated_law_ids is not None:
                hydrated_law_ids.add(law_id)
            continue
        if not force and (row[2] is not None or row[3] is not None):
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} already hydrated, skipping",
                flush=True,
            )
            if hydrated_law_ids is not None:
                hydrated_law_ids.add(law_id)
            continue
        motive_pdf_url: str | None = str(row[1]) if row[1] else None
        source_url = _normalize_law_source_url(record.source_url)
        if source_url != record.source_url and debug:
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} normalized source_url to {source_url}",
                flush=True,
            )
        transient_error_this_iter = False
        try:
            combined_member_rows = list(member_rows) + list(senator_member_rows)
            total_bytes = 0
            member_ids: list[str] = []
            initiators_text: str = ""
            initiators_source: str = "none"
            fuzzy_member_id: str | None = None
            tried_pdfs = 0
            pdf_urls_debug: list[str] = []
            cache_path = _law_initiator_pdf_cache_path(
                law_initiator_pdf_dir,
                law_id=law_id,
                identifier=record.identifier,
            )

            if cache_path.is_file():
                pdf_bytes = cache_path.read_bytes()
                total_bytes = len(pdf_bytes)
                tried_pdfs = 1
                pdf_urls_debug = [f"file:{cache_path}"]
                print(
                    f"{prefix}Law initiators OCR {index}/{len(records)}: "
                    f"{log_context} using cached initiator PDF {cache_path}",
                    flush=True,
                )
                print(
                    f"{prefix}Law initiators OCR {index}/{len(records)}: "
                    f"{log_context} OCR/text extraction on cached PDF started "
                    f"({len(pdf_bytes)} bytes, last {ocr_max_pages} page(s))",
                    flush=True,
                )
                (
                    initiators_text,
                    initiators_source,
                    member_ids,
                    fuzzy_member_id,
                ) = _extract_and_match_law_initiators_from_pdf_bytes(
                    pdf_bytes,
                    combined_member_rows=combined_member_rows,
                    ocr_language=ocr_language,
                    ocr_max_pages=ocr_max_pages,
                )
            else:
                print(
                    f"{prefix}Law initiators OCR {index}/{len(records)}: "
                    f"{log_context} fetching law page",
                    flush=True,
                )
                law_html = fetcher.fetch(source_url)
                if _hostname_is_senat_ro(source_url):
                    pdf_urls = list_senat_law_initiator_pdf_urls(law_html, source_url)
                else:
                    single = parse_law_initiator_pdf_url(law_html, source_url)
                    pdf_urls = [single] if single else []
                if not pdf_urls:
                    raise ValueError(
                        "Law initiator PDF link not found "
                        "(cdep.ro: Expunerea de motive; senat.ro: Expunerea de motive "
                        "EM.PDF, or Forma inițiatorului as fallback)"
                    )
                pdf_urls_debug = list(pdf_urls)
                motive_pdf_url = pdf_urls[0]
                for pdf_index, pdf_url in enumerate(pdf_urls, start=1):
                    tried_pdfs += 1
                    label = (
                        f"PDF {pdf_index}/{len(pdf_urls)}"
                        if len(pdf_urls) > 1
                        else "initiator source PDF"
                    )
                    print(
                        f"{prefix}Law initiators OCR {index}/{len(records)}: "
                        f"{log_context} downloading {label}",
                        flush=True,
                    )
                    pdf_bytes = fetcher.fetch_bytes(pdf_url)
                    _write_bytes_atomic(cache_path, pdf_bytes)
                    total_bytes += len(pdf_bytes)
                    print(
                        f"{prefix}Law initiators OCR {index}/{len(records)}: "
                        f"{log_context} cached {label} at {cache_path}",
                        flush=True,
                    )
                    print(
                        f"{prefix}Law initiators OCR {index}/{len(records)}: "
                        f"{log_context} OCR/text extraction on {label} started "
                        f"({len(pdf_bytes)} bytes, last {ocr_max_pages} page(s))",
                        flush=True,
                    )
                    (
                        initiators_text,
                        initiators_source,
                        member_ids,
                        fuzzy_member_id,
                    ) = _extract_and_match_law_initiators_from_pdf_bytes(
                        pdf_bytes,
                        combined_member_rows=combined_member_rows,
                        ocr_language=ocr_language,
                        ocr_max_pages=ocr_max_pages,
                    )
                    if initiators_text or member_ids:
                        motive_pdf_url = pdf_url
                        break
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} OCR/text extraction finished "
                f"({len(initiators_text)} chars from {tried_pdfs}/{max(1, len(pdf_urls_debug))} "
                f"PDF(s), {total_bytes} bytes total)",
                flush=True,
            )
            conn.execute(
                """
                UPDATE dep_act_laws
                SET motive_pdf_url = ?,
                    initiators_text = ?,
                    initiators_source = ?,
                    initiators_parse_error = NULL,
                    initiators_extracted_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE law_id = ?
                """,
                (motive_pdf_url, initiators_text, initiators_source, law_id),
            )
            conn.execute(
                "UPDATE dep_act_member_laws SET is_initiator = 0 WHERE law_id = ?",
                (law_id,),
            )
            for member_id in member_ids:
                conn.execute(
                    """
                    INSERT INTO dep_act_member_laws (member_id, law_id, is_initiator)
                    VALUES (?, ?, 1)
                    ON CONFLICT(member_id, law_id)
                    DO UPDATE SET is_initiator = 1
                    """,
                    (member_id, law_id),
                )
            result.stored += 1
            result.associated += len(member_ids)
            match_note = _law_initiator_match_log_suffix(
                initiators_text,
                matched_member_ids=member_ids,
                deputy_rows=member_rows,
                senator_rows=senator_member_rows,
            )
            source_note = f" [source={initiators_source}]" if initiators_source else ""
            fuzzy_note = (
                f" [fuzzy single-signature match: {fuzzy_member_id}]"
                if fuzzy_member_id
                else ""
            )
            print(
                f"{prefix}Law initiators OCR {index}/{len(records)}: "
                f"{log_context} matched {len(member_ids)} initiator(i)"
                f"{match_note}{source_note}{fuzzy_note}",
                flush=True,
            )
            if debug:
                print(
                    "[law initiators debug] "
                    f"law_id={law_id} pdf_urls={pdf_urls_debug} "
                    f"source={initiators_source} matched={member_ids}"
                )
        except Exception as exc:  # noqa: BLE001 - keep crawling other laws.
            # Fix 8: don't persist transient network failures as permanent
            # `initiators_parse_error` — that prevents a retry on the next
            # run. Leaving `initiators_text` NULL causes the next hydration
            # pass to reprocess the same law, which is the desired behaviour
            # when a host is momentarily down.
            if _is_transient_network_exception(exc):
                transient_error_this_iter = True
                print(
                    f"{prefix}Law initiators OCR {index}/{len(records)}: "
                    f"{log_context} transient network error, will retry next run: {exc}",
                    flush=True,
                )
                if debug:
                    print(
                        "[law initiators debug] "
                        f"law_id={law_id} transient_error={exc}"
                    )
            else:
                _record_law_initiator_error(
                    conn,
                    law_id=law_id,
                    motive_pdf_url=motive_pdf_url,
                    error=str(exc),
                )
                if debug:
                    print(
                        "[law initiators debug] "
                        f"law_id={law_id} error={exc}"
                    )
                else:
                    print(
                        f"{prefix}Law initiators OCR {index}/{len(records)}: "
                        f"{log_context} failed: {exc}",
                        flush=True,
                    )
        finally:
            if hydrated_law_ids is not None:
                hydrated_law_ids.add(law_id)
            if commit_each:
                conn.commit()
        # Post-iteration throttle + circuit breaker. Kept outside the
        # try/finally so an abort decision is made only after all DB
        # state for this law has been flushed. The counter tracks only
        # transient network errors — successful fetches and permanent
        # parse errors reset it, because those mean the remote host is
        # actually responding.
        if transient_error_this_iter:
            consecutive_transient_failures += 1
            if (
                consecutive_failure_abort_after > 0
                and consecutive_transient_failures >= consecutive_failure_abort_after
            ):
                abort_reason = (
                    f"{consecutive_transient_failures} consecutive transient "
                    "network errors; aborting hydration early. Remaining "
                    "laws will be retried on the next run."
                )
                break
            if (
                consecutive_failure_pause_after > 0
                and consecutive_transient_failures
                > 0
                and consecutive_transient_failures % consecutive_failure_pause_after
                == 0
                and consecutive_failure_pause_seconds > 0
            ):
                print(
                    f"{prefix}Law initiators OCR: pausing "
                    f"{consecutive_failure_pause_seconds:.0f}s after "
                    f"{consecutive_transient_failures} consecutive transient "
                    "network errors (circuit breaker).",
                    flush=True,
                )
                sleeper(consecutive_failure_pause_seconds)
                continue
        else:
            consecutive_transient_failures = 0
        if per_fetch_sleep_seconds > 0 and index < len(records):
            sleeper(per_fetch_sleep_seconds)
    if abort_reason:
        print(f"{prefix}Law initiators OCR: {abort_reason}", flush=True)
    return result


def load_law_records_for_initiator_hydration(
    conn: sqlite3.Connection,
    *,
    member_ids: list[str],
    only_without_initiators: bool = True,
    limit: int | None = None,
) -> list[ListingRecord]:
    if not member_ids:
        return []
    placeholders = ", ".join("?" for _ in member_ids)
    limit_sql = "LIMIT ?" if limit is not None else ""
    missing_initiators_sql = (
        """
        AND NOT EXISTS (
            SELECT 1
            FROM dep_act_member_laws iml
            WHERE iml.law_id = l.law_id
              AND iml.is_initiator = 1
        )
        """
        if only_without_initiators
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT
               l.law_id,
               l.source_url,
               l.identifier,
               l.adopted_law_identifier,
               l.title,
               l.details_text,
               l.columns_json
        FROM dep_act_laws l
        JOIN dep_act_member_laws ml
          ON ml.law_id = l.law_id
        WHERE ml.member_id IN ({placeholders})
        {missing_initiators_sql}
        ORDER BY l.law_id
        {limit_sql}
        """,
        [*member_ids, limit] if limit is not None else member_ids,
    ).fetchall()
    records: list[ListingRecord] = []
    for row in rows:
        columns: list[str] = []
        if row[6]:
            try:
                parsed = json.loads(str(row[6]))
                if isinstance(parsed, list):
                    columns = [str(value) for value in parsed]
            except json.JSONDecodeError:
                columns = []
        records.append(
            ListingRecord(
                record_id=str(row[0]),
                source_url=str(row[1]),
                identifier=str(row[2]) if row[2] else None,
                adopted_law_identifier=str(row[3]) if row[3] else None,
                title=str(row[4]),
                details_text=str(row[5]),
                columns=columns,
            )
        )
    return records


def _insert_or_update_entity(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    record_id: str,
    source_url: str,
    values: dict[str, Any],
    update_existing: bool,
) -> tuple[str, bool]:
    existing = conn.execute(
        f"""
        SELECT {id_column}
        FROM {table}
        WHERE {id_column} = ? OR source_url = ?
        ORDER BY CASE WHEN source_url = ? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (record_id, source_url, source_url),
    ).fetchone()
    if existing:
        actual_id = str(existing[0])
        if not update_existing:
            return actual_id, False
        assignments = ["source_url = ?"] + [f"{column} = ?" for column in values]
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        params = [source_url, *values.values(), actual_id]
        cursor = conn.execute(
            f"""
            UPDATE {table}
            SET {", ".join(assignments)}
            WHERE {id_column} = ?
            """,
            params,
        )
        return actual_id, cursor.rowcount > 0

    columns = [id_column, "source_url", *values.keys()]
    placeholders = ", ".join("?" for _ in columns)
    cursor = conn.execute(
        f"""
        INSERT INTO {table} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT DO NOTHING
        """,
        (record_id, source_url, *values.values()),
    )
    if cursor.rowcount:
        return record_id, True

    # A concurrent or pre-existing source_url conflict may have won between the
    # SELECT and INSERT. Use the existing row for the association and leave it
    # untouched unless the caller explicitly requested updates on a later pass.
    row = conn.execute(
        f"SELECT {id_column} FROM {table} WHERE {id_column} = ? OR source_url = ? LIMIT 1",
        (record_id, source_url),
    ).fetchone()
    return (str(row[0]) if row else record_id), False


def _store_laws(
    conn: sqlite3.Connection,
    member_id: str,
    records: list[ListingRecord],
    *,
    update_existing: bool = False,
) -> StoreResult:
    result = StoreResult(seen=len(records))
    for record in records:
        actual_id, changed = _insert_or_update_entity(
            conn,
            table="dep_act_laws",
            id_column="law_id",
            record_id=record.record_id,
            source_url=record.source_url,
            values={
                "identifier": record.identifier,
                "adopted_law_identifier": record.adopted_law_identifier,
                "title": record.title,
                "details_text": record.details_text,
                "columns_json": json.dumps(record.columns, ensure_ascii=False),
            },
            update_existing=update_existing,
        )
        if changed:
            result.stored += 1
        cursor = conn.execute(
            """
            INSERT INTO dep_act_member_laws (member_id, law_id)
            VALUES (?, ?)
            ON CONFLICT(member_id, law_id) DO NOTHING
            """,
            (member_id, actual_id),
        )
        result.associated += cursor.rowcount
    return result


def _store_decision_projects(
    conn: sqlite3.Connection,
    member_id: str,
    records: list[ListingRecord],
    *,
    update_existing: bool = False,
) -> StoreResult:
    result = StoreResult(seen=len(records))
    for record in records:
        actual_id, changed = _insert_or_update_entity(
            conn,
            table="dep_act_decision_projects",
            id_column="decision_project_id",
            record_id=record.record_id,
            source_url=record.source_url,
            values={
                "identifier": record.identifier,
                "title": record.title,
                "details_text": record.details_text,
                "columns_json": json.dumps(record.columns, ensure_ascii=False),
            },
            update_existing=update_existing,
        )
        if changed:
            result.stored += 1
        cursor = conn.execute(
            """
            INSERT INTO dep_act_member_decision_projects (member_id, decision_project_id)
            VALUES (?, ?)
            ON CONFLICT(member_id, decision_project_id) DO NOTHING
            """,
            (member_id, actual_id),
        )
        result.associated += cursor.rowcount
    return result


def hydrate_question_records(
    *,
    listing_records: list[ListingRecord],
    fetcher: Fetcher,
    debug: bool = False,
) -> list[ListingRecord]:
    questions: list[ListingRecord] = []
    for index, record in enumerate(listing_records, start=1):
        detail_html = fetcher.fetch(record.source_url)
        recipient = parse_question_detail_recipient(detail_html)
        if debug:
            _print_question_detail_debug(
                index=index,
                record=record,
                detail_html=detail_html,
                recipient=recipient,
            )
        questions.append(
            ListingRecord(
                record_id=record.record_id,
                source_url=record.source_url,
                identifier=record.identifier,
                title=record.title,
                details_text=record.details_text,
                columns=record.columns,
                recipient=recipient,
            )
        )
    return questions


def _existing_question_records(
    conn: sqlite3.Connection,
    member_id: str,
) -> list[ListingRecord]:
    try:
        rows = conn.execute(
            """
            SELECT question_id,
                   source_url,
                   identifier,
                   text,
                   columns_json
            FROM dep_act_questions_interpellations
            WHERE member_id = ?
            ORDER BY question_id
            """,
            (member_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    records: list[ListingRecord] = []
    for question_id, source_url, identifier, text, columns_json in rows:
        columns: list[str] = []
        if columns_json:
            try:
                parsed = json.loads(columns_json)
                if isinstance(parsed, list):
                    columns = [str(item) for item in parsed]
            except json.JSONDecodeError:
                columns = []
        details_text = _clean_text(str(text or ""))
        records.append(
            ListingRecord(
                record_id=str(question_id),
                source_url=str(source_url),
                identifier=str(identifier) if identifier else None,
                title=details_text,
                details_text=details_text,
                columns=columns,
            )
        )
    return records


def _store_questions(
    conn: sqlite3.Connection,
    member_id: str,
    records: list[ListingRecord],
    *,
    update_existing: bool = False,
) -> StoreResult:
    result = StoreResult(seen=len(records))
    for record in records:
        _, changed = _insert_or_update_entity(
            conn,
            table="dep_act_questions_interpellations",
            id_column="question_id",
            record_id=record.record_id,
            source_url=record.source_url,
            values={
                "member_id": member_id,
                "identifier": record.identifier,
                "recipient": record.recipient,
                "text": record.details_text,
                "columns_json": json.dumps(record.columns, ensure_ascii=False),
            },
            update_existing=update_existing,
        )
        if changed:
            result.stored += 1
    return result


def _store_motions(
    conn: sqlite3.Connection,
    member_id: str,
    records: list[ListingRecord],
    *,
    update_existing: bool = False,
) -> StoreResult:
    result = StoreResult(seen=len(records))
    for record in records:
        actual_id, changed = _insert_or_update_entity(
            conn,
            table="dep_act_motions",
            id_column="motion_id",
            record_id=record.record_id,
            source_url=record.source_url,
            values={
                "title": record.title,
                "details_text": record.details_text,
                "columns_json": json.dumps(record.columns, ensure_ascii=False),
            },
            update_existing=update_existing,
        )
        if changed:
            result.stored += 1
        cursor = conn.execute(
            """
            INSERT INTO dep_act_member_motions (member_id, motion_id)
            VALUES (?, ?)
            ON CONFLICT(member_id, motion_id) DO NOTHING
            """,
            (member_id, actual_id),
        )
        result.associated += cursor.rowcount
    return result


def hydrate_political_declaration_records(
    *,
    member_id: str,
    listing_records: list[ListingRecord],
    fetcher: Fetcher,
    debug: bool = False,
) -> list[PoliticalDeclarationRecord]:
    declarations: list[PoliticalDeclarationRecord] = []
    for index, record in enumerate(listing_records, start=1):
        detail_html = fetcher.fetch(record.source_url)
        text_url, detail_title, full_text = parse_political_declaration_detail(
            detail_html,
            record.source_url,
            fetcher=fetcher,
        )
        if debug:
            _print_political_declaration_detail_debug(
                index=index,
                record=record,
                detail_html=detail_html,
                text_url=text_url,
                detail_title=detail_title,
                full_text=full_text,
            )
        title = record.title or detail_title or record.details_text
        declarations.append(
            PoliticalDeclarationRecord(
                declaration_id=record.record_id,
                member_id=member_id,
                source_url=record.source_url,
                text_url=text_url,
                title=title,
                full_text=full_text or record.details_text,
                details_text=record.details_text,
                columns=record.columns,
            )
        )
    return declarations


def _store_political_declarations(
    conn: sqlite3.Connection,
    member_id: str,
    records: list[PoliticalDeclarationRecord],
    *,
    update_existing: bool = False,
) -> StoreResult:
    result = StoreResult(seen=len(records))
    if update_existing:
        record_ids = [record.declaration_id for record in records]
        if record_ids:
            placeholders = ", ".join("?" for _ in record_ids)
            conn.execute(
                f"""
                DELETE FROM dep_act_political_declarations
                WHERE member_id = ?
                  AND political_declaration_id NOT IN ({placeholders})
                """,
                (member_id, *record_ids),
            )
        else:
            conn.execute(
                "DELETE FROM dep_act_political_declarations WHERE member_id = ?",
                (member_id,),
            )
    for record in records:
        _, changed = _insert_or_update_entity(
            conn,
            table="dep_act_political_declarations",
            id_column="political_declaration_id",
            record_id=record.declaration_id,
            source_url=record.source_url,
            values={
                "member_id": record.member_id,
                "text_url": record.text_url,
                "title": record.title,
                "full_text": record.full_text,
                "details_text": record.details_text,
                "columns_json": json.dumps(record.columns, ensure_ascii=False),
            },
            update_existing=update_existing,
        )
        if changed:
            result.stored += 1
    return result


def store_records(
    conn: sqlite3.Connection,
    *,
    member_id: str,
    kind: str,
    records: list[ListingRecord] | list[PoliticalDeclarationRecord],
    update_existing: bool = False,
) -> StoreResult:
    if kind == "laws":
        return _store_laws(
            conn,
            member_id,
            records,
            update_existing=update_existing,
        )
    if kind == "decision_projects":
        return _store_decision_projects(
            conn,
            member_id,
            records,
            update_existing=update_existing,
        )
    if kind == "questions":
        return _store_questions(
            conn,
            member_id,
            records,
            update_existing=update_existing,
        )
    if kind == "motions":
        return _store_motions(
            conn,
            member_id,
            records,
            update_existing=update_existing,
        )
    if kind == "political_declarations":
        return _store_political_declarations(
            conn,
            member_id,
            records,
            update_existing=update_existing,
        )
    raise ValueError(f"Unknown listing kind: {kind}")


def update_member_activity_crawl(
    conn: sqlite3.Connection,
    *,
    member_id: str,
    profile_url: str,
    activity: dict[str, ActivityLink],
    results: dict[str, StoreResult],
    error: str | None = None,
    update_existing: bool = False,
) -> None:
    laws = activity.get("legislative_proposals", ActivityLink())
    decisions = activity.get("decision_projects", ActivityLink())
    questions = activity.get("questions", ActivityLink())
    motions = activity.get("motions", ActivityLink())
    political_declarations = activity.get("political_declarations", ActivityLink())
    columns = {
        "profile_url": profile_url,
        "legislative_proposals_url": laws.url,
        "legislative_proposals_count": laws.count,
        "promulgated_laws_count": laws.promulgated_count,
        "decision_projects_url": decisions.url,
        "decision_projects_count": decisions.count,
        "questions_url": questions.url,
        "questions_count": questions.count,
        "motions_url": motions.url,
        "motions_count": motions.count,
        "political_declarations_url": political_declarations.url,
        "political_declarations_count": political_declarations.count,
        "legislative_proposals_stored": results.get("laws", StoreResult()).stored,
        "decision_projects_stored": results.get("decision_projects", StoreResult()).stored,
        "questions_stored": results.get("questions", StoreResult()).stored,
        "motions_stored": results.get("motions", StoreResult()).stored,
        "political_declarations_stored": results.get("political_declarations", StoreResult()).stored,
        "last_error": error,
    }
    existing = conn.execute(
        "SELECT member_id FROM dep_act_member_activity_crawl WHERE member_id = ?",
        (member_id,),
    ).fetchone()
    if existing and not update_existing:
        return
    if existing:
        assignments = [f"{column} = ?" for column in columns]
        assignments.extend(["crawled_at = CURRENT_TIMESTAMP", "updated_at = CURRENT_TIMESTAMP"])
        conn.execute(
            f"""
            UPDATE dep_act_member_activity_crawl
            SET {", ".join(assignments)}
            WHERE member_id = ?
            """,
            (*columns.values(), member_id),
        )
        return
    conn.execute(
        f"""
        INSERT INTO dep_act_member_activity_crawl (
            member_id,
            {", ".join(columns.keys())},
            crawled_at,
            updated_at
        )
        VALUES (
            ?,
            {", ".join("?" for _ in columns)},
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        """,
        (member_id, *columns.values()),
    )


def update_member_activity_error(
    conn: sqlite3.Connection,
    *,
    member_id: str,
    profile_url: str,
    error: str,
) -> None:
    existing = conn.execute(
        "SELECT member_id FROM dep_act_member_activity_crawl WHERE member_id = ?",
        (member_id,),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE dep_act_member_activity_crawl
            SET last_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE member_id = ?
            """,
            (error, member_id),
        )
        return
    conn.execute(
        """
        INSERT INTO dep_act_member_activity_crawl (
            member_id,
            profile_url,
            last_error,
            crawled_at,
            updated_at
        )
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (member_id, profile_url, error),
    )


def _print_member_log(
    member: dict[str, Any],
    activity: dict[str, ActivityLink],
    results: dict[str, StoreResult],
    *,
    prefix: str,
    error: str | None = None,
) -> None:
    print(f"{prefix} {member['member_id']} {member['name']}")
    if error:
        print(f"  ERROR: {error}")
        return
    specs = [
        ("Propuneri legislative", "legislative_proposals", "laws"),
        ("Proiecte de hotarare", "decision_projects", "decision_projects"),
        ("Intrebari/interpelari", "questions", "questions"),
        ("Motiuni", "motions", "motions"),
        ("Declaratii politice scrise", "political_declarations", "political_declarations"),
    ]
    for label, activity_key, result_key in specs:
        link = activity.get(activity_key, ActivityLink())
        result = results.get(result_key, StoreResult())
        extra = ""
        if activity_key == "legislative_proposals":
            extra = f", promulgated={link.promulgated_count}"
        association_text = (
            ""
            if result_key in {"questions", "political_declarations"}
            else f", associations={result.associated}"
        )
        print(
            "  "
            f"{label}: profile_count={link.count}{extra}, "
            f"stored={result.stored}{association_text}, "
            f"url={link.url or '-'}"
        )

def crawl_member(
    conn: sqlite3.Connection,
    *,
    member: dict[str, Any],
    fetcher: Fetcher,
    dry_run: bool = False,
    update_existing: bool = False,
    debug_profile: bool = False,
    debug_questions: bool = False,
    debug_motions: bool = False,
    debug_political_declarations: bool = False,
) -> tuple[dict[str, ActivityLink], dict[str, StoreResult], str | None]:
    profile_url = str(member["profile_url"])
    activity: dict[str, ActivityLink] = {}
    results: dict[str, StoreResult] = {}
    error: str | None = None
    try:
        profile_html = fetcher.fetch(profile_url)
        activity = parse_profile_activity(profile_html, profile_url)
        if debug_profile:
            _print_profile_activity_debug(
                member_id=member["member_id"],
                profile_url=profile_url,
                profile_html=profile_html,
                activity=activity,
            )
        fetch_specs = [
            ("legislative_proposals", "laws"),
            ("decision_projects", "decision_projects"),
            ("questions", "questions"),
            ("motions", "motions"),
            ("political_declarations", "political_declarations"),
        ]
        for activity_key, listing_kind in fetch_specs:
            link = activity.get(activity_key, ActivityLink())
            synthesized_activity_link = False
            if not link.url:
                fallback_url = _standard_profile_activity_url(profile_url, activity_key)
                if fallback_url:
                    link = ActivityLink(url=fallback_url)
                    activity[activity_key] = link
                    synthesized_activity_link = True
                    if debug_profile or (listing_kind == "motions" and debug_motions):
                        print(
                            "[profile debug] "
                            f"synthesized activity[{activity_key}] url={fallback_url}"
                        )
                else:
                    if listing_kind == "motions" and debug_motions:
                        print(
                            "[motions debug] "
                            f"profile did not expose a motions URL for member_id={member['member_id']}"
                        )
                    if listing_kind == "questions" and (debug_questions or update_existing):
                        records = _existing_question_records(conn, member["member_id"])
                        if records and debug_questions:
                            print(
                                "[questions debug] "
                                "profile did not expose a questions URL; "
                                f"using {len(records)} existing DB question rows for detail hydration"
                            )
                            _print_questions_listing_debug(
                                member_id=member["member_id"],
                                listing_url="- existing DB rows -",
                                listing_html="",
                                records=records,
                            )
                        if records:
                            records = hydrate_question_records(
                                listing_records=records,
                                fetcher=fetcher,
                                debug=debug_questions,
                            )
                            if dry_run:
                                results[listing_kind] = StoreResult(
                                    seen=len(records),
                                    stored=len(records),
                                    associated=0,
                                )
                            else:
                                results[listing_kind] = store_records(
                                    conn,
                                    member_id=member["member_id"],
                                    kind=listing_kind,
                                    records=records,
                                    update_existing=update_existing,
                                )
                            continue
                    results[listing_kind] = StoreResult()
                    continue
            listing_html = fetcher.fetch(link.url)
            records = parse_listing_records(listing_html, link.url, listing_kind)
            if synthesized_activity_link:
                link.count = len(records)
                activity[activity_key] = link
            if listing_kind == "motions" and debug_motions:
                _print_motions_listing_debug(
                    member_id=member["member_id"],
                    listing_url=link.url,
                    listing_html=listing_html,
                    records=records,
                )
            if listing_kind == "questions":
                if debug_questions:
                    _print_questions_listing_debug(
                        member_id=member["member_id"],
                        listing_url=link.url,
                        listing_html=listing_html,
                        records=records,
                    )
                records = hydrate_question_records(
                    listing_records=records,
                    fetcher=fetcher,
                    debug=debug_questions,
                )
            elif listing_kind == "political_declarations":
                if debug_political_declarations:
                    _print_political_declaration_listing_debug(
                        member_id=member["member_id"],
                        listing_url=link.url,
                        listing_html=listing_html,
                        records=records,
                    )
                records = hydrate_political_declaration_records(
                    member_id=member["member_id"],
                    listing_records=records,
                    fetcher=fetcher,
                    debug=debug_political_declarations,
                )
            if dry_run:
                results[listing_kind] = StoreResult(
                    seen=len(records),
                    stored=len(records),
                    associated=(
                        0
                        if listing_kind in {"questions", "political_declarations"}
                        else len(records)
                    ),
                )
            else:
                results[listing_kind] = store_records(
                    conn,
                    member_id=member["member_id"],
                    kind=listing_kind,
                    records=records,
                    update_existing=update_existing,
                )
        if not dry_run:
            update_member_activity_crawl(
                conn,
                member_id=member["member_id"],
                profile_url=profile_url,
                activity=activity,
                results=results,
                update_existing=update_existing,
            )
    except Exception as exc:  # noqa: BLE001 - crawl should continue per member.
        error = str(exc)
        if not dry_run:
            update_member_activity_error(
                conn,
                member_id=member["member_id"],
                profile_url=profile_url,
                error=error,
            )
    return activity, results, error


def _select_members(
    deputies: list[dict[str, Any]],
    *,
    member_ids: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    selected = deputies
    if member_ids:
        wanted = set(member_ids)
        selected = [
            member
            for member in selected
            if member["member_id"] in wanted
            or member["source_member_id"] in wanted
            or member["name"] in wanted
        ]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_law_initiator_hydration_phase(
    conn: sqlite3.Connection,
    *,
    selected: list[dict[str, Any]],
    fetcher: Fetcher,
    update_existing: bool,
    ocr_language: str,
    ocr_pages: int,
    debug: bool,
    law_limit: int | None = None,
    hydrate_retries: int | None = None,
    per_fetch_sleep_seconds: float = 0.5,
    consecutive_failure_pause_after: int = 10,
    consecutive_failure_pause_seconds: float = 60.0,
    consecutive_failure_abort_after: int = 25,
) -> None:
    print("Starting law initiator OCR hydration from stored dep_act_laws rows.")
    law_records = load_law_records_for_initiator_hydration(
        conn,
        member_ids=[member["member_id"] for member in selected],
        only_without_initiators=True,
        limit=law_limit,
    )
    print(
        "Law initiator OCR hydration will process "
        f"{len(law_records)} stored law(s) with no dep_act_member_laws row "
        f"where is_initiator=1 (scoped to selected deputati"
        + (f", limited to {law_limit}" if law_limit is not None else "")
        + ").",
        flush=True,
    )
    if not law_records:
        print("Law initiator OCR hydration complete: no stored laws to process.")
        return
    validate_law_initiator_ocr_dependencies()
    # Use a dedicated fetcher for hydration when a different retries count is
    # requested. Hard blocks (HTTP-level rate limits / TCP RSTs) otherwise
    # result in retries × N, which only amplifies load against an already
    # unhappy server.
    hydration_fetcher = fetcher
    if hydrate_retries is not None and hydrate_retries != fetcher.retries:
        hydration_fetcher = Fetcher(
            user_agent=fetcher.user_agent,
            timeout=fetcher.timeout,
            retries=hydrate_retries,
            sleep_seconds=fetcher.sleep_seconds,
        )
        print(
            "Law initiator OCR hydration: using dedicated fetcher with "
            f"retries={hydrate_retries} (crawl-default was {fetcher.retries}).",
            flush=True,
        )
    print(
        "Law initiator OCR hydration: throttle settings "
        f"per_fetch_sleep={per_fetch_sleep_seconds}s, "
        f"pause_after={consecutive_failure_pause_after} failures "
        f"(pause_seconds={consecutive_failure_pause_seconds}s), "
        f"abort_after={consecutive_failure_abort_after} failures.",
        flush=True,
    )
    law_result = hydrate_law_initiators_for_records(
        conn,
        records=law_records,
        fetcher=hydration_fetcher,
        member_rows=load_deputy_member_match_rows(conn),
        force=True,
        skip_if_initiator_linked=True,
        hydrated_law_ids=set(),
        ocr_language=ocr_language,
        ocr_max_pages=ocr_pages,
        debug=debug,
        progress_prefix="[law OCR]",
        commit_each=True,
        per_fetch_sleep_seconds=per_fetch_sleep_seconds,
        consecutive_failure_pause_after=consecutive_failure_pause_after,
        consecutive_failure_pause_seconds=consecutive_failure_pause_seconds,
        consecutive_failure_abort_after=consecutive_failure_abort_after,
    )
    print(
        "Law initiator OCR hydration complete: "
        f"checked={law_result.seen}, hydrated={law_result.stored}, "
        f"matched_initiators={law_result.associated}.",
        flush=True,
    )


def crawl_deputy_activity(
    *,
    db_path: Path,
    input_path: Path,
    limit: int | None = None,
    offset: int = 0,
    member_ids: list[str] | None = None,
    timeout: int = 30,
    retries: int = 2,
    sleep_seconds: float = 0.5,
    dry_run: bool = False,
    update_existing: bool = False,
    debug_profile: bool = False,
    debug_questions: bool = False,
    debug_motions: bool = False,
    debug_political_declarations: bool = False,
    hydrate_law_initiators: bool = False,
    only_hydrate_law_initiators: bool = False,
    law_initiator_ocr_language: str = DEFAULT_OCR_LANGUAGE,
    law_initiator_ocr_pages: int = DEFAULT_LAW_INITIATOR_OCR_PAGES,
    hydrate_law_limit: int | None = None,
    debug_law_initiators: bool = False,
    hydrate_retries: int | None = 1,
    hydrate_sleep_seconds: float = 0.5,
    hydrate_failure_pause_after: int = 10,
    hydrate_failure_pause_seconds: float = 60.0,
    hydrate_failure_abort_after: int = 25,
    export_activity: bool = False,
    only_export_activity: bool = False,
    activity_output_dir: Path = DEFAULT_ACTIVITY_OUTPUT_DIR,
) -> int:
    deputies = _load_deputies(input_path)
    selected = _select_members(
        deputies,
        member_ids=member_ids,
        limit=limit,
        offset=offset,
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        ensure_activity_schema(conn)
        require_members_present(conn, selected)
        conn.commit()

        if only_export_activity:
            if dry_run:
                print(
                    "Dry run: activity export would run, but no files are written."
                )
                return 0
            print("Skipping crawl and hydration: only exporting activity snapshots.")
            export_activity_snapshots(
                conn,
                output_dir=activity_output_dir,
                progress_prefix="[activity export]",
            )
            return 0

        if only_hydrate_law_initiators:
            if dry_run:
                print(
                    "Dry run: law initiator OCR hydration would run, but DB records are not written."
                )
            print(
                f"Skipping crawl phase. Selected {len(selected)} deputat(i) "
                "only to scope stored laws for initiator OCR hydration."
            )
            if not dry_run:
                run_law_initiator_hydration_phase(
                    conn,
                    selected=selected,
                    fetcher=fetcher,
                    update_existing=update_existing,
                    ocr_language=law_initiator_ocr_language,
                    ocr_pages=law_initiator_ocr_pages,
                    law_limit=hydrate_law_limit,
                    debug=debug_law_initiators,
                    hydrate_retries=hydrate_retries,
                    per_fetch_sleep_seconds=hydrate_sleep_seconds,
                    consecutive_failure_pause_after=hydrate_failure_pause_after,
                    consecutive_failure_pause_seconds=hydrate_failure_pause_seconds,
                    consecutive_failure_abort_after=hydrate_failure_abort_after,
                )
                if export_activity:
                    export_activity_snapshots(
                        conn,
                        output_dir=activity_output_dir,
                        progress_prefix="[activity export]",
                    )
            return 0

        print(f"Crawling {len(selected)} deputat(i) from {input_path} into {db_path}")
        if dry_run:
            print("Dry run: pages are fetched and parsed, but DB records are not written.")
        elif update_existing:
            print("Update mode: existing crawler rows are refreshed from parsed pages.")
        else:
            print("Insert-only mode: existing crawler rows are left unchanged.")
        if debug_profile:
            print("Profile activity debug logging is enabled.")
        if debug_questions:
            print("Questions/interpellations debug logging is enabled.")
        if debug_motions:
            print("Motions debug logging is enabled.")
        if debug_political_declarations:
            print("Political declaration debug logging is enabled.")
        if hydrate_law_initiators and dry_run:
            print("Law initiator OCR hydration is skipped in dry-run mode.")
        elif hydrate_law_initiators:
            print(
                "Law initiator OCR hydration will run after the crawl phase "
                f"(language={law_initiator_ocr_language}, pages={law_initiator_ocr_pages})."
            )
        if debug_law_initiators:
            print("Law initiator OCR debug logging is enabled.")

        failures = 0
        for index, member in enumerate(selected, start=1):
            print(
                f"[{index}/{len(selected)}] Starting {member['member_id']} {member['name']}",
                flush=True,
            )
            activity, results, error = crawl_member(
                conn,
                member=member,
                fetcher=fetcher,
                dry_run=dry_run,
                update_existing=update_existing,
                debug_profile=debug_profile,
                debug_questions=debug_questions,
                debug_motions=debug_motions,
                debug_political_declarations=debug_political_declarations,
            )
            if error:
                failures += 1
            if not dry_run:
                conn.commit()
            _print_member_log(
                member,
                activity,
                results,
                prefix=f"[{index}/{len(selected)}]",
                error=error,
            )
            if index < len(selected) and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if hydrate_law_initiators and not dry_run:
            print("Crawl phase complete.")
            run_law_initiator_hydration_phase(
                conn,
                selected=selected,
                fetcher=fetcher,
                update_existing=update_existing,
                ocr_language=law_initiator_ocr_language,
                ocr_pages=law_initiator_ocr_pages,
                law_limit=hydrate_law_limit,
                debug=debug_law_initiators,
                hydrate_retries=hydrate_retries,
                per_fetch_sleep_seconds=hydrate_sleep_seconds,
                consecutive_failure_pause_after=hydrate_failure_pause_after,
                consecutive_failure_pause_seconds=hydrate_failure_pause_seconds,
                consecutive_failure_abort_after=hydrate_failure_abort_after,
            )

        if export_activity and not dry_run:
            export_activity_snapshots(
                conn,
                output_dir=activity_output_dir,
                progress_prefix="[activity export]",
            )

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crawl CDEP deputy profile activity into state/state.sqlite."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--member-id",
        action="append",
        dest="member_ids",
        help="Filter by member_id (deputat_1), source id (1), or exact name. Repeatable.",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help=(
            "Refresh existing rows in crawler-owned tables. By default the "
            "crawler inserts only new crawler data and leaves existing rows unchanged."
        ),
    )
    parser.add_argument(
        "--debug-political-declarations",
        action="store_true",
        help=(
            "Print political declaration listing/detail parse diagnostics, including "
            "selected URLs, paragraph previews, and parsed full_text previews."
        ),
    )
    parser.add_argument(
        "--hydrate-law-initiators",
        action="store_true",
        help=(
            "For crawled laws, download the Expunerea de motive PDF, OCR it locally, "
            "extract initiator names, and set dep_act_member_laws.is_initiator."
        ),
    )
    parser.add_argument(
        "--only-hydrate-law-initiators",
        action="store_true",
        help=(
            "Skip normal crawling and only run law initiator OCR hydration for stored "
            "dep_act_laws rows that do not yet have any is_initiator association."
        ),
    )
    parser.add_argument(
        "--law-initiator-ocr-language",
        default=DEFAULT_OCR_LANGUAGE,
        help=(
            "Tesseract language for law initiator OCR. Defaults to ron+eng and "
            "falls back to eng if the requested language is unavailable."
        ),
    )
    parser.add_argument(
        "--law-initiator-ocr-pages",
        type=int,
        default=DEFAULT_LAW_INITIATOR_OCR_PAGES,
        help=(
            "Maximum number of Expunerea de motive PDF pages to OCR per law "
            f"(default {DEFAULT_LAW_INITIATOR_OCR_PAGES}; initiator tables may not be on page 1)."
        ),
    )
    parser.add_argument(
        "--hydrate-law-limit",
        type=int,
        default=None,
        help=(
            "Limit the number of stored laws processed during the law-initiator "
            "hydration phase. Useful for local testing."
        ),
    )
    parser.add_argument(
        "--debug-law-initiators",
        action="store_true",
        help="Print law initiator PDF/OCR diagnostics and matched deputy IDs.",
    )
    parser.add_argument(
        "--hydrate-sleep-seconds",
        type=float,
        default=0.5,
        help=(
            "Seconds to sleep between law-page fetches during the OCR "
            "hydration phase (default: 0.5). 0 disables the throttle."
        ),
    )
    parser.add_argument(
        "--hydrate-retries",
        type=int,
        default=1,
        help=(
            "Retries for each individual law fetch during hydration "
            "(default: 1). Kept lower than --retries so a hard block "
            "doesn't get hammered on every law. Set to 0 to disable "
            "per-fetch retries entirely."
        ),
    )
    parser.add_argument(
        "--hydrate-failure-pause-after",
        type=int,
        default=10,
        help=(
            "Circuit-breaker: after this many consecutive transient "
            "network failures, pause for --hydrate-failure-pause-seconds "
            "and continue (default: 10). Set to 0 to disable the pause."
        ),
    )
    parser.add_argument(
        "--hydrate-failure-pause-seconds",
        type=float,
        default=60.0,
        help=(
            "Pause duration, in seconds, after "
            "--hydrate-failure-pause-after consecutive transient failures "
            "(default: 60). Set to 0 to disable the pause."
        ),
    )
    parser.add_argument(
        "--hydrate-failure-abort-after",
        type=int,
        default=25,
        help=(
            "Circuit-breaker: after this many consecutive transient "
            "network failures, abort the hydration phase early so the "
            "remaining laws are retried on the next run (default: 25). "
            "Set to 0 to disable the abort."
        ),
    )
    parser.add_argument(
        "--export-activity",
        action="store_true",
        help=(
            "After the crawl (and/or law-initiator hydration) phase, write "
            "per-member and per-party activity snapshots as JSON to "
            "outputs/activity/members/ and outputs/activity/parties/. "
            "Previous activity_*.json files in those folders are removed "
            "before the new ones are written."
        ),
    )
    parser.add_argument(
        "--only-export-activity",
        action="store_true",
        help=(
            "Skip crawling and OCR entirely; just read the already-populated "
            "DB and regenerate the activity snapshot files. Useful when you "
            "tweak the snapshot shape or after a hydration run."
        ),
    )
    parser.add_argument(
        "--activity-output-dir",
        default=str(DEFAULT_ACTIVITY_OUTPUT_DIR),
        help=(
            "Root directory for activity snapshots. Two subfolders are "
            f"created under it: members/ and parties/ (default: {DEFAULT_ACTIVITY_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--debug-questions",
        action="store_true",
        help=(
            "Print question/interpellation listing/detail parse diagnostics, including "
            "detail-page rows and parsed recipient values."
        ),
    )
    parser.add_argument(
        "--debug-motions",
        action="store_true",
        help=(
            "Print motions listing parse diagnostics, including profile-discovered "
            "listing URL, listing rows, links, and parsed motion records."
        ),
    )
    parser.add_argument(
        "--debug-profile",
        action="store_true",
        help=(
            "Print profile activity parse diagnostics, including activity-like rows "
            "and discovered activity links/counts."
        ),
    )
    args = parser.parse_args()

    try:
        return crawl_deputy_activity(
            db_path=Path(args.db_path),
            input_path=Path(args.input),
            limit=args.limit,
            offset=args.offset,
            member_ids=args.member_ids,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
            dry_run=args.dry_run,
            update_existing=args.update_existing,
            debug_profile=args.debug_profile,
            debug_questions=args.debug_questions,
            debug_motions=args.debug_motions,
            debug_political_declarations=args.debug_political_declarations,
            hydrate_law_initiators=args.hydrate_law_initiators,
            only_hydrate_law_initiators=args.only_hydrate_law_initiators,
            law_initiator_ocr_language=args.law_initiator_ocr_language,
            law_initiator_ocr_pages=args.law_initiator_ocr_pages,
            hydrate_law_limit=args.hydrate_law_limit,
            debug_law_initiators=args.debug_law_initiators,
            hydrate_retries=args.hydrate_retries,
            hydrate_sleep_seconds=args.hydrate_sleep_seconds,
            hydrate_failure_pause_after=args.hydrate_failure_pause_after,
            hydrate_failure_pause_seconds=args.hydrate_failure_pause_seconds,
            hydrate_failure_abort_after=args.hydrate_failure_abort_after,
            export_activity=args.export_activity,
            only_export_activity=args.only_export_activity,
            activity_output_dir=Path(args.activity_output_dir),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
