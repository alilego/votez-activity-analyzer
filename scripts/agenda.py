from __future__ import annotations

import re


_AGENDA_ITEM_PATTERN = re.compile(
    r"(?P<type>Proiectul\s+de\s+(?:Lege|Hotărâre)|Propunerea\s+legislativă)"
    r"\s+(?P<title>(?:privind|pentru|de)\s+.+?)(?=\.\s|\.\s*$|;\s|,\s*transmis\b|,\s*PL)",
    re.IGNORECASE | re.DOTALL,
)

_PL_X_PATTERN = re.compile(r"PL-?x\s+(\d{1,5}/\d{4})", re.IGNORECASE)
_PHCD_PATTERN = re.compile(r"PHCD\s+(\d{1,5}/\d{4})", re.IGNORECASE)
_OUG_IN_TITLE_PATTERN = re.compile(
    r"(?:Ordonanţ[aăei]+|OUG)\s+(?:de\s+urgenţă\s+a\s+Guvernului\s+)?nr\.?\s*(\d{1,4}/\d{4})",
    re.IGNORECASE,
)
_LEGEA_IN_TITLE_PATTERN = re.compile(
    r"Legea\s+nr\.?\s*(\d{1,4}/\d{4})", re.IGNORECASE
)

_AGENDA_INTRO_MARKERS = (
    "intrăm în ordinea de zi",
    "intram in ordinea de zi",
    "ordinea de zi",
    "supunem dezbaterii",
)

_CHAIR_CONFIRM_PATTERN = re.compile(
    r"(?:proiectul de lege|propunerea legislativă|proiectul de hotărâre)\s+rămâne\s+(?:la\s+)?votul\s+final",
    re.IGNORECASE,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _default_title(law_id: str | None) -> str:
    if not law_id:
        return ""
    if law_id.startswith("PL-x"):
        return f"Proiect legislativ {law_id}"
    if law_id.startswith("PHCD"):
        return f"Proiect de Hotărâre {law_id}"
    if law_id.startswith("OUG"):
        return f"Ordonanță de urgență {law_id}"
    return f"Act normativ {law_id}"


def _extract_law_id_from_text(text: str) -> str | None:
    """Extract the primary law identifier from an agenda-announcing speech fragment."""
    m = _PL_X_PATTERN.search(text)
    if m:
        return f"PL-x {m.group(1)}"
    m = _PHCD_PATTERN.search(text)
    if m:
        return f"PHCD {m.group(1)}"
    m = _OUG_IN_TITLE_PATTERN.search(text)
    if m:
        return f"OUG nr. {m.group(1)}"
    m = _LEGEA_IN_TITLE_PATTERN.search(text)
    if m:
        return f"Legea nr. {m.group(1)}"
    return None


def _extract_title_from_text(text: str) -> str | None:
    """Extract a short agenda item title from a speech fragment."""
    m = _AGENDA_ITEM_PATTERN.search(text[:600])
    if m:
        item_type = _normalize_whitespace(m.group("type"))
        title_body = _normalize_whitespace(m.group("title"))
        title_body = title_body[:150]
        if title_body.endswith(","):
            title_body = title_body[:-1].strip()
        return f"{item_type} {title_body}"
    return None


def _is_agenda_announcing_speech(text: str) -> bool:
    """Check if a speech is likely announcing/introducing a legislative agenda item."""
    lower = text.lower()
    if _CHAIR_CONFIRM_PATTERN.search(text):
        return False
    has_item_type = bool(re.search(
        r"proiectul\s+de\s+(lege|hotărâre)|propunerea\s+legislativă",
        lower,
    ))
    has_law_ref = bool(
        _PL_X_PATTERN.search(text)
        or _PHCD_PATTERN.search(text)
        or re.search(r"privind\s+aprobarea", lower)
    )
    has_report = bool(re.search(r"raport\w*\s+(?:comun\s+)?(?:asupra|privind)", lower))
    return has_item_type and (has_law_ref or has_report)


def extract_agenda_from_session(
    initial_notes: str,
    speeches: list[dict],
    max_items: int = 25,
) -> list[dict]:
    """
    Extract a structured legislative agenda from session data.

    Scans both `initial_notes` and speech texts for agenda item patterns.
    Returns a list of dicts: [{item_number, title, law_id}].

    Agenda items are extracted from:
    - initial_notes (rare, but captured when present)
    - Chair speeches that introduce legislative items
    - Committee report speeches that present law references
    """
    seen_law_ids: set[str] = set()
    seen_titles: set[str] = set()
    items: list[dict] = []
    item_counter = 0

    def _add_item(title: str | None, law_id: str | None) -> None:
        nonlocal item_counter
        if item_counter >= max_items:
            return
        if not title and not law_id:
            return
        dedup_key = (law_id or "").lower() + "|" + (title or "")[:60].lower()
        if law_id and law_id.lower() in seen_law_ids:
            return
        title_key = (title or "")[:60].lower()
        if title_key and title_key in seen_titles and not law_id:
            return

        item_counter += 1
        if law_id:
            seen_law_ids.add(law_id.lower())
        if title_key:
            seen_titles.add(title_key)
        items.append({
            "item_number": item_counter,
            "title": title or "",
            "law_id": law_id,
        })

    notes = (initial_notes or "").strip()
    if notes:
        law_id = _extract_law_id_from_text(notes)
        title = _extract_title_from_text(notes)
        if law_id or title:
            _add_item(title or _default_title(law_id), law_id)

    for sp in speeches:
        text = str(sp.get("text") or "")
        if not text.strip() or len(text) < 40:
            continue

        if not _is_agenda_announcing_speech(text):
            continue

        law_id = _extract_law_id_from_text(text)
        title = _extract_title_from_text(text)

        if law_id or title:
            _add_item(title or _default_title(law_id), law_id)

    return items
