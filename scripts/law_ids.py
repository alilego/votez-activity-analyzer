from __future__ import annotations

import re


_LAW_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PL-x", re.compile(r"\bPL[\s\-–—]*x[\s\-–—]*(\d{1,4}/\d{4})\b", re.IGNORECASE)),
    ("Legea nr.", re.compile(r"\bLegea\s+nr\.?\s*(\d{1,4}/\d{4})\b", re.IGNORECASE)),
    ("OUG nr.", re.compile(r"\bOUG\s+nr\.?\s*(\d{1,4}/\d{4})\b", re.IGNORECASE)),
    ("HG nr.", re.compile(r"\bHG\s+nr\.?\s*(\d{1,4}/\d{4})\b", re.IGNORECASE)),
    (
        "Directiva UE",
        re.compile(r"\bDirectiva\s+(?:UE|CE|UE/CE)?\s*([0-9]{2,4}/[A-Z]{2,4}|\d{1,4}/\d{4})\b", re.IGNORECASE),
    ),
    (
        "Regulamentul UE",
        re.compile(
            r"\bRegulamentul\s+(?:UE|CE|UE/CE)?(?:\s*\(UE\))?\s*(\d{1,4}/\d{4})\b",
            re.IGNORECASE,
        ),
    ),
)

_GENERIC_NR_PATTERN = re.compile(r"\bnr\.?\s*(\d{1,4}/\d{4})\b", re.IGNORECASE)
_LEGISLATIVE_CONTEXT_PATTERN = re.compile(
    r"\b(lege|proiect(?:ul)?\s+de\s+lege|ordonan(?:t|ț)a|oug|hot[aă]r[aâ]re(?:a)?\s+de\s+guvern|amendament|art\.?)\b",
    re.IGNORECASE,
)


def _normalize_law_id(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"PL\s*-\s*x", "PL-x", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _add_law_id(bucket: dict[str, list[int]], law_id: str, speech_index: int) -> None:
    normalized = _normalize_law_id(law_id)
    if not normalized:
        return
    indexes = bucket.setdefault(normalized, [])
    if speech_index not in indexes:
        indexes.append(speech_index)


def extract_law_id_index_from_speeches(speeches: list[dict]) -> dict[str, list[int]]:
    """
    Build a per-session law-id index: {law_id: [speech_indices]}.
    """
    out: dict[str, list[int]] = {}
    for sp in speeches:
        text = str(sp.get("text") or "")
        if not text.strip():
            continue
        try:
            speech_index = int(sp.get("speech_index"))
        except (TypeError, ValueError):
            continue

        for prefix, pattern in _LAW_PATTERNS:
            for match in pattern.finditer(text):
                token = match.group(1)
                if not token:
                    continue
                _add_law_id(out, f"{prefix} {token}", speech_index)

        if _LEGISLATIVE_CONTEXT_PATTERN.search(text):
            for match in _GENERIC_NR_PATTERN.finditer(text):
                token = match.group(1)
                if token:
                    _add_law_id(out, f"nr. {token}", speech_index)

    return out


def allowed_law_ids(index: dict[str, list[int]]) -> set[str]:
    return {_normalize_law_id(k) for k in index.keys() if str(k).strip()}


def keep_only_allowed_law_id(raw_law_id: str | None, allowed: set[str]) -> str | None:
    if not raw_law_id:
        return None
    normalized = _normalize_law_id(raw_law_id)
    if not normalized:
        return None
    if normalized in allowed:
        return normalized
    return None
