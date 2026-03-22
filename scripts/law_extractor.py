"""
Deterministic extraction of Romanian legislative references from text.

Covers:
  - PL-x NNN/YYYY (parliamentary bill identifiers)
  - Legea nr. NNN/YYYY
  - OUG nr. NNN/YYYY (emergency ordinances)
  - OG nr. NNN/YYYY (government ordinances)
  - HG nr. NNN/YYYY (government decisions)
  - Directiva UE ... / Regulamentul UE ...
  - Generic nr. NNN/YYYY in legislative context
  - Legea bugetului, Legea pensiilor, etc. (well-known law names)

Used by:
  - analyze_interventions.py (baseline extraction)
  - llm_agent.py / llm_session_topics.py (prompt enrichment + LLM output validation)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LawReference:
    """A single extracted law/bill reference."""
    raw_text: str
    canonical_id: str
    ref_type: str  # plx, lege, oug, og, hg, ordonanta, directiva, regulament
    number: str  # e.g. "107/1996"


def _norm_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.casefold()


# Compiled patterns: (regex, ref_type, canonical_template)
# Group 1 must capture the NNN/YYYY part where applicable.
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # PL-x variants: "PL-x nr. 211/2011", "PLx 211/2011", "PL-x211/2011"
    (re.compile(r"\bpl[-\s]?x\s*(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b", re.IGNORECASE),
     "plx", "PL-x {ref}"),

    # Legea nr. NNN/YYYY or Legea NNN/YYYY
    (re.compile(r"\blegea?\s+(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b", re.IGNORECASE),
     "lege", "Legea nr. {ref}"),

    # OUG nr. NNN/YYYY or "ordonanța de urgență nr. NNN/YYYY"
    (re.compile(r"\boug\s+(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b", re.IGNORECASE),
     "oug", "OUG nr. {ref}"),
    (re.compile(
        r"\bordonant[aă]\s+de\s+urgent[aă]\s+(?:a\s+guvernului\s+)?(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b",
        re.IGNORECASE),
     "oug", "OUG nr. {ref}"),

    # OG nr. NNN/YYYY or "ordonanța Guvernului nr. NNN/YYYY"
    (re.compile(r"\bog\s+(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b", re.IGNORECASE),
     "og", "OG nr. {ref}"),
    (re.compile(
        r"\bordonant[aă]\s+(?:guvernului\s+)?(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b",
        re.IGNORECASE),
     "og", "OG nr. {ref}"),

    # HG nr. NNN/YYYY or "Hotărârea Guvernului nr. NNN/YYYY"
    (re.compile(r"\bhg\s+(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b", re.IGNORECASE),
     "hg", "HG nr. {ref}"),
    (re.compile(
        r"\bhotararea\s+(?:guvernului\s+)?(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b",
        re.IGNORECASE),
     "hg", "HG nr. {ref}"),

    # Directiva UE NNN/YYYY or "Directiva (UE) NNN/YYYY"
    (re.compile(
        r"\bdirectiva\s+(?:\(?\s*(?:ue|ce)\s*\)?\s*)?(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b",
        re.IGNORECASE),
     "directiva", "Directiva UE {ref}"),

    # Regulamentul UE NNN/YYYY
    (re.compile(
        r"\bregulamentul\s+(?:\(?\s*(?:ue|ce)\s*\)?\s*)?(?:nr\.?\s*)?(\d+\s*/\s*\d{4})\b",
        re.IGNORECASE),
     "regulament", "Regulamentul UE {ref}"),

    # Codul ... (Codul fiscal, Codul penal, Codul muncii, etc.)
    # These don't have NNN/YYYY references, handled separately below.

    # Generic "nr. NNN/YYYY" in legislative context — only when preceded by
    # legislative keywords within 60 chars.
]

# Context keywords that make a bare "nr. NNN/YYYY" likely legislative.
_LEGISLATIVE_CONTEXT_WORDS = {
    "lege", "legea", "legii", "proiect", "proiectul", "proiectului",
    "ordonanta", "ordonanța", "ordonantei", "ordonanței",
    "hotarare", "hotărâre", "hotararea", "hotărârea",
    "regulament", "regulamentul", "directiva", "directivei",
    "articol", "articolul", "alineat", "alineatul",
    "amendament", "amendamentul", "comisie", "comisia",
    "raport", "raportul", "aviz", "avizul",
    "cod", "codul", "codului",
}

# Bare nr. NNN/YYYY pattern
_BARE_NR_PATTERN = re.compile(r"\bnr\.?\s*(\d+\s*/\s*\d{4})\b", re.IGNORECASE)


def _normalize_ref(ref: str) -> str:
    """Normalize whitespace in NNN/YYYY references."""
    return re.sub(r"\s*/\s*", "/", ref.strip())


def extract_law_references(text: str) -> list[LawReference]:
    """Extract all legislative references from text.

    Returns deduplicated references in order of appearance.
    """
    if not text:
        return []

    norm = _norm_text(text)
    seen_canonical: set[str] = set()
    results: list[LawReference] = []

    for pattern, ref_type, template in _PATTERNS:
        for match in pattern.finditer(norm):
            ref = _normalize_ref(match.group(1))
            canonical = template.format(ref=ref)
            if canonical.lower() in seen_canonical:
                continue
            seen_canonical.add(canonical.lower())
            results.append(LawReference(
                raw_text=match.group(0).strip(),
                canonical_id=canonical,
                ref_type=ref_type,
                number=ref,
            ))

    # Bare "nr. NNN/YYYY" with legislative context nearby.
    for match in _BARE_NR_PATTERN.finditer(norm):
        ref = _normalize_ref(match.group(1))
        # Check if any typed pattern already captured this number.
        if any(r.number == ref for r in results):
            continue

        start = max(0, match.start() - 80)
        context_window = norm[start:match.start()]
        context_words = set(re.findall(r"[a-z]{3,}", context_window))
        if context_words & _LEGISLATIVE_CONTEXT_WORDS:
            canonical = f"nr. {ref}"
            if canonical.lower() not in seen_canonical:
                seen_canonical.add(canonical.lower())
                results.append(LawReference(
                    raw_text=match.group(0).strip(),
                    canonical_id=canonical,
                    ref_type="generic",
                    number=ref,
                ))

    return results


@dataclass
class SessionLawIndex:
    """Per-session index mapping law references to speech positions."""
    session_id: str
    law_to_speeches: dict[str, list[int]] = field(default_factory=dict)
    speech_to_laws: dict[int, list[str]] = field(default_factory=dict)
    all_law_ids: list[str] = field(default_factory=list)

    def add(self, canonical_id: str, speech_index: int) -> None:
        self.law_to_speeches.setdefault(canonical_id, [])
        if speech_index not in self.law_to_speeches[canonical_id]:
            self.law_to_speeches[canonical_id].append(speech_index)
        self.speech_to_laws.setdefault(speech_index, [])
        if canonical_id not in self.speech_to_laws[speech_index]:
            self.speech_to_laws[speech_index].append(canonical_id)
        if canonical_id not in self.all_law_ids:
            self.all_law_ids.append(canonical_id)

    def format_for_prompt(self) -> str:
        """Format the law index as a text block for LLM prompts."""
        if not self.all_law_ids:
            return ""
        lines = ["## Pre-extracted law/bill references (from session text)"]
        for law_id in self.all_law_ids:
            speeches = self.law_to_speeches.get(law_id, [])
            lines.append(f"- {law_id} (mentioned in speech indices: {speeches})")
        lines.append(
            "NOTE: Use these verified references when attributing law_id. "
            "Do not hallucinate law IDs not listed here."
        )
        return "\n".join(lines)


def build_session_law_index(
    session_id: str,
    initial_notes: str,
    speeches: list[dict],
) -> SessionLawIndex:
    """Scan all speech text and initial_notes for law references.

    Args:
        session_id: The session identifier.
        initial_notes: The session's initial notes text.
        speeches: List of speech dicts, each having 'text' and optionally
                  'text2', 'text3' fields, indexed by position.

    Returns:
        SessionLawIndex with all discovered references.
    """
    index = SessionLawIndex(session_id=session_id)

    if initial_notes:
        for ref in extract_law_references(initial_notes):
            index.add(ref.canonical_id, -1)  # -1 = from initial notes

    for speech_idx, speech in enumerate(speeches):
        if not isinstance(speech, dict):
            continue
        parts = []
        for key in ("text", "text2", "text3"):
            value = speech.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        full_text = " ".join(parts)
        for ref in extract_law_references(full_text):
            index.add(ref.canonical_id, speech_idx)

    return index


@dataclass
class AgendaItem:
    """A single parsed agenda item from session initial_notes."""
    item_number: int | None
    title: str
    law_ids: list[str]


def parse_agenda_from_notes(initial_notes: str) -> list[AgendaItem]:
    """Parse initial_notes for numbered agenda items with law references.

    Romanian parliamentary session notes typically list agenda items as:
      1. Proiectul de Lege privind ... (PL-x 45/2025)
      2. Dezbaterea OUG nr. 114/2018
      ...
    or using dashes/bullets:
      - Raportul Comisiei ... Legea nr. 107/1996
    """
    if not initial_notes or not initial_notes.strip():
        return []

    items: list[AgendaItem] = []

    # Split on numbered items (1. / 1) / I. / a.) or bullet points (- / •)
    # Each "chunk" is one potential agenda item.
    item_pattern = re.compile(
        r"(?:^|\n)\s*"
        r"(?:"
        r"(\d+)\s*[\.\)\-]\s*"  # numbered: "1." / "1)" / "1-"
        r"|"
        r"[IVXivx]+\s*[\.\)]\s*"  # roman numerals
        r"|"
        r"[a-z]\s*[\.\)]\s*"  # lettered: "a." / "a)"
        r"|"
        r"[-•–]\s*"  # bullets
        r")"
        r"(.+?)(?=\n\s*(?:\d+\s*[\.\)\-]|[IVXivx]+\s*[\.\)]|[a-z]\s*[\.\)]|[-•–])|\Z)",
        re.DOTALL,
    )

    for match in item_pattern.finditer(initial_notes):
        num_str = match.group(1)
        item_number = int(num_str) if num_str else None
        raw_title = match.group(2).strip()

        # Clean up the title.
        title = re.sub(r"\s+", " ", raw_title).strip()
        if len(title) < 5:
            continue

        # Extract any law references from this agenda item.
        refs = extract_law_references(title)
        law_ids = [r.canonical_id for r in refs]

        items.append(AgendaItem(
            item_number=item_number,
            title=title[:200],
            law_ids=law_ids,
        ))

    # Also try a simpler split on newlines if the structured pattern found nothing.
    if not items:
        for line in initial_notes.split("\n"):
            line = line.strip()
            if len(line) < 10:
                continue
            refs = extract_law_references(line)
            if refs:
                # Strip leading numbering/bullets.
                clean_title = re.sub(r"^\s*(?:\d+[\.\)\-]|[IVXivx]+[\.\)]|[a-z][\.\)]|[-•–])\s*", "", line)
                items.append(AgendaItem(
                    item_number=None,
                    title=clean_title.strip()[:200],
                    law_ids=[r.canonical_id for r in refs],
                ))

    return items


def format_agenda_for_prompt(agenda: list[AgendaItem]) -> str:
    """Format parsed agenda items as text for LLM prompts."""
    if not agenda:
        return ""
    lines = ["## Session agenda (from initial notes)"]
    for item in agenda:
        num = f"{item.item_number}. " if item.item_number else "- "
        law_part = f" [{', '.join(item.law_ids)}]" if item.law_ids else ""
        lines.append(f"{num}{item.title}{law_part}")
    return "\n".join(lines)


def validate_law_ids(
    llm_law_ids: list[str],
    session_law_index: SessionLawIndex,
) -> list[str]:
    """Filter LLM-returned law IDs against the pre-extracted session index.

    Returns only law IDs that match (exact or partial) the session's
    pre-extracted references. Rejects hallucinated IDs.
    """
    if not llm_law_ids or not session_law_index.all_law_ids:
        return llm_law_ids  # nothing to validate against

    valid_numbers = set()
    for law_id in session_law_index.all_law_ids:
        nums = re.findall(r"\d+/\d{4}", law_id)
        valid_numbers.update(nums)

    valid_canonical_lower = {lid.lower() for lid in session_law_index.all_law_ids}

    validated: list[str] = []
    for lid in llm_law_ids:
        lid_lower = lid.lower().strip()

        # Exact match against canonical IDs.
        if lid_lower in valid_canonical_lower:
            validated.append(lid)
            continue

        # Partial: check if the NNN/YYYY part matches any known reference.
        lid_nums = re.findall(r"\d+/\d{4}", lid_lower)
        if lid_nums and any(n in valid_numbers for n in lid_nums):
            validated.append(lid)
            continue

        # Substring containment.
        if any(lid_lower in vc or vc in lid_lower for vc in valid_canonical_lower if len(vc) >= 8):
            validated.append(lid)
            continue

    return validated
