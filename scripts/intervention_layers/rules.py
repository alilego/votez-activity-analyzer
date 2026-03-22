from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(ch)
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text))


# Greeting/thanks patterns that indicate purely procedural ultra-short speeches.
_GREETING_PATTERNS = re.compile(
    r"^(?:multumesc|multumim|va multumesc|va multumim|buna ziua|buna seara"
    r"|buna dimineata|salut|stimati colegi|stimate colege|domnule presedinte"
    r"|doamna presedinta|mulțumesc|mulțumim|vă mulțumesc|vă mulțumim"
    r"|bună ziua|bună seara|bună dimineața"
    r"|stimați colegi|stimate colege)(?:[.,!?\s]|$)",
    re.IGNORECASE,
)

# Vote announcement patterns.
_VOTE_PATTERNS = re.compile(
    r"(?:supun (?:la )?vot|cine este pentru|cine este contra|cine se abtine"
    r"|votul a fost|s-a adoptat|nu s-a adoptat|a fost respins"
    r"|a fost adoptat|procedura de vot|va rog sa votati|va rog sa votați"
    r"|supunem? votului"
    r"|cu unanimitate de voturi|cu majoritate de voturi|rezultatul votului"
    r"|cu \d+ voturi|pentru.*contra.*abtiner)",
    re.IGNORECASE,
)

# Committee report reading patterns.
_COMMITTEE_REPORT_PATTERNS = re.compile(
    r"(?:raportul comisiei|raport(?:ul)?\s+comun|comisia\s+(?:pentru|de)\s+\w+"
    r"|raportul\s+asupra\s+proiectul|propunem?\s+(?:adoptarea|respingerea)\s+proiectului"
    r"|aviz\s+(?:favorabil|nefavorabil)\s+(?:de\s+la|al)\s+comisi"
    r"|camera\s+(?:decizionala|decisională))",
    re.IGNORECASE,
)

# Session chair/president procedural patterns.
_CHAIR_PROCEDURAL_PATTERNS = re.compile(
    r"(?:are cuvantul|are cuvântul|dau cuvantul|dau cuvântul"
    r"|declar (?:deschisa|închisă|deschisă|inchisa) sedinta|declar (?:deschisă|deschisa) ședința"
    r"|trecem la (?:urmatorul|următorul) punct|trecem la punctul"
    r"|se pregateste|se pregătește|este cineva impotriva|este cineva împotrivă"
    r"|va rog frumos|vă rog frumos|va rog sa luati|vă rog să luați"
    r"|ordine de zi|ordinea de zi)",
    re.IGNORECASE,
)


def apply_pre_llm_shortcuts(
    speech_text: str,
    raw_speaker: str = "",
) -> dict | None:
    """Apply text-based deterministic shortcuts BEFORE any LLM call.

    Returns a dict with shortcut classification, or None if no shortcut applies.
    The returned dict matches the final payload schema so it can be used directly.
    """
    if not speech_text:
        return None

    norm = _norm(speech_text)
    words = _word_count(speech_text)

    # Ultra-short speeches (<=10 words) that are greetings/thanks → neutral.
    if words <= 10:
        if _GREETING_PATTERNS.search(norm):
            return {
                "shortcut_label": "neutral",
                "shortcut_confidence": 0.95,
                "shortcut_reason": "pre_llm: ultra-short greeting/thanks",
            }
        # Very short non-greeting: "Da.", "Nu.", "Prezent." etc.
        if words <= 3 and not any(
            k in norm for k in ("hot", "hoti", "rusine", "mincinos", "penal", "tradator")
        ):
            return {
                "shortcut_label": "neutral",
                "shortcut_confidence": 0.92,
                "shortcut_reason": "pre_llm: ultra-short procedural reply",
            }

    # Vote announcement patterns → neutral.
    if _VOTE_PATTERNS.search(norm):
        has_substantive = any(
            k in norm for k in (
                "propun", "analiz", "consider", "amendament",
                "problema", "soluți", "soluti", "argument",
            )
        )
        if not has_substantive:
            return {
                "shortcut_label": "neutral",
                "shortcut_confidence": 0.92,
                "shortcut_reason": "pre_llm: vote announcement pattern",
            }

    # Session chair procedural lines → neutral (only for short speeches).
    if words <= 50 and _CHAIR_PROCEDURAL_PATTERNS.search(norm):
        return {
            "shortcut_label": "neutral",
            "shortcut_confidence": 0.90,
            "shortcut_reason": "pre_llm: chair procedural line",
        }

    # Committee report with formal structure → constructive candidate
    # (not a shortcut, but flags it for the LLM to consider).
    if _COMMITTEE_REPORT_PATTERNS.search(norm) and words >= 20:
        return {
            "shortcut_label": None,
            "shortcut_confidence": None,
            "shortcut_reason": "",
            "candidate_labels": ["constructive"],
            "is_committee_report": True,
        }

    return None


def apply_deterministic_rules(layer_a: dict) -> dict:
    """Return deterministic shortcuts/candidates from Layer A signals."""
    policy_proposal = layer_a.get("policy_proposal")
    policy_analysis = layer_a.get("policy_analysis")
    legislative_engagement = layer_a.get("legislative_engagement")
    procedural_content = layer_a.get("procedural_content")
    partisan_rhetoric = layer_a.get("partisan_rhetoric")
    argumentation_quality = layer_a.get("argumentation_quality")
    public_interest_orientation = layer_a.get("public_interest_orientation")

    out = {
        "shortcut_label": None,
        "shortcut_confidence": None,
        "shortcut_reason": "",
        "candidate_labels": [],
    }

    # 1) Neutral shortcut
    neutral_shortcut = (
        procedural_content == "yes"
        and policy_proposal == "no"
        and policy_analysis == "no"
        and legislative_engagement == "no"
    )
    if neutral_shortcut:
        out["shortcut_label"] = "neutral"
        out["shortcut_confidence"] = 0.9
        out["shortcut_reason"] = (
            "neutral_shortcut: procedural_content=yes with no substantive proposal/analysis/legislative engagement"
        )
        out["candidate_labels"] = ["neutral"]
        return out

    # 2) Non-constructive shortcut candidate
    if (
        partisan_rhetoric == "yes"
        and policy_proposal == "no"
        and policy_analysis == "no"
        and argumentation_quality == "none"
    ):
        out["candidate_labels"].append("non_constructive")

    # 3) Constructive shortcut candidate
    if (
        (policy_proposal == "yes" or policy_analysis == "yes" or legislative_engagement == "yes")
        and procedural_content != "yes"
    ):
        out["candidate_labels"].append("constructive")

    return out
