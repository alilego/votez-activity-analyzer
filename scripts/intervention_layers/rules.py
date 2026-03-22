from __future__ import annotations

import re
import unicodedata


def _strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _text_key(text: str) -> str:
    return _strip_diacritics(text.casefold()).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text))


# ---- Greeting / thanks tokens (diacritics-stripped, lowered) ----
_GREETING_THANKS_TOKENS = {
    "multumesc", "multumim", "va rog", "buna", "buna ziua",
    "buna dimineata", "buna seara",
}

_VOTE_PATTERNS = (
    "supun la vot",
    "supunem la vot",
    "trecem la vot",
    "cine este pentru",
    "cine este contra",
    "cine se abtine",
    "votul a fost",
    "votul final",
    "ramane la votul final",
    "ramane a votul final",
)

_COMMITTEE_REPORT_MARKERS = (
    "raportul comisiei",
    "raport comun",
    "raport asupra",
    "in conformitate cu prevederile art",
    "comisia pentru",
)

_COMMITTEE_REPORT_STRUCTURE = re.compile(
    r"(?:în conformitate cu prevederile|raport\w*\s+(?:comun\s+)?(?:asupra|privind))"
    r".+(?:comisi[aei]|sesizat)",
    re.IGNORECASE | re.DOTALL,
)


def apply_pre_llm_shortcuts(
    speech_text: str,
    raw_speaker: str = "",
    session_chairs: set[str] | None = None,
    interruption_type: str | None = None,
) -> dict | None:
    """
    Deterministic classification that bypasses the LLM entirely.

    Returns a dict with synthetic Layer A signals + final decision if a shortcut
    applies, or None if the speech should go through the normal LLM pipeline.

    When ``interruption_type`` is ``"procedure_violation"``, all shortcuts are
    suppressed — the speech must go through full LLM evaluation to assess
    whether its content benefits citizens despite the procedural infraction.

    Shortcuts:
    1. Ultra-short greetings/thanks (≤10 words) → neutral
    2. Vote announcement patterns → neutral
    3. Very short chair name-calls (≤5 words, "Domnul/Doamna X") → neutral
    """
    if interruption_type == "procedure_violation":
        return None

    text = (speech_text or "").strip()
    key = _text_key(text)
    wc = _word_count(text)

    # 1) Ultra-short greetings / thanks (≤10 words)
    if wc <= 10 and wc >= 1:
        for token in _GREETING_THANKS_TOKENS:
            if token in key:
                return _build_pre_llm_neutral(
                    text,
                    reason="pre_llm_shortcut: ultra-short greeting/thanks",
                    confidence=0.95,
                )

    # 2) Vote announcement patterns (any length ≤60 words)
    if wc <= 60:
        for pattern in _VOTE_PATTERNS:
            if pattern in key:
                return _build_pre_llm_neutral(
                    text,
                    reason="pre_llm_shortcut: vote announcement",
                    confidence=0.95,
                )

    # 3) Ultra-short chair name-calls: "Domnul X." / "Doamna Y."
    if wc <= 5:
        name_call = re.match(
            r"^(Domnul|Doamna)\s+[A-ZĂÂÎȘȚŞŢ]",
            text,
        )
        if name_call:
            return _build_pre_llm_neutral(
                text,
                reason="pre_llm_shortcut: chair name-call",
                confidence=0.95,
            )

    # 4) Ultra-short floor responses: "Da.", "Nu.", "Prezent.", etc.
    if wc <= 3:
        short_tokens = {"da", "nu", "prezent", "absent", "abtinere", "pentru", "contra"}
        words_lower = {_text_key(w) for w in re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text)}
        if words_lower and words_lower <= short_tokens:
            return _build_pre_llm_neutral(
                text,
                reason="pre_llm_shortcut: ultra-short floor response",
                confidence=0.95,
            )

    return None


def _build_pre_llm_neutral(text: str, reason: str, confidence: float = 0.95) -> dict:
    """Build a complete synthetic result for a pre-LLM neutral shortcut."""
    words = re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text)
    quote = " ".join(words[:10]).strip()[:80] if words else ""
    return {
        "layer_a": {
            "speech_index": -1,
            "policy_proposal": "no",
            "policy_analysis": "no",
            "public_interest_orientation": "no",
            "partisan_rhetoric": "no",
            "legislative_engagement": "no",
            "procedural_content": "yes",
            "argumentation_quality": "none",
            "primary_function": "procedural",
            "reasoning": "Intervenție procedurală scurtă, fără conținut substanțial.",
            "evidence_quote": quote,
        },
        "decision": {
            "speech_index": -1,
            "constructiveness_label": "neutral",
            "confidence": confidence,
            "topics": [],
            "reasoning": "Intervenție procedurală scurtă, fără conținut substanțial.",
            "evidence_quote": quote,
        },
        "reason": reason,
        "qa_action": "confirmed",
    }


def detect_committee_report(speech_text: str) -> bool:
    """Detect if a speech is a committee report reading (→ constructive candidate)."""
    text = (speech_text or "").strip()
    key = _text_key(text)
    wc = _word_count(text)
    if wc < 50:
        return False
    marker_count = sum(1 for m in _COMMITTEE_REPORT_MARKERS if m in key)
    if marker_count < 2:
        return False
    if _COMMITTEE_REPORT_STRUCTURE.search(text):
        return True
    return False


def detect_session_chair_procedural(
    speech_text: str,
    raw_speaker: str,
    session_chairs: set[str] | None = None,
) -> bool:
    """
    Detect if a speech is a session chair/president procedural line.

    Returns True if the speaker is a known session chair and the content
    is procedural (≤40 words with procedural markers).
    """
    if not session_chairs:
        return False
    is_chair = any(name in raw_speaker for name in session_chairs)
    if not is_chair:
        return False
    wc = _word_count(speech_text or "")
    if wc > 40:
        return False
    key = _text_key(speech_text or "")
    proc_markers = (
        "va rog", "multumesc", "ramane la vot", "votul final",
        "dezbateri generale", "ordinea de zi", "trecem la",
        "cuvantul", "interventii", "nefiind amendamente",
    )
    return any(m in key for m in proc_markers)


def extract_session_chairs(initial_notes: str) -> set[str]:
    """Extract session chair names from initial_notes text."""
    chairs: set[str] = set()
    for m in re.finditer(
        r"(?:domnul|doamna)\s+(?:deputat\s+|senator\s+)?"
        r"([A-ZĂÂÎȘȚ][a-zăâîșț]+(?:[-][A-ZĂÂÎȘȚ]?[a-zăâîșț]+)*"
        r"(?:\s+[A-ZĂÂÎȘȚ][a-zăâîșț]+(?:[-][A-ZĂÂÎȘȚ]?[a-zăâîșț]+)*)*)",
        initial_notes or "",
    ):
        name = m.group(1).strip()
        if name and len(name) > 3:
            chairs.add(name)
    return chairs


def apply_deterministic_rules(
    layer_a: dict,
    speech_text: str = "",
    is_session_chair: bool = False,
) -> dict:
    """Return deterministic shortcuts/candidates from Layer A signals.

    Optional ``speech_text`` enables text-based heuristics (committee report
    detection). ``is_session_chair`` biases short procedural speeches from the
    session chair toward neutral.
    """
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

    # 1) Neutral shortcut: procedural with no substance
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

    # 1b) Session chair procedural bias: short speeches from the chair with
    # procedural=partial and no strong substantive signals → neutral shortcut
    if is_session_chair and _word_count(speech_text) <= 40:
        if (
            procedural_content in ("yes", "partial")
            and policy_proposal != "yes"
            and policy_analysis != "yes"
            and legislative_engagement != "yes"
            and partisan_rhetoric != "yes"
        ):
            out["shortcut_label"] = "neutral"
            out["shortcut_confidence"] = 0.85
            out["shortcut_reason"] = (
                "neutral_shortcut: session chair short procedural speech"
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

    # 4) Committee report → constructive candidate
    if speech_text and detect_committee_report(speech_text):
        if "constructive" not in out["candidate_labels"]:
            out["candidate_labels"].append("constructive")

    return out
