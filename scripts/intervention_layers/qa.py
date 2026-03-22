from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(ch))


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]+", text))


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _has_substantive_yes(layer_a: dict) -> bool:
    return any(
        layer_a.get(k) == "yes"
        for k in ("policy_proposal", "policy_analysis", "legislative_engagement", "public_interest_orientation")
    )


def _reasoning_inconsistent(layer_a: dict, layer_b: dict) -> bool:
    label = layer_b.get("constructiveness_label")
    if label == "neutral":
        return layer_a.get("procedural_content") != "yes" and _has_substantive_yes(layer_a)
    if label == "constructive":
        return (
            layer_a.get("partisan_rhetoric") == "yes"
            and layer_a.get("policy_proposal") != "yes"
            and layer_a.get("policy_analysis") != "yes"
            and layer_a.get("legislative_engagement") != "yes"
        )
    if label == "non_constructive":
        return layer_a.get("partisan_rhetoric") != "yes" and _has_substantive_yes(layer_a)
    return False


def _label_weakly_supported(layer_a: dict, layer_b: dict) -> bool:
    label = layer_b.get("constructiveness_label")
    if label == "constructive":
        return not (
            layer_a.get("policy_proposal") == "yes"
            or layer_a.get("policy_analysis") == "yes"
            or layer_a.get("legislative_engagement") == "yes"
        )
    if label == "neutral":
        return layer_a.get("procedural_content") != "yes"
    if label == "non_constructive":
        return layer_a.get("partisan_rhetoric") != "yes"
    return True


def _speech_mentions_session_topic(speech_text: str, session_topics: list) -> bool:
    key = _norm(speech_text or "")
    if not key:
        return False
    for topic in session_topics or []:
        if isinstance(topic, dict):
            label = str(topic.get("label", "")).strip()
        else:
            label = str(topic).strip()
        if not label:
            continue
        label_key = _norm(label)
        tokens = [t for t in re.split(r"[^a-z0-9]+", label_key) if len(t) >= 4]
        if not tokens:
            continue
        matches = sum(1 for t in tokens if t in key)
        if matches >= 2 or (len(tokens) == 1 and tokens[0] in key):
            return True
    return False


def _is_clear_procedural_short(speech_text: str, layer_a: dict) -> bool:
    """Check if a short speech is clearly procedural and handled by deterministic rules."""
    if layer_a.get("procedural_content") != "yes":
        return False
    if _has_substantive_yes(layer_a):
        return False
    norm = _norm(speech_text)
    procedural_markers = (
        "multumesc", "multumim", "da", "nu", "prezent", "absent",
        "va rog", "are cuvantul", "declar",
    )
    return any(m in norm for m in procedural_markers)


def evaluate_qa_triggers(
    layer_a: dict,
    layer_b: dict,
    speech_text: str,
    session_topics: list,
) -> list[str]:
    """Return trigger reasons for Layer C QA.

    Tightened thresholds (Phase 2.4):
    - low_confidence: 0.65 → 0.70 (fewer unnecessary QA triggers)
    - very_short_speech: suppressed when already handled by deterministic
      rules (procedural shorts are correctly classified by Layer B alone)
    """
    reasons: list[str] = []
    confidence = float(layer_b.get("confidence", 0.0))

    # Raised from 0.65 to 0.70: moderate-confidence predictions are usually
    # correct and don't benefit from a Layer C pass.
    if confidence < 0.70:
        reasons.append("low_confidence")

    if layer_a.get("primary_function") == "mixed":
        reasons.append("primary_function_mixed")
    if layer_a.get("policy_analysis") == "yes" and layer_a.get("partisan_rhetoric") == "yes":
        reasons.append("analysis_and_partisan_conflict")
    if layer_a.get("procedural_content") == "partial" and _has_substantive_yes(layer_a):
        reasons.append("procedural_partial_with_substantive_yes")

    # Only trigger very_short_speech when the speech is NOT clearly procedural.
    # Short procedural speeches (greetings, "Da.", "Mulțumesc.") are already
    # well-handled by deterministic rules and Layer B; sending them to Layer C
    # wastes an LLM call without accuracy gain.
    words = _word_count(speech_text)
    sentences = _sentence_count(speech_text)
    if (words <= 25 or sentences <= 2) and not _is_clear_procedural_short(speech_text, layer_a):
        reasons.append("very_short_speech")

    if _reasoning_inconsistent(layer_a, layer_b):
        reasons.append("layer_b_inconsistent_with_layer_a")
    if (not layer_b.get("topics")) and _speech_mentions_session_topic(speech_text, session_topics):
        reasons.append("missing_topics_despite_topic_reference")
    if _label_weakly_supported(layer_a, layer_b):
        reasons.append("label_weakly_supported_by_layer_a")
    # Preserve stable order and uniqueness
    unique: list[str] = []
    for r in reasons:
        if r not in unique:
            unique.append(r)
    return unique

