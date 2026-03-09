from __future__ import annotations

from typing import Any


YES_PARTIAL_NO = {"yes", "partial", "no"}
ARG_QUALITY = {"strong", "weak", "none"}
PRIMARY_FUNCTIONS = {
    "procedural",
    "substantive_support",
    "substantive_opposition",
    "partisan_attack",
    "symbolic_political_statement",
    "mixed",
}
FINAL_LABELS = {"constructive", "neutral", "non_constructive"}
QA_ACTIONS = {"confirmed", "revised_label", "revised_topics", "revised_confidence"}


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer")


def _as_choice(value: Any, allowed: set[str], field: str) -> str:
    raw = str(value or "").strip().lower()
    if raw not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}")
    return raw


def _as_topics(value: Any, max_topics: int = 3) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("topics must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("topics entries must be strings")
        t = item.strip()
        if not t:
            continue
        if t.casefold() in seen:
            continue
        seen.add(t.casefold())
        out.append(t)
        if len(out) >= max_topics:
            break
    return out


def _as_reasoning(value: Any) -> str:
    return str(value or "").strip()


def _as_quote(value: Any) -> str:
    return str(value or "").strip().strip("\"“”")


def validate_layer_a_item(item: dict) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Layer A output must be an object")
    return {
        "speech_index": _as_int(item.get("speech_index"), "speech_index"),
        "policy_proposal": _as_choice(item.get("policy_proposal"), YES_PARTIAL_NO, "policy_proposal"),
        "policy_analysis": _as_choice(item.get("policy_analysis"), YES_PARTIAL_NO, "policy_analysis"),
        "public_interest_orientation": _as_choice(
            item.get("public_interest_orientation"), YES_PARTIAL_NO, "public_interest_orientation"
        ),
        "partisan_rhetoric": _as_choice(item.get("partisan_rhetoric"), YES_PARTIAL_NO, "partisan_rhetoric"),
        "legislative_engagement": _as_choice(
            item.get("legislative_engagement"), YES_PARTIAL_NO, "legislative_engagement"
        ),
        "procedural_content": _as_choice(item.get("procedural_content"), YES_PARTIAL_NO, "procedural_content"),
        "argumentation_quality": _as_choice(item.get("argumentation_quality"), ARG_QUALITY, "argumentation_quality"),
        "primary_function": _as_choice(item.get("primary_function"), PRIMARY_FUNCTIONS, "primary_function"),
        "reasoning": _as_reasoning(item.get("reasoning")),
        "evidence_quote": _as_quote(item.get("evidence_quote")),
    }


def validate_layer_b_item(item: dict, max_topics: int = 3) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Layer B output must be an object")
    try:
        confidence = float(item.get("confidence"))
    except (TypeError, ValueError):
        raise ValueError("confidence must be a number")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")
    return {
        "speech_index": _as_int(item.get("speech_index"), "speech_index"),
        "constructiveness_label": _as_choice(
            item.get("constructiveness_label"),
            FINAL_LABELS,
            "constructiveness_label",
        ),
        "confidence": confidence,
        "topics": _as_topics(item.get("topics"), max_topics=max_topics),
        "reasoning": _as_reasoning(item.get("reasoning")),
        "evidence_quote": _as_quote(item.get("evidence_quote")),
    }


def validate_layer_c_item(item: dict, max_topics: int = 3) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Layer C output must be an object")
    try:
        final_confidence = float(item.get("final_confidence"))
    except (TypeError, ValueError):
        raise ValueError("final_confidence must be a number")
    if not (0.0 <= final_confidence <= 1.0):
        raise ValueError("final_confidence must be between 0.0 and 1.0")
    return {
        "speech_index": _as_int(item.get("speech_index"), "speech_index"),
        "final_label": _as_choice(item.get("final_label"), FINAL_LABELS, "final_label"),
        "final_confidence": final_confidence,
        "topics": _as_topics(item.get("topics"), max_topics=max_topics),
        "reasoning": _as_reasoning(item.get("reasoning")),
        "evidence_quote": _as_quote(item.get("evidence_quote")),
        "qa_action": _as_choice(item.get("qa_action"), QA_ACTIONS, "qa_action"),
    }

