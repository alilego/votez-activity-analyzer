from __future__ import annotations


def build_shortcut_decision(layer_a: dict, shortcut: dict) -> dict | None:
    """Build a direct final decision when a deterministic shortcut applies."""
    if shortcut.get("shortcut_label") != "neutral":
        return None
    return {
        "speech_index": layer_a["speech_index"],
        "constructiveness_label": "neutral",
        "confidence": float(shortcut.get("shortcut_confidence") or 0.9),
        "topics": [],
        "reasoning": str(layer_a.get("reasoning") or "").strip(),
        "evidence_quote": str(layer_a.get("evidence_quote") or "").strip(),
    }


def merge_for_compatibility(
    layer_a: dict,
    decision: dict,
    qa_action: str = "confirmed",
) -> dict:
    """
    Compose backward-compatible final payload:
    - Layer A rubric signals
    - final decision from Layer B or Layer C
    """
    return {
        "speech_index": int(decision.get("speech_index", layer_a.get("speech_index", -1))),
        "constructiveness_label": str(decision.get("constructiveness_label", "")),
        "policy_proposal": str(layer_a.get("policy_proposal", "partial")),
        "policy_analysis": str(layer_a.get("policy_analysis", "partial")),
        "public_interest_orientation": str(layer_a.get("public_interest_orientation", "partial")),
        "partisan_rhetoric": str(layer_a.get("partisan_rhetoric", "partial")),
        "legislative_engagement": str(layer_a.get("legislative_engagement", "partial")),
        "procedural_content": str(layer_a.get("procedural_content", "partial")),
        "argumentation_quality": str(layer_a.get("argumentation_quality", "weak")),
        "debate_advancement": str(layer_a.get("debate_advancement", "partial")),
        "confidence": float(decision.get("confidence", 0.5)),
        "topics": list(decision.get("topics", [])),
        "reasoning": str(decision.get("reasoning", "")).strip(),
        "evidence_quote": str(decision.get("evidence_quote", "")).strip(),
        "_qa_action": qa_action,
    }


def decision_from_layer_b(layer_b: dict) -> dict:
    return {
        "speech_index": int(layer_b["speech_index"]),
        "constructiveness_label": str(layer_b["constructiveness_label"]),
        "confidence": float(layer_b["confidence"]),
        "topics": list(layer_b.get("topics", [])),
        "reasoning": str(layer_b.get("reasoning", "")).strip(),
        "evidence_quote": str(layer_b.get("evidence_quote", "")).strip(),
    }


def decision_from_layer_c(layer_c: dict) -> dict:
    return {
        "speech_index": int(layer_c["speech_index"]),
        "constructiveness_label": str(layer_c["final_label"]),
        "confidence": float(layer_c["final_confidence"]),
        "topics": list(layer_c.get("topics", [])),
        "reasoning": str(layer_c.get("reasoning", "")).strip(),
        "evidence_quote": str(layer_c.get("evidence_quote", "")).strip(),
    }
