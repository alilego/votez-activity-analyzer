from __future__ import annotations


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
