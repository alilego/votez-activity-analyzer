#!/usr/bin/env python3
"""
Analyze extracted adopted-law text into citizen-friendly impact summaries.

The pipeline deliberately separates:
Stage A: factual legal extraction
Stage B: citizen interpretation
Stage C: validation/critic producing the final LawAnalysis
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from init_db import DEFAULT_DB_PATH, init_db
from model_profiles import DEFAULT_MODEL_OLLAMA, infer_ollama_num_ctx
from openai_runtime import create_chat_completion, resolve_openai_service_tier


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_OPENAI_LAW_ANALYSIS = "gpt-5-mini"
DEFAULT_MAX_TEXT_CHARS = 50000
DEFAULT_MAX_RETRIES = 3
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

ImpactType = Literal[
    "symbolic",
    "administrative",
    "financial",
    "rights",
    "restrictions",
    "public_services",
    "safety",
    "justice",
    "education",
    "healthcare",
    "taxation",
    "environment",
    "unclear",
]
ImpactDirection = Literal["positive", "negative", "mixed", "neutral", "unclear"]

IMPACT_TYPES = {
    "symbolic",
    "administrative",
    "financial",
    "rights",
    "restrictions",
    "public_services",
    "safety",
    "justice",
    "education",
    "healthcare",
    "taxation",
    "environment",
    "unclear",
}
IMPACT_DIRECTIONS = {"positive", "negative", "mixed", "neutral", "unclear"}


@dataclass
class LawInput:
    law_id: str
    title: str
    status: str | None
    source_url: str
    extracted_text: str
    initiators: list[str] = field(default_factory=list)
    adoption_date: str | None = None
    chamber: str | None = None
    document_type: str | None = None


@dataclass
class LawAnalysis:
    law_id: str
    original_title: str
    plain_language_title: str
    one_sentence_summary: str
    what_changes: list[str]
    who_is_affected: list[str]
    practical_impact: list[str]
    impact_type: ImpactType
    impact_direction: ImpactDirection
    citizen_relevance_score: int
    public_interest_score: int
    impact_magnitude_score: int
    clarity_score: int
    implementation_feasibility_score: int
    evidence_quality_score: int
    risk_of_narrow_interest_score: int
    symbolic_only: bool
    requires_budget: bool | None
    creates_new_bureaucracy: bool | None
    affects_many_citizens: bool | None
    affected_legal_acts: list[str]
    institutions_responsible: list[str]
    obligations_created: list[str]
    rights_created: list[str]
    penalties_or_costs: list[str]
    implementation_date: str | None
    possible_risks_or_criticism: list[str]
    missing_information: list[str]
    confidence_score: int
    source_fragments: list[str]


FACTUAL_SCHEMA: dict[str, Any] = {
    "affected_legal_acts": list,
    "changed_articles": list,
    "new_obligations": list,
    "new_rights": list,
    "penalties_costs_or_sanctions": list,
    "institutions_responsible": list,
    "implementation_date": (str, type(None)),
    "target_groups": list,
    "budget_implications": (str, type(None)),
    "mostly_symbolic": bool,
    "source_fragments": list,
    "missing_information": list,
}

INTERPRETATION_SCHEMA: dict[str, Any] = {
    "plain_language_title": str,
    "one_sentence_summary": str,
    "what_changes": list,
    "who_is_affected": list,
    "practical_impact": list,
    "impact_type": str,
    "impact_direction": str,
    "citizen_relevance_score": int,
    "public_interest_score": int,
    "impact_magnitude_score": int,
    "clarity_score": int,
    "implementation_feasibility_score": int,
    "evidence_quality_score": int,
    "risk_of_narrow_interest_score": int,
    "symbolic_only": bool,
    "requires_budget": (bool, type(None)),
    "creates_new_bureaucracy": (bool, type(None)),
    "affects_many_citizens": (bool, type(None)),
    "possible_risks_or_criticism": list,
    "missing_information": list,
    "source_fragments": list,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_code_fence(raw)
    if not text:
        raise ValueError("LLM returned an empty response instead of a JSON object")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object")
    return parsed


def _require_fields(payload: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    for key, expected_type in schema.items():
        if key not in payload:
            raise ValueError(f"{label}.{key} is missing")
        if not isinstance(payload[key], expected_type):
            raise ValueError(f"{label}.{key} has invalid type")


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _score(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer 1-5")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer 1-5") from exc
    if score < 1 or score > 5:
        raise ValueError(f"{field_name} must be between 1 and 5")
    return score


def validate_factual_extraction(payload: dict[str, Any]) -> dict[str, Any]:
    _require_fields(payload, FACTUAL_SCHEMA, "factual_extraction")
    normalized = dict(payload)
    for key in (
        "affected_legal_acts",
        "changed_articles",
        "new_obligations",
        "new_rights",
        "penalties_costs_or_sanctions",
        "institutions_responsible",
        "target_groups",
        "source_fragments",
        "missing_information",
    ):
        normalized[key] = _string_list(payload[key], key)
    return normalized


def validate_interpretation(payload: dict[str, Any]) -> dict[str, Any]:
    _require_fields(payload, INTERPRETATION_SCHEMA, "citizen_interpretation")
    impact_type = str(payload["impact_type"]).strip()
    if impact_type not in IMPACT_TYPES:
        raise ValueError("impact_type is not one of the allowed values")
    impact_direction = str(payload["impact_direction"]).strip()
    if impact_direction not in IMPACT_DIRECTIONS:
        raise ValueError("impact_direction is not one of the allowed values")
    normalized = dict(payload)
    for key in (
        "what_changes",
        "who_is_affected",
        "practical_impact",
        "possible_risks_or_criticism",
        "missing_information",
        "source_fragments",
    ):
        normalized[key] = _string_list(payload[key], key)
    for key in (
        "citizen_relevance_score",
        "public_interest_score",
        "impact_magnitude_score",
        "clarity_score",
        "implementation_feasibility_score",
        "evidence_quality_score",
        "risk_of_narrow_interest_score",
    ):
        normalized[key] = _score(payload[key], key)
    return normalized


def validate_law_analysis(payload: dict[str, Any]) -> LawAnalysis:
    required = {
        "law_id": str,
        "original_title": str,
        "plain_language_title": str,
        "one_sentence_summary": str,
        "what_changes": list,
        "who_is_affected": list,
        "practical_impact": list,
        "impact_type": str,
        "impact_direction": str,
        "citizen_relevance_score": int,
        "public_interest_score": int,
        "impact_magnitude_score": int,
        "clarity_score": int,
        "implementation_feasibility_score": int,
        "evidence_quality_score": int,
        "risk_of_narrow_interest_score": int,
        "symbolic_only": bool,
        "requires_budget": (bool, type(None)),
        "creates_new_bureaucracy": (bool, type(None)),
        "affects_many_citizens": (bool, type(None)),
        "affected_legal_acts": list,
        "institutions_responsible": list,
        "obligations_created": list,
        "rights_created": list,
        "penalties_or_costs": list,
        "implementation_date": (str, type(None)),
        "possible_risks_or_criticism": list,
        "missing_information": list,
        "confidence_score": int,
        "source_fragments": list,
    }
    _require_fields(payload, required, "law_analysis")
    impact_type = str(payload["impact_type"]).strip()
    if impact_type not in IMPACT_TYPES:
        raise ValueError("impact_type is not one of the allowed values")
    impact_direction = str(payload["impact_direction"]).strip()
    if impact_direction not in IMPACT_DIRECTIONS:
        raise ValueError("impact_direction is not one of the allowed values")
    return LawAnalysis(
        law_id=str(payload["law_id"]).strip(),
        original_title=str(payload["original_title"]).strip(),
        plain_language_title=str(payload["plain_language_title"]).strip(),
        one_sentence_summary=str(payload["one_sentence_summary"]).strip(),
        what_changes=_string_list(payload["what_changes"], "what_changes"),
        who_is_affected=_string_list(payload["who_is_affected"], "who_is_affected"),
        practical_impact=_string_list(payload["practical_impact"], "practical_impact"),
        impact_type=impact_type,  # type: ignore[arg-type]
        impact_direction=impact_direction,  # type: ignore[arg-type]
        citizen_relevance_score=_score(payload["citizen_relevance_score"], "citizen_relevance_score"),
        public_interest_score=_score(payload["public_interest_score"], "public_interest_score"),
        impact_magnitude_score=_score(payload["impact_magnitude_score"], "impact_magnitude_score"),
        clarity_score=_score(payload["clarity_score"], "clarity_score"),
        implementation_feasibility_score=_score(payload["implementation_feasibility_score"], "implementation_feasibility_score"),
        evidence_quality_score=_score(payload["evidence_quality_score"], "evidence_quality_score"),
        risk_of_narrow_interest_score=_score(payload["risk_of_narrow_interest_score"], "risk_of_narrow_interest_score"),
        symbolic_only=bool(payload["symbolic_only"]),
        requires_budget=payload["requires_budget"],
        creates_new_bureaucracy=payload["creates_new_bureaucracy"],
        affects_many_citizens=payload["affects_many_citizens"],
        affected_legal_acts=_string_list(payload["affected_legal_acts"], "affected_legal_acts"),
        institutions_responsible=_string_list(payload["institutions_responsible"], "institutions_responsible"),
        obligations_created=_string_list(payload["obligations_created"], "obligations_created"),
        rights_created=_string_list(payload["rights_created"], "rights_created"),
        penalties_or_costs=_string_list(payload["penalties_or_costs"], "penalties_or_costs"),
        implementation_date=payload["implementation_date"],
        possible_risks_or_criticism=_string_list(payload["possible_risks_or_criticism"], "possible_risks_or_criticism"),
        missing_information=_string_list(payload["missing_information"], "missing_information"),
        confidence_score=_score(payload["confidence_score"], "confidence_score"),
        source_fragments=_string_list(payload["source_fragments"], "source_fragments"),
    )


def format_reader_card(analysis: LawAnalysis) -> str:
    summary = build_reader_summary(analysis)
    bullets = "\n".join(f"- {item}" for item in summary["practical_impact"])
    return (
        f"Title:\n{summary['title']}\n\n"
        f"What it does:\n{summary['what_it_does']}\n\n"
        f"Who is affected:\n{', '.join(summary['who_is_affected']) or 'neclar'}\n\n"
        f"Practical impact:\n{bullets}\n\n"
        f"Impact label:\n{summary['impact_label']['impact_type']} / {summary['impact_label']['citizen_relevance_score']}/5\n\n"
        f"Important caveat:\n{summary['important_caveat']}"
    )


def build_reader_summary(analysis: LawAnalysis) -> dict[str, Any]:
    affected = ", ".join(analysis.who_is_affected[:5]) or "neclar"
    impact_items = analysis.practical_impact[:4] or analysis.what_changes[:4] or ["Impact practic neclar din textul legii."]
    if analysis.missing_information:
        caveat = analysis.missing_information[0]
    elif analysis.possible_risks_or_criticism:
        caveat = analysis.possible_risks_or_criticism[0]
    else:
        caveat = "Nu există o rezervă majoră identificată strict din textul legii."
    return {
        "schema_version": 1,
        "law_id": analysis.law_id,
        "title": analysis.plain_language_title,
        "what_it_does": analysis.one_sentence_summary,
        "who_is_affected": analysis.who_is_affected[:5] or [affected],
        "practical_impact": impact_items,
        "impact_label": {
            "impact_type": analysis.impact_type,
            "citizen_relevance_score": analysis.citizen_relevance_score,
        },
        "important_caveat": caveat,
    }


STAGE_A_SYSTEM_PROMPT = (
    "You extract facts from Romanian legislative text. "
    "You MUST always respond with a single valid JSON object and nothing else — no prose, no apologies, no markdown. "
    "Be concise: each list item should be one short sentence; omit duplicates and obvious fillers. "
    "Do not provide legal advice. Do not judge whether the law is good or bad. Do not discuss parties or politicians. "
    "Ground every conclusion in the law text. Prefer null, [] or \"unclear\" when the text is insufficient. "
    "Keep source_fragments short (≤15 words each, at most 5 fragments)."
)

STAGE_B_SYSTEM_PROMPT = (
    "You explain Romanian laws for ordinary citizens. "
    "You MUST always respond with a single valid JSON object and nothing else — no prose, no apologies, no markdown. "
    "Be concise: each list item should be one short sentence; omit duplicates and obvious fillers. "
    "Avoid legal advice, political bias, and praise/blame of parties or politicians. "
    "Use the factual extraction and the law text. "
    "Do not invent benefits, risks, budget effects, or affected groups. "
    "Prefer \"unclear\" when unsupported. Keep source_fragments short (≤15 words each, at most 5 fragments)."
)

STAGE_C_SYSTEM_PROMPT = (
    "You are a validation critic for Romanian law summaries. "
    "You MUST always respond with a single valid JSON object and nothing else — no prose, no apologies, no markdown. "
    "Even if you find nothing to correct, you must still output the complete final_analysis JSON. "
    "Be concise: each list item should be one short sentence; omit duplicates and obvious fillers. "
    "Compare the factual extraction and citizen interpretation (and the short law excerpt for context). "
    "Remove or correct unsupported claims. Prefer caveats and \"unclear\" over speculation. "
    "Avoid legal advice and political judgments. "
    "Keep source_fragments short (≤15 words each, at most 5 fragments). "
    "Output a final validated LawAnalysis."
)


def build_stage_a_prompt(law: LawInput, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    text = law.extracted_text[:max_text_chars]
    return f"""
Return exactly this JSON object shape. Keep every string value concise (one short sentence). Lists should have at most 5–8 items; omit empty or duplicate entries.
{{
  "affected_legal_acts": ["..."],
  "changed_articles": ["..."],
  "new_obligations": ["..."],
  "new_rights": ["..."],
  "penalties_costs_or_sanctions": ["..."],
  "institutions_responsible": ["..."],
  "implementation_date": "..." or null,
  "target_groups": ["..."],
  "budget_implications": "..." or null,
  "mostly_symbolic": true or false,
  "source_fragments": ["short exact snippets, ≤15 words each"],
  "missing_information": ["..."]
}}

Law metadata:
{json.dumps(asdict(law) | {"extracted_text": f"[provided below, {len(law.extracted_text)} chars]"}, ensure_ascii=False)}

Full extracted law text is the primary source of truth:
{text}
""".strip()


def build_stage_b_prompt(
    law: LawInput,
    factual: dict[str, Any],
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    text = law.extracted_text[:max_text_chars]
    return f"""
Return exactly this JSON object shape. Keep every string value concise (one short sentence). Lists should have at most 5–8 items; omit empty or duplicate entries.
{{
  "plain_language_title": "...",
  "one_sentence_summary": "...",
  "what_changes": ["..."],
  "who_is_affected": ["..."],
  "practical_impact": ["..."],
  "impact_type": "symbolic|administrative|financial|rights|restrictions|public_services|safety|justice|education|healthcare|taxation|environment|unclear",
  "impact_direction": "positive|negative|mixed|neutral|unclear",
  "citizen_relevance_score": 1,
  "public_interest_score": 1,
  "impact_magnitude_score": 1,
  "clarity_score": 1,
  "implementation_feasibility_score": 1,
  "evidence_quality_score": 1,
  "risk_of_narrow_interest_score": 1,
  "symbolic_only": true,
  "requires_budget": true or false or null,
  "creates_new_bureaucracy": true or false or null,
  "affects_many_citizens": true or false or null,
  "possible_risks_or_criticism": ["..."],
  "missing_information": ["..."],
  "source_fragments": ["short exact snippets"]
}}

Scoring guidance:
citizen_relevance_score: 1 almost no direct effect; 2 symbolic/niche; 3 visible limited category; 4 many citizens or important services; 5 broad daily-life impact.
public_interest_score: 1 mostly private/symbolic/narrow; 2 weak public-interest justification; 3 plausible; 4 clear benefit; 5 strong broad evidence-backed benefit.
impact_magnitude_score: 1 minor/symbolic; 2 small administrative; 3 moderate; 4 substantial; 5 major systemic/nationwide.
clarity_score: 1 unclear; 2 hard to understand; 3 understandable with effort; 4 clear; 5 very clear.
implementation_feasibility_score: 1 no mechanism; 2 vague; 3 feasible with dependencies; 4 clear path; 5 easy.
evidence_quality_score: 1 no visible evidence; 2 weak; 3 some reasonable justification; 4 good; 5 strong data-backed.
risk_of_narrow_interest_score: 1 low risk; 2 limited; 3 unclear/mixed; 4 significant; 5 high risk.

Law metadata:
{json.dumps(asdict(law) | {"extracted_text": f"[provided below, {len(law.extracted_text)} chars]"}, ensure_ascii=False)}

Factual extraction:
{json.dumps(factual, ensure_ascii=False)}

Law text:
{text}
""".strip()


# Stage C only needs a short law excerpt for grounding — stages A and B already
# processed the full text, and the factual/interpretation JSONs carry the substance.
STAGE_C_MAX_TEXT_CHARS = 6000


def build_stage_c_prompt(
    law: LawInput,
    factual: dict[str, Any],
    interpretation: dict[str, Any],
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    # Use a short excerpt so the full context window is available for the output JSON.
    excerpt_chars = min(max_text_chars, STAGE_C_MAX_TEXT_CHARS)
    excerpt = law.extracted_text[:excerpt_chars]
    excerpt_note = f"(first {excerpt_chars} of {len(law.extracted_text)} chars)"
    return f"""
Return exactly this JSON object shape (all fields are required). Keep every string value concise (one short sentence). Lists should have at most 5–8 items; omit empty or duplicate entries.
{{
  "unsupported_claims": ["claims removed or flagged"],
  "missing_caveats": ["important caveats"],
  "final_analysis": {{
    "law_id": "{law.law_id}",
    "original_title": "{law.title}",
    "plain_language_title": "...",
    "one_sentence_summary": "...",
    "what_changes": ["..."],
    "who_is_affected": ["..."],
    "practical_impact": ["..."],
    "impact_type": "symbolic|administrative|financial|rights|restrictions|public_services|safety|justice|education|healthcare|taxation|environment|unclear",
    "impact_direction": "positive|negative|mixed|neutral|unclear",
    "citizen_relevance_score": 1,
    "public_interest_score": 1,
    "impact_magnitude_score": 1,
    "clarity_score": 1,
    "implementation_feasibility_score": 1,
    "evidence_quality_score": 1,
    "risk_of_narrow_interest_score": 1,
    "symbolic_only": true,
    "requires_budget": true,
    "creates_new_bureaucracy": true,
    "affects_many_citizens": true,
    "affected_legal_acts": ["..."],
    "institutions_responsible": ["..."],
    "obligations_created": ["..."],
    "rights_created": ["..."],
    "penalties_or_costs": ["..."],
    "implementation_date": null,
    "possible_risks_or_criticism": ["..."],
    "missing_information": ["..."],
    "confidence_score": 1,
    "source_fragments": ["short exact snippets"]
  }}
}}

IMPORTANT: requires_budget, creates_new_bureaucracy, affects_many_citizens must be true, false, or null — never omitted.
All score fields must be integers 1-5.

Law metadata:
{json.dumps(asdict(law) | {"extracted_text": f"[excerpt provided below, {excerpt_note}]"}, ensure_ascii=False)}

Factual extraction (from stage A):
{json.dumps(factual, ensure_ascii=False)}

Citizen interpretation (from stage B):
{json.dumps(interpretation, ensure_ascii=False)}

Law text excerpt {excerpt_note}:
{excerpt}
""".strip()


LLMCaller = Callable[[str, str, str], str]


def call_with_json_retry(
    *,
    stage_name: str,
    system_prompt: str,
    user_prompt: str,
    caller: LLMCaller,
    validator: Callable[[dict[str, Any]], Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    repair_note = ""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        prompt = user_prompt if not repair_note else f"{user_prompt}\n\n{repair_note}"
        try:
            payload = parse_json_object(caller(stage_name, system_prompt, prompt))
            return validator(payload)
        except Exception as exc:
            last_error = exc
            repair_note = (
                "SCHEMA REPAIR REQUIRED. Return strict JSON only. "
                f"Stage {stage_name} failed validation with: {exc}"
            )
            if attempt < max_retries:
                time.sleep(0.5)
    assert last_error is not None
    raise last_error


def validate_critic(payload: dict[str, Any]) -> LawAnalysis:
    """Validate stage C (critic) JSON and merge unsupported_claims into missing_information."""
    if "final_analysis" not in payload or not isinstance(payload["final_analysis"], dict):
        raise ValueError("critic.final_analysis is missing")
    analysis = validate_law_analysis(payload["final_analysis"])
    unsupported = _string_list(payload.get("unsupported_claims", []), "unsupported_claims")
    missing_caveats = _string_list(payload.get("missing_caveats", []), "missing_caveats")
    if unsupported:
        analysis.missing_information.extend(
            f"Claim nesusținut eliminat/semnalat: {item}" for item in unsupported
        )
    for caveat in missing_caveats:
        if caveat not in analysis.missing_information:
            analysis.missing_information.append(caveat)
    return analysis


def analyze_law(
    law: LawInput,
    caller: LLMCaller,
    *,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> LawAnalysis:
    """Run all three stages in sequence (no DB persistence). Used in tests."""
    factual = call_with_json_retry(
        stage_name="stage_a_factual_extraction",
        system_prompt=STAGE_A_SYSTEM_PROMPT,
        user_prompt=build_stage_a_prompt(law, max_text_chars=max_text_chars),
        caller=caller,
        validator=validate_factual_extraction,
        max_retries=max_retries,
    )
    interpretation = call_with_json_retry(
        stage_name="stage_b_citizen_interpretation",
        system_prompt=STAGE_B_SYSTEM_PROMPT,
        user_prompt=build_stage_b_prompt(law, factual, max_text_chars=max_text_chars),
        caller=caller,
        validator=validate_interpretation,
        max_retries=max_retries,
    )
    return call_with_json_retry(
        stage_name="stage_c_validation_critic",
        system_prompt=STAGE_C_SYSTEM_PROMPT,
        user_prompt=build_stage_c_prompt(law, factual, interpretation, max_text_chars=max_text_chars),
        caller=caller,
        validator=validate_critic,
        max_retries=max_retries,
    )


def _build_client(provider: str, model: str):
    import openai as openai_module

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")
        service_tier = resolve_openai_service_tier(provider)
        print(f"Provider: OpenAI  |  Model: {model}  |  Service tier: {service_tier}")
        client = openai_module.OpenAI(api_key=api_key)
        client._provider = "openai"
        client._model = model
        client._openai_service_tier = service_tier
        return client

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3)
        except urllib.error.URLError:
            raise SystemExit(f"Cannot reach Ollama at {host}. Start it with: ollama serve")
        print(f"Provider: Ollama ({host})  |  Model: {model}")
        client = openai_module.OpenAI(api_key="ollama", base_url=f"{host}/v1")
        client._provider = "ollama"
        client._model = model
        client._ollama_num_ctx = infer_ollama_num_ctx(model)
        return client

    raise SystemExit(f"Unknown provider: {provider!r}")


# Stage C produces a full nested LawAnalysis JSON (20+ fields inside final_analysis)
# and needs more output budget than stages A/B.  We set a generous ceiling so the
# model never gets truncated mid-JSON, but the prompts instruct it to be concise so
# it does not pad the response unnecessarily.
_STAGE_MAX_TOKENS: dict[str, int] = {
    "stage_a_factual_extraction": 8192,
    "stage_b_citizen_interpretation": 8192,
    "stage_c_validation_critic": 16384,
}
_DEFAULT_MAX_TOKENS = 8192


def make_llm_caller(client: Any, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> LLMCaller:
    def caller(stage_name: str, system_prompt: str, user_prompt: str) -> str:
        stage_max_tokens = _STAGE_MAX_TOKENS.get(stage_name, max_tokens)
        extra_kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}}
        if getattr(client, "_provider", "") == "ollama":
            extra_kwargs["extra_body"] = {
                "num_ctx": getattr(client, "_ollama_num_ctx", infer_ollama_num_ctx(client._model))
            }
        response = create_chat_completion(
            client,
            model=client._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=stage_max_tokens,
            timeout=300,
            **extra_kwargs,
        )
        content = response.choices[0].message.content or ""
        print(f"  {stage_name}: {content[:180].replace(chr(10), ' ')}")
        return content

    return caller


def _law_input_from_row(row: sqlite3.Row) -> LawInput:
    text_payload = json.loads(row["adopted_law_text_json"] or "{}")
    extracted_text = str(text_payload.get("full_text") or text_payload.get("plain_text_excerpt") or "")
    if not extracted_text.strip():
        raise ValueError("adopted_law_text_json does not contain full_text")
    return LawInput(
        law_id=str(row["law_id"]),
        title=str(row["title"]),
        status=str(row["law_status"]) if "law_status" in row.keys() and row["law_status"] else None,
        source_url=str(row["source_url"]),
        extracted_text=extracted_text,
        initiators=[],
        adoption_date=None,
        chamber=None,
        document_type=str(row["adopted_law_identifier"] or ""),
    )


def _iter_laws_to_analyze(conn: sqlite3.Connection, *, force: bool, limit: int | None) -> list[sqlite3.Row]:
    where = [
        "adopted_law_text_json IS NOT NULL",
        "TRIM(adopted_law_text_json) <> ''",
    ]
    if not force:
        # Include rows that have never been fully analyzed OR previously failed.
        # Rows with a valid analysis_json are skipped unless --force is used.
        where.append(
            "("
            "adopted_law_analysis_json IS NULL OR TRIM(adopted_law_analysis_json) = ''"
            " OR (adopted_law_analysis_error IS NOT NULL AND TRIM(adopted_law_analysis_error) <> '')"
            ")"
        )
    sql = f"""
        SELECT law_id, source_url, identifier, title, law_status,
               adopted_law_identifier, adopted_law_text_json,
               adopted_law_factual_json, adopted_law_interpretation_json
        FROM dep_act_laws
        WHERE {' AND '.join(where)}
        ORDER BY law_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (limit,)).fetchall()
    return conn.execute(sql).fetchall()


def _load_cached_stage(row: sqlite3.Row, col: str) -> dict[str, Any] | None:
    """Return a previously persisted stage JSON, or None if absent/invalid."""
    raw = row[col] if col in row.keys() else None
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def analyze_adopted_laws_in_db(
    *,
    db_path: Path,
    provider: str,
    model: str,
    force: bool,
    limit: int | None,
    max_text_chars: int,
) -> int:
    init_db(db_path)
    client = _build_client(provider, model)
    caller = make_llm_caller(client)
    source = f"{provider}:{model}:law_analysis_v1"
    analyzed = 0
    failed = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = _iter_laws_to_analyze(conn, force=force, limit=limit)
        print(f"Found {len(rows)} adopted laws with extracted text to analyze.")
        for row in rows:
            law_id = str(row["law_id"])
            try:
                law = _law_input_from_row(row)

                # ── Stage A ───────────────────────────────────────────────
                factual = None if force else _load_cached_stage(row, "adopted_law_factual_json")
                if factual is None:
                    factual = call_with_json_retry(
                        stage_name="stage_a_factual_extraction",
                        system_prompt=STAGE_A_SYSTEM_PROMPT,
                        user_prompt=build_stage_a_prompt(law, max_text_chars=max_text_chars),
                        caller=caller,
                        validator=validate_factual_extraction,
                    )
                    conn.execute(
                        "UPDATE dep_act_laws SET adopted_law_factual_json = ?, updated_at = CURRENT_TIMESTAMP WHERE law_id = ?",
                        (json.dumps(factual, ensure_ascii=False), law_id),
                    )
                    conn.commit()
                else:
                    print(f"  stage_a_factual_extraction: (cached)")

                # ── Stage B ───────────────────────────────────────────────
                interpretation = None if force else _load_cached_stage(row, "adopted_law_interpretation_json")
                if interpretation is None:
                    interpretation = call_with_json_retry(
                        stage_name="stage_b_citizen_interpretation",
                        system_prompt=STAGE_B_SYSTEM_PROMPT,
                        user_prompt=build_stage_b_prompt(law, factual, max_text_chars=max_text_chars),
                        caller=caller,
                        validator=validate_interpretation,
                    )
                    conn.execute(
                        "UPDATE dep_act_laws SET adopted_law_interpretation_json = ?, updated_at = CURRENT_TIMESTAMP WHERE law_id = ?",
                        (json.dumps(interpretation, ensure_ascii=False), law_id),
                    )
                    conn.commit()
                else:
                    print(f"  stage_b_citizen_interpretation: (cached)")

                # ── Stage C ───────────────────────────────────────────────
                analysis = call_with_json_retry(
                    stage_name="stage_c_validation_critic",
                    system_prompt=STAGE_C_SYSTEM_PROMPT,
                    user_prompt=build_stage_c_prompt(law, factual, interpretation, max_text_chars=max_text_chars),
                    caller=caller,
                    validator=validate_critic,
                )

                summary = build_reader_summary(analysis)
                conn.execute(
                    """
                    UPDATE dep_act_laws
                    SET adopted_law_analysis_json = ?,
                        adopted_law_reader_summary = ?,
                        adopted_law_analyzed_at = ?,
                        adopted_law_analysis_source = ?,
                        adopted_law_analysis_error = NULL,
                        adopted_law_factual_json = NULL,
                        adopted_law_interpretation_json = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE law_id = ?
                    """,
                    (
                        json.dumps(asdict(analysis), ensure_ascii=False, sort_keys=True),
                        json.dumps(summary, ensure_ascii=False, sort_keys=True),
                        _now_iso(),
                        source,
                        law_id,
                    ),
                )
                conn.commit()
                analyzed += 1
                print(f"  OK analyzed {law_id}")
            except Exception as exc:
                failed += 1
                conn.execute(
                    """
                    UPDATE dep_act_laws
                    SET adopted_law_analysis_error = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE law_id = ?
                    """,
                    (str(exc)[:1000], law_id),
                )
                conn.commit()
                print(f"  ERROR {law_id}: {exc}", file=sys.stderr)
    print(f"Adopted-law analysis complete: {analyzed} analyzed, {failed} failed.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze extracted adopted-law text into citizen-friendly summaries.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--provider", choices=["openai", "ollama"], default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"LLM model. Defaults to {DEFAULT_MODEL_OPENAI_LAW_ANALYSIS} for "
            f"OpenAI and {DEFAULT_MODEL_OLLAMA} for Ollama."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-analyze laws that already have adopted_law_analysis_json.")
    parser.add_argument("--limit", type=int, default=None, help="Analyze at most N laws.")
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS, help=f"Maximum extracted law text chars sent to each prompt (default: {DEFAULT_MAX_TEXT_CHARS}).")
    args = parser.parse_args()

    model = args.model or (
        DEFAULT_MODEL_OPENAI_LAW_ANALYSIS
        if args.provider == "openai"
        else DEFAULT_MODEL_OLLAMA
    )
    return analyze_adopted_laws_in_db(
        db_path=Path(args.db_path),
        provider=args.provider,
        model=model,
        force=args.force,
        limit=args.limit,
        max_text_chars=args.max_text_chars,
    )


if __name__ == "__main__":
    raise SystemExit(main())
