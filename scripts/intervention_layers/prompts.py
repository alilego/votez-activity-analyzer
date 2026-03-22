from __future__ import annotations

import json


LAYER_A_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Task (Layer A): extract rubric signals from ONE target speech.

You will receive ONE target speech from a single parliamentary session, plus up to 9 previous speeches for context.
Evaluate ONLY the target speech. Do NOT evaluate/classify context speeches marked with [ctx].
Context speeches are provided only to interpret references/replies/implied meaning in the target speech.
Never transfer claims, tone, or accusations from [ctx] speeches into the target-speech reasoning.
If the target speech is short/procedural, keep reasoning strictly about that line only.

Being on-topic is NOT sufficient for constructive analysis.

Early filter (apply first):
If the target speech is clearly procedural (speaking order, vote instructions/announcements, greetings, chair logistics without policy substance),
mark it as primarily procedural and continue with rubric fields accordingly.
For very short floor interjections (e.g., "Nu.", "Da.", "Mulțumesc.") without explicit insult/accusation or policy content:
- set procedural_content = yes
- set partisan_rhetoric = no (or partial only if explicit attack words are present)

Criteria to extract:
1) policy_proposal: concrete policy action/amendment/solution (including compromise/refinement/implementation proposals)
2) policy_analysis: reasoning/analysis about policy outcomes (including evidence/facts/consequences/substantive questions)
3) public_interest_orientation: focus on outcomes for citizens/public good
4) partisan_rhetoric: attacks/partisan messaging without substantive argument
5) legislative_engagement: references to legislative material (article, committee report, amendment, bill ID like PL-x)
6) procedural_content: procedural/logistical content
7) argumentation_quality: support quality (reasoning/evidence/examples/logic)
8) primary_function: one of procedural | substantive_support | substantive_opposition | partisan_attack | symbolic_political_statement | mixed

Scale:
- criteria 1-6: yes | partial | no
- argumentation_quality: strong | weak | none

Output requirements:
- reasoning must be one Romanian sentence grounded in the speech.
- evidence_quote must be short and verbatim from the target speech.
- evidence_quote MUST be from the target speech, never from [ctx] speeches.

Output JSON only (no prose), wrapped as {"results":[...]} with exactly one item:
{
  "speech_index": <int>,
  "policy_proposal": "yes" | "partial" | "no",
  "policy_analysis": "yes" | "partial" | "no",
  "public_interest_orientation": "yes" | "partial" | "no",
  "partisan_rhetoric": "yes" | "partial" | "no",
  "legislative_engagement": "yes" | "partial" | "no",
  "procedural_content": "yes" | "partial" | "no",
  "argumentation_quality": "strong" | "weak" | "none",
  "primary_function": "procedural" | "substantive_support" | "substantive_opposition" | "partisan_attack" | "symbolic_political_statement" | "mixed",
  "reasoning": "o propoziție în română...",
  "evidence_quote": "citat verbatim din discurs"
}
"""


LAYER_B_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Task (Layer B): assign final constructiveness label + confidence + topics using rubric signals from Layer A.

You will receive ONE target speech, optional context speeches (do NOT classify them), session topics, and Layer A extracted signals.

Rules to preserve:
- Being on-topic is NOT sufficient for constructive.
- Ideology/party/policy direction must not affect label decisions.
- Harsh but evidence-based criticism can still be constructive.
- Formal report speeches with substantive recommendations can be constructive.
- Topic selection must NOT influence constructiveness_label.

Decision guidance:
- constructive if:
  - policy_proposal = yes OR
  - policy_analysis = yes OR
  - legislative_engagement = yes
  AND
  - partisan_rhetoric != yes
- neutral if:
  - procedural_content = yes
  AND
  - all other criteria are no or partial
- non_constructive if:
  - partisan_rhetoric = yes
  AND
  - policy_proposal = no
  AND
  - policy_analysis = no

Conflict resolution:
- partisan_rhetoric=yes and BOTH policy_proposal/policy_analysis=no => non_constructive
- procedural_content=yes and all substantive criteria no/partial => neutral
- any substantive yes and partisan_rhetoric!=yes => constructive
- strong substantive + strong partisan rhetoric => classify by dominant share; lower confidence

Confidence guidance:
- clear single-rule: 0.80-0.95
- mixed but one-sided: 0.65-0.79
- balanced/ambiguous mixed: 0.50-0.64
- highly uncertain/insufficient evidence/unresolved conflict: 0.30-0.49

Topics:
- choose up to 3 topics ONLY from provided session topics
- if none apply return []
- do not invent new topics
- may infer topic from context if target clearly responds to it

Output requirements:
- reasoning must be one Romanian sentence grounded in speech + Layer A.
- evidence_quote must be short and verbatim from target speech.

Output JSON only, wrapped as {"results":[...]} with exactly one item:
{
  "speech_index": <int>,
  "constructiveness_label": "constructive" | "neutral" | "non_constructive",
  "confidence": 0.0-1.0,
  "topics": ["..."],
  "reasoning": "o propoziție în română...",
  "evidence_quote": "citat verbatim din discurs"
}
"""


LAYER_C_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Task (Layer C QA): review Layer A + Layer B outputs for ONE target speech and confirm or minimally revise.

Do NOT classify context speeches.
Do NOT invent facts not present in target speech.
Evidence quote must be verbatim from the target speech.
Reasoning must be one Romanian sentence.

QA objective:
- confirm Layer B when coherent
- revise only what is necessary (label/topics/confidence)
- keep topics constrained to provided session topics

Output JSON only, wrapped as {"results":[...]} with exactly one item:
{
  "speech_index": <int>,
  "final_label": "constructive" | "neutral" | "non_constructive",
  "final_confidence": 0.0-1.0,
  "topics": ["..."],
  "reasoning": "o propoziție în română despre confirmare sau corecție",
  "evidence_quote": "citat verbatim din discurs",
  "qa_action": "confirmed" | "revised_label" | "revised_topics" | "revised_confidence"
}
"""


def _format_session_topics(session_topics: list) -> str:
    if not session_topics:
        return ""
    lines: list[str] = []
    for t in session_topics:
        if isinstance(t, dict):
            label = str(t.get("label", "")).strip()
            desc = str(t.get("description", "")).strip()
            law_id = str(t.get("law_id", "")).strip()
            if not label:
                continue
            item = f"- {label}"
            if law_id:
                item += f" ({law_id})"
            if desc:
                item += f": {desc}"
            lines.append(item)
        else:
            raw = str(t).strip()
            if raw:
                lines.append(f"- {raw}")
    return "\n".join(lines)


def _format_context(context_speeches: list[dict] | None) -> str:
    if not context_speeches:
        return ""
    lines = [f"## Previous speeches for context ({len(context_speeches)}; do NOT classify)"]
    for sp in context_speeches:
        lines.append(f"[ctx {sp['speech_index']}] Speaker: {sp['raw_speaker']}\n{sp['text'].strip()}")
    return "\n\n".join(lines)


def _format_target_speech(target_speech: dict) -> str:
    return (
        "## Speech to classify (1 target speech)\n\n"
        f"[{target_speech['speech_index']}] Speaker: {target_speech['raw_speaker']}\n"
        f"{target_speech['text'].strip()}"
    )


def build_layer_a_user_message(
    session: dict,
    session_topics: list,
    target_speech: dict,
    context_speeches: list[dict] | None = None,
    law_index_text: str = "",
) -> str:
    parts: list[str] = [f"## Session\nDate: {session.get('session_date', '')}"]
    notes = str(session.get("initial_notes") or "").strip()
    if notes:
        parts[0] += f"\nInitial notes: {notes[:300]}"
    topics = _format_session_topics(session_topics)
    if topics:
        parts.append(f"## Session topics (grounding context)\n{topics}")
    if law_index_text:
        parts.append(law_index_text)
    ctx = _format_context(context_speeches)
    if ctx:
        parts.append(ctx)
    parts.append(_format_target_speech(target_speech))
    parts.append("Return Layer A rubric extraction only for the target speech.")
    return "\n\n".join(parts)


def build_layer_b_user_message(
    session: dict,
    session_topics: list,
    target_speech: dict,
    layer_a_output: dict,
    context_speeches: list[dict] | None = None,
    law_index_text: str = "",
) -> str:
    parts: list[str] = [f"## Session\nDate: {session.get('session_date', '')}"]
    topics = _format_session_topics(session_topics)
    if topics:
        parts.append(f"## Session topics (grounding context)\n{topics}")
    if law_index_text:
        parts.append(law_index_text)
    ctx = _format_context(context_speeches)
    if ctx:
        parts.append(ctx)
    parts.append(_format_target_speech(target_speech))
    parts.append("## Layer A extracted signals\n" + json.dumps(layer_a_output, ensure_ascii=False, indent=2))
    parts.append("Assign final label, confidence, and topics for the target speech.")
    return "\n\n".join(parts)


def build_layer_c_user_message(
    session: dict,
    session_topics: list,
    target_speech: dict,
    layer_a_output: dict,
    layer_b_output: dict,
    qa_reasons: list[str],
    context_speeches: list[dict] | None = None,
    law_index_text: str = "",
) -> str:
    parts: list[str] = [f"## Session\nDate: {session.get('session_date', '')}"]
    topics = _format_session_topics(session_topics)
    if topics:
        parts.append(f"## Session topics (grounding context)\n{topics}")
    if law_index_text:
        parts.append(law_index_text)
    ctx = _format_context(context_speeches)
    if ctx:
        parts.append(ctx)
    parts.append(_format_target_speech(target_speech))
    parts.append("## Layer A extracted signals\n" + json.dumps(layer_a_output, ensure_ascii=False, indent=2))
    parts.append("## Layer B decision\n" + json.dumps(layer_b_output, ensure_ascii=False, indent=2))
    parts.append("## QA triggers\n" + json.dumps(qa_reasons, ensure_ascii=False, indent=2))
    parts.append("Review and confirm/revise minimally.")
    return "\n\n".join(parts)
