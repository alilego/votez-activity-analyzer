from __future__ import annotations

import json
import re
import unicodedata

from intervention_layers.rules import extract_session_chairs


LAYER_A_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Task (Layer A): extract rubric signals from ONE target speech.

You will receive ONE target speech from a single parliamentary session, plus up to 9 previous speeches for context.
Evaluate ONLY the target speech. Do NOT evaluate/classify context speeches marked with [ctx].
Context speeches are provided only to interpret references/replies/implied meaning in the target speech.

Context interpretation:
- First identify the active debate thread using the agenda, session topics, and immediate context speeches.
- Evaluate the target speech relative to that active debate thread, not in isolation.
- A speech may sound coherent on its own but still be off-thread, derail the current debate, or revive an unrelated controversy.
- For short, fragmentary, or interruption-affected speeches, use immediate context to infer whether the line is procedural, substantive, or part of a personal/partisan attack.
- If the speaker appears to be acting as chair/moderator, distinguish neutral procedural moderation from partisan commentary delivered from the chair.

Criteria to extract:
1) policy_proposal: concrete policy action/amendment/solution (including compromise/refinement/implementation proposals)
2) policy_analysis: reasoning/analysis about policy outcomes (including evidence, facts, comparisons, consequences, institutional consequences, risk analysis, or substantive questions)
3) public_interest_orientation: focus on outcomes for citizens/public good
4) partisan_rhetoric: attacks/partisan messaging without substantive argument
5) legislative_engagement: references to legislative material (article, committee report, amendment, bill ID like PL-x)
6) procedural_content: procedural/logistical content
7) argumentation_quality: support quality (reasoning/evidence/examples/logic)
8) debate_advancement: whether the speech helps the debate progress by clarifying the issue, narrowing disagreement, proposing a concrete next step, identifying a relevant legislative or institutional consequence, documenting a serious public risk, correcting a misunderstanding in a useful way, or asking a substantive question that enables a better decision
9) primary_function: one of procedural | substantive_support | substantive_opposition | partisan_attack | symbolic_political_statement | mixed

Scale:
- criteria 1-6 and 8: yes | partial | no
- argumentation_quality: strong | weak | none

Output requirements:
- reasoning must be one Romanian sentence grounded in the speech.
- evidence_quote must be a short exact verbatim quote from the target speech, chosen to be as relevant as possible for the reasoning.

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
  "debate_advancement": "yes" | "partial" | "no",
  "primary_function": "procedural" | "substantive_support" | "substantive_opposition" | "partisan_attack" | "symbolic_political_statement" | "mixed",
  "reasoning": "o propoziție în română...",
  "evidence_quote": "citat scurt exact din discurs, ales pentru a susține cât mai bine reasoning-ul"
}
"""


LAYER_B_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Task (Layer B): assign final constructiveness label + confidence + topics using rubric signals from Layer A.

You will receive ONE target speech, optional context speeches (do NOT classify them), session topics, and Layer A extracted signals.

Rules to preserve:
- Being on-topic is NOT sufficient for constructive.
- Ideology/party/policy direction must not affect label decisions.
- Topic selection must NOT influence constructiveness_label.
- Harsh but evidence-based criticism can still be constructive.
- Formal report speeches with substantive recommendations can be constructive.

Context-first interpretation:
- First identify the active debate thread using the agenda, session topics, and immediate context speeches.
- Evaluate the target speech relative to that active debate thread, not in isolation.
- A speech may sound coherent on its own but still be `non_constructive` if it derails the current debate, revives an unrelated controversy, or abuses a procedural slot for off-topic political messaging.
- For very short, fragmentary, or interruption-affected speeches, use immediate context to infer whether the line is procedural, substantive, or part of a personal/partisan attack.
- When the speaker appears to be acting as chair/moderator, procedural moderation is usually `neutral`, but partisan commentary from the chair should weigh toward `non_constructive`.

Label guidance:
- `constructive`: the speech makes a substantive contribution that helps Parliament understand, evaluate, or improve the issue under discussion. This can happen through a concrete proposal, legislative engagement, or well-supported analysis of risks, harms, tradeoffs, constitutional concerns, security concerns, or institutional consequences. Usually at least one of `policy_proposal`, `policy_analysis`, `legislative_engagement`, or `debate_advancement` is `yes`, and partisan rhetoric is not dominant.
- `neutral`: mainly procedural, logistical, or formally administrative, with little or no substantive policy contribution. Use `neutral` for vote instructions, chair logistics, speaking-order remarks, greetings, quorum/time-management remarks, or short procedural clarifications. Do NOT use `neutral` when the speech is mainly accusatory, slogan-heavy, obstructionist, or politically performative, even if it is framed as a procedural intervention.
- `non_constructive`: mainly partisan, obstructive, self-serving, or politically performative without substantive contribution or meaningful debate advancement.

Conflict resolution:
- partisan_rhetoric=yes and BOTH policy_proposal/policy_analysis=no and debate_advancement!=yes => non_constructive
- procedural_content=yes and all substantive criteria no/partial and debate_advancement!=yes => neutral
- if a speech is framed as a procedural complaint but mainly contains accusations, outrage, slogans, personal or party attacks, or obstruction without substantive policy analysis or useful clarification => non_constructive, not neutral
- if a speech is procedurally framed but redirects the floor toward a different political controversy than the active agenda item or local debate thread, prefer non_constructive
- if a speech is short or fragmentary, classify it using the immediate exchange it belongs to; do not assume neutral only because the isolated text is brief
- if the speaker is acting as chair/moderator and uses the procedural floor to insert partisan blame, mockery, or political talking points, prefer non_constructive
- a procedural intervention can still be constructive when it invokes a rule, article, report, or parliamentary mechanism in order to produce a concrete accountability or decision-making step
- a speech can be constructive even without a concrete proposal when it offers evidence-based warning, risk analysis, factual comparison, or well-supported public-interest reasoning that materially improves understanding of the issue
- do NOT require an amendment, bill text reference, or operational plan if the speech still contributes substantive diagnostic value
- in political declarations, a speech may be constructive when it presents structured and evidence-based analysis of a public problem, democratic risk, security threat, or institutional failure, even if it does not propose legislative text
- do NOT upgrade to constructive when the speech only invokes values, dangers, patriotism, democracy, freedom, or public interest in a vague or symbolic way without concrete reasoning, issue-specific substance, or identifiable consequences
- distinguish evidence-based risk analysis from alarmist rhetoric:
  - documented harms, comparisons, mechanisms, or consequences can support constructive
  - slogans, conspiracies, generalized outrage, and symbolic fear appeals without substantiation should favor non_constructive
- any substantive yes or debate_advancement=yes can support constructive, but do NOT upgrade to constructive when those signals are only partial and argumentation_quality is weak or none, unless the speech clearly advances the debate in a concrete way
- professional tone, formal language, or orderly delivery do NOT by themselves make a speech constructive
- a speech is NOT constructive merely because it contains an imperative, demand, or proposal-shaped phrase; if it lacks substantive support, evidence, legislative reasoning, realistic policy elaboration, or useful debate advancement and is mainly rhetorical or performative, prefer non_constructive
- strong substantive + strong partisan rhetoric => use argumentation_quality as a tie-breaker:
  - argumentation_quality=strong => favor constructive
  - argumentation_quality=none => favor non_constructive
  - argumentation_quality=weak => classify by dominant share and lower confidence
- political declarations or ideological speeches are non_constructive when they mainly deliver slogans, blame, symbolic positioning, or broad grievances without concrete policy reasoning or actionable substance

Short examples:
- "Este o procedură care a fost aprobată în Biroul permanent..." => non_constructive if, in context, it revives an unrelated political controversy instead of the active legislative debate
- "USR-ul a plecat de la guvernare..." => non_constructive when spoken from the chair during moderation, because it uses procedural authority for partisan attack
- "Conform art. 211, vă rog să supuneţi la vot chemarea prim-ministrului în plen." => constructive because it uses a concrete parliamentary mechanism to advance accountability
- "Eu sunt de acord că domnul X nu se confundă cu..." => use immediate context; if it is the opening of a personal attack, do not default to neutral just because the fragment is short
- "România a coborât de la 6,45 la 5,99 în Indexul Democrațiilor și a pierdut 12 locuri; asta arată degradarea instituțională și necesitatea unor alegeri libere." => constructive because it offers evidence-based democratic risk analysis, even without legislative text
- "Lipsa României din rețeaua EuroHPC și absența unei metodologii clare pentru sistemele IA cu risc ridicat creează vulnerabilități concrete pentru drepturi și competitivitate." => constructive because it identifies concrete institutional and policy risks
- "Critic proiectul pentru că transferă costuri către primării; propun amendarea art. 5 pentru a evita incapacitatea de plată." => constructive
- "Vă rog să precizați care este textul final al raportului, pentru că forma primită cu 20 de minute înainte schimbă sensul amendamentului." => constructive
- "România este în pericol, globaliștii ne distrug viitorul!" => non_constructive
- "Nu ne puteți reduce la tăcere! România va fi liberă!" => non_constructive because it is mainly symbolic rhetoric about values without concrete reasoning or issue-specific analysis
- "Stimați colegi, vă rog să respectăm programul și să continuăm ședința." => neutral

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
- evidence_quote must be a short exact verbatim quote from the target speech, chosen to be as relevant as possible for the reasoning.

Output JSON only, wrapped as {"results":[...]} with exactly one item:
{
  "speech_index": <int>,
  "constructiveness_label": "constructive" | "neutral" | "non_constructive",
  "confidence": 0.0-1.0,
  "topics": ["..."],
  "reasoning": "o propoziție în română...",
  "evidence_quote": "citat scurt exact din discurs, ales pentru a susține cât mai bine reasoning-ul"
}
"""


LAYER_C_SYSTEM_PROMPT = """You are a parliamentary debate analyst specialising in the Romanian Parliament (Camera Deputaților and Senat).

Task (Layer C QA): review Layer A + Layer B outputs for ONE target speech and confirm or minimally revise.

Do NOT classify context speeches.
Do NOT invent facts not present in target speech.
Evidence quote must be a short exact verbatim quote from the target speech, chosen to be as relevant as possible for the reasoning.
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
  "evidence_quote": "citat scurt exact din discurs, ales pentru a susține cât mai bine reasoning-ul",
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


_CONTEXT_STOPWORDS = {
    "acest", "aceasta", "aceste", "acesti", "pentru", "privind", "asupra", "care",
    "este", "sunt", "fara", "despre", "dupa", "inainte", "intre", "unde", "cand",
    "doar", "foarte", "mai", "mult", "putin", "lege", "legea", "proiect", "proiectul",
    "camerei", "deputatilor", "senatului", "plen", "sedinta", "sedintei", "politica",
    "politice", "interventii", "publice", "public", "national", "nationale", "romania",
    "romanilor", "romaniei", "domnul", "doamna", "stimați", "stimati", "colegi",
}
_IMMEDIATE_CONTEXT_WINDOW = 3


def _normalize_overlap_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _tokenize_overlap(text: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[a-z0-9]+", _normalize_overlap_text(text))
        if len(tok) >= 4 and tok not in _CONTEXT_STOPWORDS
    }


def _candidate_texts(session_topics: list, agenda: list[dict] | None) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    for t in session_topics or []:
        if isinstance(t, dict):
            label = str(t.get("label", "")).strip()
            desc = str(t.get("description", "")).strip()
            law_id = str(t.get("law_id", "")).strip()
            if label:
                detail = " ".join(part for part in (label, desc, law_id) if part)
                candidates.append(("session topic", label, detail))
        else:
            raw = str(t).strip()
            if raw:
                candidates.append(("session topic", raw, raw))
    for item in agenda or []:
        title = str(item.get("title", "")).strip()
        law_id = str(item.get("law_id", "")).strip()
        if title:
            label = f"{title} ({law_id})" if law_id else title
            detail = " ".join(part for part in (title, law_id) if part)
            candidates.append(("agenda item", label, detail))
    return candidates


def _format_active_debate_thread(
    session_topics: list,
    agenda: list[dict] | None,
    context_speeches: list[dict] | None,
    target_text: str,
) -> str:
    recent_context = context_speeches[-_IMMEDIATE_CONTEXT_WINDOW:] if context_speeches else []
    recent_text = " ".join(str(sp.get("text", "")) for sp in recent_context)
    tokens = _tokenize_overlap(f"{recent_text} {target_text}")
    if not tokens:
        return ""
    scored: list[tuple[int, str, str]] = []
    recent_norm = _normalize_overlap_text(f"{recent_text} {target_text}")
    for source, label, detail in _candidate_texts(session_topics, agenda):
        cand_tokens = _tokenize_overlap(detail)
        if not cand_tokens:
            continue
        overlap = len(tokens & cand_tokens)
        exact_bonus = 2 if _normalize_overlap_text(label) and _normalize_overlap_text(label) in recent_norm else 0
        score = overlap + exact_bonus
        if score > 0:
            scored.append((score, source, label))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    lines = [
        "## Likely active debate thread",
        "- Prioritize the issue repeated in the immediate context below, not older unrelated controversies.",
    ]
    for _, source, label in scored[:3]:
        lines.append(f"- Candidate from {source}: {label}")
    return "\n".join(lines)


def _speaker_line(
    raw_speaker: str,
    target_speaker: str,
    session_chairs: set[str],
    *,
    immediate: bool = False,
    immediately_before: bool = False,
    include_same_speaker: bool = True,
) -> str:
    tags: list[str] = []
    if session_chairs and any(name in raw_speaker for name in session_chairs):
        tags.append("chair")
    if include_same_speaker and raw_speaker.strip() == target_speaker.strip():
        tags.append("same speaker")
    if immediately_before:
        tags.append("immediately before target")
    elif immediate:
        tags.append("immediate context")
    if not tags:
        return raw_speaker
    return f"{raw_speaker} ({', '.join(tags)})"


def _format_context(
    context_speeches: list[dict] | None,
    *,
    target_speech: dict,
    session_topics: list,
    agenda: list[dict] | None,
    session_chairs: set[str],
) -> str:
    if not context_speeches:
        return ""
    parts: list[str] = []
    active_thread = _format_active_debate_thread(
        session_topics=session_topics,
        agenda=agenda,
        context_speeches=context_speeches,
        target_text=str(target_speech.get("text", "")),
    )
    if active_thread:
        parts.append(active_thread)
    parts.append(
        "## Context use guidance\n"
        "- Use the immediate context first to determine the active debate thread.\n"
        "- Decide whether the target speech follows that thread, redirects it, or derails it.\n"
        "- Short fragments should be interpreted as part of the exchange they belong to, not in isolation."
    )
    recent = context_speeches[-_IMMEDIATE_CONTEXT_WINDOW:]
    older = context_speeches[:-_IMMEDIATE_CONTEXT_WINDOW]
    target_speaker = str(target_speech.get("raw_speaker", ""))
    if recent:
        lines = [f"## Immediate context ({len(recent)} previous speeches; highest priority; do NOT classify)"]
        for idx, sp in enumerate(recent):
            lines.append(
                f"[ctx {sp['speech_index']}] Speaker: "
                f"{_speaker_line(str(sp['raw_speaker']), target_speaker, session_chairs, immediate=True, immediately_before=(idx == len(recent) - 1))}\n"
                f"{sp['text'].strip()}"
            )
        parts.append("\n\n".join(lines))
    if older:
        lines = [f"## Earlier previous context ({len(older)} speeches; lower priority; do NOT classify)"]
        for sp in older:
            lines.append(
                f"[ctx {sp['speech_index']}] Speaker: "
                f"{_speaker_line(str(sp['raw_speaker']), target_speaker, session_chairs)}\n"
                f"{sp['text'].strip()}"
            )
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts)


def _format_target_speech(target_speech: dict, session_chairs: set[str]) -> str:
    speaker = _speaker_line(
        str(target_speech["raw_speaker"]),
        str(target_speech["raw_speaker"]),
        session_chairs,
        include_same_speaker=False,
    )
    return (
        "## Speech to classify (1 target speech)\n\n"
        f"[{target_speech['speech_index']}] Speaker: {speaker}\n"
        f"{target_speech['text'].strip()}"
    )


def _format_preextracted_law_ids(law_id_index: dict[str, list[int]] | None) -> str:
    if not law_id_index:
        return ""
    lines = ["## Pre-extracted law IDs (ground truth candidates)"]
    for law_id, speech_indexes in law_id_index.items():
        indices_text = ", ".join(str(i) for i in sorted(set(speech_indexes))[:12])
        if indices_text:
            lines.append(f"- {law_id} [speech_index: {indices_text}]")
        else:
            lines.append(f"- {law_id}")
    return "\n".join(lines)


def _format_agenda(agenda: list[dict] | None) -> str:
    if not agenda:
        return ""
    lines = ["## Legislative agenda (pre-extracted from session)"]
    for item in agenda:
        entry = ""
        item_num = item.get("item_number")
        if item_num is not None:
            entry += f"{item_num}. "
        title = str(item.get("title", "")).strip()
        if title:
            entry += title
        law_id = str(item.get("law_id") or "").strip()
        if law_id:
            entry += f" ({law_id})"
        if entry.strip():
            lines.append(f"- {entry.strip()}")
    return "\n".join(lines) if len(lines) > 1 else ""


_INTERRUPTION_HINT = (
    "## Interruption context\n"
    "This speaker was interrupted by the session chair for not following "
    "parliamentary procedure. Consider whether the speech content, despite "
    "the procedural violation, contains substantive policy proposals or "
    "analysis that benefit citizens. A procedural violation alone does not "
    "automatically make a speech non-constructive if the content is "
    "substantively valuable, but purely disruptive behaviour without "
    "substantive content should weigh toward non-constructive."
)


def build_layer_a_user_message(
    session: dict,
    session_topics: list,
    target_speech: dict,
    context_speeches: list[dict] | None = None,
    law_id_index: dict[str, list[int]] | None = None,
    agenda: list[dict] | None = None,
    interruption_context: str | None = None,
) -> str:
    parts: list[str] = [f"## Session\nDate: {session.get('session_date', '')}"]
    notes = str(session.get("initial_notes") or "").strip()
    session_chairs = extract_session_chairs(notes)
    if notes:
        parts[0] += f"\nInitial notes: {notes[:300]}"
    topics = _format_session_topics(session_topics)
    if topics:
        parts.append(f"## Session topics (grounding context)\n{topics}")
    agenda_text = _format_agenda(agenda)
    if agenda_text:
        parts.append(agenda_text)
    laws = _format_preextracted_law_ids(law_id_index)
    if laws:
        parts.append(laws)
    if interruption_context == "procedure_violation":
        parts.append(_INTERRUPTION_HINT)
    ctx = _format_context(
        context_speeches,
        target_speech=target_speech,
        session_topics=session_topics,
        agenda=agenda,
        session_chairs=session_chairs,
    )
    if ctx:
        parts.append(ctx)
    parts.append(_format_target_speech(target_speech, session_chairs))
    parts.append("Return Layer A rubric extraction only for the target speech.")
    return "\n\n".join(parts)


def build_layer_b_user_message(
    session: dict,
    session_topics: list,
    target_speech: dict,
    layer_a_output: dict,
    context_speeches: list[dict] | None = None,
    law_id_index: dict[str, list[int]] | None = None,
    agenda: list[dict] | None = None,
    interruption_context: str | None = None,
) -> str:
    parts: list[str] = [f"## Session\nDate: {session.get('session_date', '')}"]
    notes = str(session.get("initial_notes") or "").strip()
    session_chairs = extract_session_chairs(notes)
    if notes:
        parts[0] += f"\nInitial notes: {notes[:300]}"
    topics = _format_session_topics(session_topics)
    if topics:
        parts.append(f"## Session topics (grounding context)\n{topics}")
    agenda_text = _format_agenda(agenda)
    if agenda_text:
        parts.append(agenda_text)
    laws = _format_preextracted_law_ids(law_id_index)
    if laws:
        parts.append(laws)
    if interruption_context == "procedure_violation":
        parts.append(_INTERRUPTION_HINT)
    ctx = _format_context(
        context_speeches,
        target_speech=target_speech,
        session_topics=session_topics,
        agenda=agenda,
        session_chairs=session_chairs,
    )
    if ctx:
        parts.append(ctx)
    parts.append(_format_target_speech(target_speech, session_chairs))
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
    law_id_index: dict[str, list[int]] | None = None,
    agenda: list[dict] | None = None,
    interruption_context: str | None = None,
) -> str:
    parts: list[str] = [f"## Session\nDate: {session.get('session_date', '')}"]
    notes = str(session.get("initial_notes") or "").strip()
    session_chairs = extract_session_chairs(notes)
    if notes:
        parts[0] += f"\nInitial notes: {notes[:300]}"
    topics = _format_session_topics(session_topics)
    if topics:
        parts.append(f"## Session topics (grounding context)\n{topics}")
    agenda_text = _format_agenda(agenda)
    if agenda_text:
        parts.append(agenda_text)
    laws = _format_preextracted_law_ids(law_id_index)
    if laws:
        parts.append(laws)
    if interruption_context == "procedure_violation":
        parts.append(_INTERRUPTION_HINT)
    ctx = _format_context(
        context_speeches,
        target_speech=target_speech,
        session_topics=session_topics,
        agenda=agenda,
        session_chairs=session_chairs,
    )
    if ctx:
        parts.append(ctx)
    parts.append(_format_target_speech(target_speech, session_chairs))
    parts.append("## Layer A extracted signals\n" + json.dumps(layer_a_output, ensure_ascii=False, indent=2))
    parts.append("## Layer B decision\n" + json.dumps(layer_b_output, ensure_ascii=False, indent=2))
    parts.append("## QA triggers\n" + json.dumps(qa_reasons, ensure_ascii=False, indent=2))
    parts.append("Review and confirm/revise minimally.")
    return "\n\n".join(parts)
