# classification-rubric.md
`votez-activity-analyzer`

This document defines the operational rubric used to classify parliamentary interventions.

The classification must be:

- Grounded in retrieved session context (RAG)
- Applied consistently across sessions
- Structured via MCP tool calls
- Explainable and auditable

---

# 1) Classification Labels

Each intervention must be classified into one of the following:

- `relevant`
- `neutral`
- `non_relevant`

---

# 2) Definition of "Relevant"

An intervention is **relevant** if:

- It directly addresses the main topic(s) of the session or agenda item.
- It contributes arguments, explanations, objections, or clarifications related to the debated subject.
- It proposes amendments, policy ideas, or positions tied to the session’s legislative focus.
- It responds substantively to another intervention about the same debated topic.

Examples:

- Arguing for or against a bill under discussion.
- Explaining economic impact of a proposed law.
- Responding to a criticism related to the legislative topic.

---

# 3) Definition of "Neutral"

An intervention is **neutral** if:

- It is procedural (e.g., voting instructions, agenda management).
- It concerns technical session logistics (microphones, breaks, quorum).
- It includes formal announcements (e.g., resignations, oath-taking).
- It is a short interjection that does not meaningfully engage with the session topic.
- It includes respectful but non-substantive remarks.

Examples:

- "Vă rog să luați loc."
- Voting instructions or test votes.
- Announcing resignation from a parliamentary group.
- Formal reading of administrative text.

---

# 4) Definition of "Non-Relevant"

An intervention is **non_relevant** if:

- It introduces unrelated political accusations or unrelated topics.
- It diverts discussion away from the session's legislative subject.
- It consists primarily of rhetorical attacks unrelated to the agenda item.
- It intentionally shifts focus to an external issue not under debate.

Examples:

- Attacking a political figure unrelated to the bill under discussion.
- Discussing a different law not on the agenda.
- Introducing conspiracy claims unrelated to the current debate.

---

# 5) Edge Case Handling

## A) Mixed Content

If an intervention contains both relevant and irrelevant parts:

- Classify as `relevant` if the dominant portion engages with the debated topic.
- Classify as `non_relevant` if the majority of the speech is diversionary.

## B) Procedural but Politically Charged

If procedural language includes politically charged commentary:

- If commentary meaningfully connects to the debated subject → `relevant`
- If commentary is unrelated to session topic → `non_relevant`
- If primarily procedural → `neutral`

## C) Very Short Interventions

If extremely short and lacking semantic substance:

- Default to `neutral` unless clearly diversionary.

---

# 6) Evidence Requirement (RAG Grounding)

For every classification:

The system must store:

- `relevance_label`
- `confidence` (0.0 – 1.0)
- `evidence_chunk_ids[]` (retrieved chunks supporting the decision)

The model must not classify based on general knowledge.
It must justify classification using retrieved session context.

---

# 7) Topic Extraction Rules

Topics should:

- Be concise (1–4 words preferred).
- Reflect policy areas, legislative themes, or recurring debate subjects.
- Avoid generic terms such as "politics" or "discussion".
- Prefer normalized phrasing (e.g., "tax reform", not "changing taxes").

If topic cannot be clearly identified:

- Return empty topic list.
- Do not hallucinate.

---

# 8) Confidence Guidelines

Confidence should be:

- High (0.8–1.0) if session topic clearly matches intervention content.
- Medium (0.5–0.79) if partial match or mixed signals.
- Low (0.0–0.49) if weak contextual alignment.

Low-confidence cases may later be flagged for review.

---

# 9) Non-Goals

The classifier is NOT:

- A sentiment analyzer.
- A truth verifier.
- A fact-checking system.
- A political bias detector.

It strictly evaluates contextual relevance to session topic.

---

End of classification rubric.
