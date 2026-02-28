# classification-rubric.md
`votez-activity-analyzer`

This document defines the operational rubric used to classify parliamentary interventions.

The classification must be:

- Grounded in retrieved session context (RAG)
- Applied consistently across sessions
- Structured via MCP tool calls
- Explainable and auditable

## See also

- `architecture.md`
- `rag-indexing.md`
- `mcp-tools.md`
- `output-contract.md`

---

# 1) Classification Labels

Each intervention must be classified into one of the following:

- `constructive`
- `neutral`
- `non_constructive`

---

# 2) Definition of "Constructive"

An intervention is **constructive** if:

- It directly engages with the session topic AND genuinely tries to advance the public good.
- It proposes or supports concrete solutions, amendments, or improvements to legislation.
- It contributes substantive arguments, analysis, or evidence aimed at better outcomes for citizens.
- It responds meaningfully to another speaker in a way that advances the debate.
- The primary intent is to find common ground, clarify facts, or improve policy — not to score political points.

Examples:

- Proposing a specific amendment that improves the bill under discussion.
- Explaining the practical impact of a law on citizens with concrete data.
- Raising a well-founded concern about a bill's implementation and suggesting a fix.
- Responding to a criticism with substantive counter-argument and new information.

---

# 3) Definition of "Neutral"

An intervention is **neutral** if:

- It is procedural (e.g., voting instructions, agenda management, quorum checks).
- It concerns technical session logistics (microphones, breaks, seating).
- It includes formal announcements (e.g., resignations, oath-taking, reading of official communications, greetings).
- It is a short interjection that does not meaningfully engage in either direction.
- It includes respectful but non-substantive remarks.

Examples:

- "Vă rog să luați loc."
- Voting instructions or test votes.
- Announcing resignation from a parliamentary group.
- Formal reading of administrative text.

---

# 4) Definition of "Non-Constructive"

An intervention is **non_constructive** if:

- It is primarily aimed at serving narrow interests: own party, own political career, a sponsor, or a specific entity — rather than citizens broadly.
- It intentionally blocks or derails the debate without offering an alternative.
- It consists primarily of rhetorical attacks, accusations, or insults — whether or not the target is related to the session topic.
- It introduces unrelated political narratives or conspiracy claims.
- It is filibustering: speaking at length to waste time rather than contribute substance.
- It is on-topic but purely partisan positioning with no intent to find solutions.

Note: An intervention can be topically on-point yet still `non_constructive`. Being on-topic is not sufficient for `constructive`.

Examples:

- Attacking a political opponent by name, with no substantive connection to the bill.
- Making a speech that repeats talking points without engaging with the debate.
- Raising a procedural objection purely to delay a vote, with no legal or substantive basis.
- Defending a policy change that primarily benefits a sponsor or party donor, with no public interest framing.
- Filibustering through repetition or reading unrelated text.

---

# 5) Edge Case Handling

## A) On-Topic but Self-Serving

If an intervention addresses the session topic but its primary purpose is partisan positioning or narrow interest:

- Classify as `non_constructive`.
- The topic match alone does not elevate an intervention to `constructive`.

## B) Mixed Content

If an intervention contains both constructive and non-constructive parts:

- Classify as `constructive` if the dominant portion genuinely advances the public interest.
- Classify as `non_constructive` if the majority is partisan, self-serving, or blocking.

## C) Procedural but Politically Charged

If procedural language includes politically charged commentary:

- If the commentary meaningfully advances the debate → `constructive`
- If the commentary is partisan or diversionary → `non_constructive`
- If primarily procedural with no substantive impact → `neutral`

## D) Very Short Interventions

If extremely short and lacking semantic substance:

- Default to `neutral` unless clearly adversarial or self-serving.

## E) Legitimate Opposition

Opposing a bill is not inherently `non_constructive`. Opposition is `constructive` when:

- It raises substantive concerns grounded in evidence or legal analysis.
- It proposes alternatives or amendments.

Opposition is `non_constructive` when it is purely rhetorical, partisan, or blocking without substance.

---

# 6) Evidence Requirement (RAG Grounding)

For every classification:

The system must store:

- `constructiveness_label`
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

- High (0.8–1.0) if session context clearly supports the classification decision.
- Medium (0.5–0.79) if partial match or mixed signals.
- Low (0.0–0.49) if weak contextual alignment.

Low-confidence cases may later be flagged for review.

---

# 9) Non-Goals

The classifier is NOT:

- A sentiment analyzer.
- A truth verifier.
- A fact-checking system.
- A political bias detector between parties.

It evaluates whether the intervention serves the public good and advances the debate, grounded in the session context.

---

End of classification rubric.
