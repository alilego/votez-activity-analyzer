# output-contract.md

## Purpose

Define the structured output produced by votez-activity-analyzer.

The frontend will consume this data to display:

- Per-member intervention statistics
- Lists of interventions categorized by relevance
- Top 20 debate topics per member

This contract is versioned and may evolve.

---

# Version

Output contract version: v0

---

# Output Format

- JSON files
- UTF-8 encoding
- Deterministic structure
- No embedded HTML

---

# File Structure (v0)

outputs/
  members/
    index.json
    {memberId}.json
  parties/
    index.json
    {partyId}.json

---

# 1. Members Index

File: `outputs/members/index.json`

Purpose:
- Lightweight list for frontend listing views

Structure:

[
  {
    "member_id": "string",
    "name": "string",
    "party": "string | null",
    "interventions_total": number,
    "relevant_count": number,
    "neutral_count": number,
    "non_relevant_count": number,
    "top_topics": [
      {
        "topic": "string",
        "count": number
      }
    ]
  }
]

Notes:
- `top_topics` contains at most 20 items
- Sorted descending by count

---

# 2. Member Detail

File: `outputs/members/{memberId}.json`

Purpose:
- Detailed view for a specific member

Structure:

{
  "member_id": "string",
  "name": "string",
  "party": "string | null",

  "stats": {
    "total": number,
    "relevant": number,
    "neutral": number,
    "non_relevant": number
  },

  "top_topics": [
    {
      "topic": "string",
      "count": number
    }
  ],

  "interventions": {
    "relevant": [
      {
        "session_id": "string",
        "session_date": "YYYY-MM-DD",
        "excerpt": "string",
        "topics": ["string"],
        "confidence": number,
        "stenogram_name": "string",
        "stenogram_link": "string"
      }
    ],
    "neutral": [
        {
        "session_id": "string",
        "session_date": "YYYY-MM-DD",
        "excerpt": "string",
        "topics": ["string"],
        "confidence": number,
        "stenogram_name": "string",
        "stenogram_link": "string"
      }
    ],
    "non_relevant": [
        {
        "session_id": "string",
        "session_date": "YYYY-MM-DD",
        "excerpt": "string",
        "topics": ["string"],
        "confidence": number,
        "stenogram_name": "string",
        "stenogram_link": "string"
      }
    ]
  }
}

---

# 3. Parties Index

File: `outputs/parties/index.json`

Purpose:
- Lightweight list for frontend party-level views
- Provides totals aggregated across members of the party

Structure:

[
  {
    "party_id": "string",
    "party_name": "string",

    "members_count": number,

    "interventions_total": number,
    "relevant_count": number,
    "neutral_count": number,
    "non_relevant_count": number,

    "top_topics": [
      {
        "topic": "string",
        "count": number
      }
    ]
  }
]

Notes:
- `top_topics` contains at most 20 items
- Sorted descending by count
- Counts represent the sum of interventions for members currently attributed to this party in the output dataset

---

# 4. Party Detail

File: `outputs/parties/{partyId}.json`

Purpose:
- Detailed party view for drill-down pages

Structure:

{
  "party_id": "string",
  "party_name": "string",

  "stats": {
    "members_count": number,
    "total": number,
    "relevant": number,
    "neutral": number,
    "non_relevant": number
  },

  "top_topics": [
    {
      "topic": "string",
      "count": number
    }
  ],

  "members": [
    {
      "member_id": "string",
      "name": "string",
      "interventions_total": number,
      "relevant_count": number,
      "neutral_count": number,
      "non_relevant_count": number,
      "top_topics": [
        { "topic": "string", "count": number }
      ]
    }
  ]
}

---

# Field Definitions

member_id:
- Stable unique identifier (slug or canonical ID)
- Deterministic across runs

topics:
- Extracted semantic subjects of the intervention
- Normalized string labels (no duplicates with case differences)

confidence:
- Optional numeric score between 0 and 1
- Represents model confidence in classification

party_id:
- Stable unique identifier (slug or canonical party ID)
- Deterministic across runs

party_name:
- Display name used by the frontend

---

# Party Attribution Model (v0)

In version v0, party attribution is based on the last known party affiliation available in the input data at the time of processing.

This means:

- Each member is associated with a single party.
- If multiple stenograms contain different party affiliations for the same member, the analyzer will use the most recent one chronologically.
- Historical party changes are NOT preserved in this version.
- All member and party aggregates are computed using this last-known affiliation.

Implications:

- Party totals represent current-aligned aggregates.
- Past interventions are not re-attributed historically if the member changed party.
- This simplifies aggregation logic and keeps the dataset deterministic.

---

# Guarantees (v0)

- All counts match the actual intervention arrays
- Top topics derived strictly from processed interventions
- No silent field omissions
- Deterministic output for same input + same model version
- Party totals equal the sum of the member totals for members listed under that party in the same output version.

---

# Not Defined Yet

- Topic normalization strategy
- Multi-session aggregation window
- Cross-party metadata enrichment
- Versioning strategy beyond v0