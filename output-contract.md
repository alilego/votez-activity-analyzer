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

---

# Guarantees (v0)

- All counts match the actual intervention arrays
- Top topics derived strictly from processed interventions
- No silent field omissions
- Deterministic output for same input + same model version

---

# Not Defined Yet

- Topic normalization strategy
- Multi-session aggregation window
- Cross-party metadata enrichment
- Versioning strategy beyond v0