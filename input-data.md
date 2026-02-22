# 📥 Input Data Specification  
`votez-activity-analyzer`

This document defines the structure and expectations of all input files consumed by the analyzer.

The analyzer runs locally and reads structured JSON files from the `input/` directory.

---

# 📂 Directory Structure

```
input/
│
├── stenograme/
│   ├── stenograma_YYYY-MM-DD_X.json
│   ├── stenograma_YYYY-MM-DD_Y.json
│   └── ...
│
├── toti_deputatii.json
└── toti_senatorii.json
```

---

# 1️⃣ Members Registry Files

These files represent the **latest known snapshot** of all Members of Parliament.

They are treated as the authoritative source for:
- Member identity
- Party affiliation
- Constituency
- Profile URL

Party affiliation is considered valid **as of last known input**.

---

## `toti_deputatii.json`

## `toti_senatorii.json`

Both files follow the same structure.

### Structure

```json
{
  "metadata": {
    "build_date": "ISO_TIMESTAMP",
    "total_count": number
  },
  "members": [
    {
      "name": "FULL NAME",
      "profile_url": "URL",
      "circumscriptie": "STRING",
      "party": "SHORT_CODE",
      "party_full_name": "STRING"
    }
  ]
}
```

---

### Relevant Fields for Analyzer

| Field | Required | Purpose |
|-------|----------|----------|
| `name` | ✅ | Used to match speech speaker |
| `party` | ✅ | Used for party-level aggregation |
| `party_full_name` | Optional | Frontend display |
| `circumscriptie` | Optional | Contextual data |
| `profile_url` | Optional | Enrichment |

---

### Important Assumptions

- Names are unique across the full registry.
- Party code (`party`) is stable and short (e.g., `PSD`, `PNL`, `USR`, `AUR`, etc.).
- These files represent the latest known political affiliation snapshot.
- Historical party changes are currently not tracked.

---

# 2️⃣ Stenogram Files

Each file inside `input/stenograme/` represents a parliamentary session.

---

## Structure

```json
{
  "source_url": "STRING",
  "session_id": "STRING",
  "stenograma_date": "YYYY-MM-DD",
  "initial_notes": "STRING",
  "speeches": [
    {
      "speaker": "STRING",
      "text": "STRING",
      "text2": "STRING (optional)",
      "text3": "STRING (optional)"
    }
  ]
}
```

---

## Relevant Fields

| Field | Required | Purpose |
|--------|----------|----------|
| `stenograma_date` | ✅ | Used for time-based grouping |
| `speeches` | ✅ | Core data |
| `speaker` | ✅ | Used for member matching |
| `text`, `text2`, `text3` | Optional | Speech content |

---

# 3️⃣ Speaker Normalization Rules

The `speaker` field contains prefixes and annotations that must be cleaned before matching.

### Observed Patterns

- `Domnul Ciprian-Constantin Şerban`
- `Doamna Ecaterina-Mariana Szőke`
- `Domnul Gabriel Andronache (din sală)`
- `Domnul Mihai-Adrian Enache (fără microfon)`

---

### Required Cleaning Steps

Before matching with registry:

1. Remove prefixes:
   - `Domnul`
   - `Doamna`

2. Remove any content inside parentheses:
   - `(din sală)`
   - `(fără microfon)`
   - Any other annotation

3. Trim whitespace

4. Normalize case (case-insensitive comparison)

5. (TBD) Diacritics normalization strategy

---

# 4️⃣ Matching Logic (Initial Version)

Speaker → Member mapping strategy:

1. Clean speaker string
2. Attempt exact match with:
   - `toti_deputatii.json`
   - `toti_senatorii.json`
3. Fallback: case-insensitive comparison
4. If no match → mark as `unmatched`

Future enhancements (not in initial scope):

- Fuzzy matching
- Alias mapping file
- Manual override configuration
- Historical party resolution by date

---

# 5️⃣ Input Processing Assumptions

- All files are valid JSON.
- Stenograms are additive over time.
- Analyzer processes all available stenogram files per run.
- Files may contain speakers that are not MPs (e.g., ministers, guests).
- Unmatched speakers must not break execution.

---

# 6️⃣ Known Limitations (Current Version)

- No historical party tracking (last known affiliation used)
- No distinction between Chamber vs Senate in matching
- No validation of registry uniqueness
- No deduplication logic between sessions

---

# 7️⃣ Open Questions

1. Should party affiliation be time-aware?
2. Should Senate and Chamber be processed separately?
3. Should unmatched speakers be exported in a separate report?
4. Should we validate registry integrity on each run?
