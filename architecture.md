# architecture.md

## Current direction 

### Runtime
- Local, periodic batch processing (CLI-style)

### Language
- Python is preferred for the analyzer due to:
  - fast iteration on text processing + RAG experimentation
  - strong ecosystem for NLP / evaluation loops
  - fits the “run locally, generate artifacts” workflow

### Output contract
- Generates structured artifacts the frontend can read and display.
- Likely shape: JSON files (exact schema will be defined iteratively).

## High-level architecture (conceptual)

1. Ingest stenograms
2. Parse structure (session, speakers, interventions)
3. Determine or retrieve session topic context
4. Classify each intervention:
   - relevant / neutral / non-relevant to session topic
5. Extract topics per intervention
6. Aggregate per member:
   - categorized interventions lists
   - top 20 topics
7. Export frontend-readable dataset

## RAG concept in this project (conceptual)

RAG will be used to retrieve context that improves classification quality, such as:
- session topic description
- bill descriptions or agenda items (when available)
- surrounding debate context

## MCP concept in this project (conceptual)

Instead of free-form “write anything anywhere” behavior, the agent/processor should operate through structured actions such as:
- store intervention classification
- update per-member aggregates
- write output artifacts