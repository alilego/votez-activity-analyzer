# votez-activity-analyzer
AI-powered analysis of parliamentary activity, classifying interventions and extracting debate topics from stenograms.

## What it does

Given stenograms of parliamentary sessions, it outputs a dataset that maps each parliament member to:
- interventions classified as relevant / neutral / non-relevant to the session topic
- top 20 debate subjects (topics) where the member intervened

## How it runs

- Runs locally on a developer machine
- Intended to be run periodically to refresh the dataset
- Produces frontend-friendly output artifacts (format/schema will evolve)

## Docs

- `goal.md` — what the project targets to achieve (requirements/scope)
- `architecture.md` — current technology and system architecture direction

## Status

Early stage: requirements + architecture are being defined step by step.