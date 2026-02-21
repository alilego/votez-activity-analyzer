# goal.md

## Purpose

Build a local, standalone analyzer that processes Romanian Parliament stenograms and outputs structured data that the frontend can display.

This project is meant to be run periodically on a developer machine to refresh the dataset.

## Target outcome

Given a set of stenograms of parliament activity, generate a dataset that correlates each parliament member with:

- A list of interventions classified as:
  - relevant to the session topic
  - neutral
  - non-relevant (off-topic) to the session topic
- The top 20 debate subjects (topics) where the member intervened

## Operating mode

- Runs locally on the developer machine
- Runs periodically (manually or cron job)
- Outputs data in a frontend-friendly format so the UI can pick it up and display it

## Out of scope (for now)

- Deploying this as a hosted service
- Tight integration into the rest of the backend
- Final schema guarantees (we’ll iterate as the frontend needs evolve)
- Any commitment to a specific AI vendor/model or vector DB choice