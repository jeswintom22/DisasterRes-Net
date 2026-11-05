---
name: capture-behavior
description: Record implemented behavior as living BDD or Gherkin documentation. Use for creating or updating feature files, auditing behavioral coverage, refreshing behavior metadata, and documenting the current implemented behavior of a product surface.
---

# /capture-behavior

Generic workflow skill for behavioral documentation after implementation and review.

## Phase 1: Load Context

Read:
1. `CLAUDE.md`
2. the project context documents that `CLAUDE.md` points to
3. `context/INDEX.md`
4. `context/bdd-surface-map.md`
5. `bdd/bdd-metadata.json` if it exists
6. any relevant feature `spec.md` or `audit.md`

## Phase 2: Identify The Target

Determine the requested target surface from `context/bdd-surface-map.md`.

If the request is broad:
- use `all` for full regeneration
- use `status` or equivalent metadata inspection when the user wants freshness only
- scope the work to the surfaces materially affected by the implementation when possible

## Phase 3: Read Source Code First

Before writing any scenarios:
- read the implemented frontend behavior
- read the backend contracts or event behavior that make it observable
- review related specs and audits when available

Do not generate from memory.

## Phase 4: Write Or Update Feature Files

Feature files should be:
- behavioral, not implementation-led
- user-observable
- concrete and specific
- aligned with the actual current system

Use the target map to preserve file grouping and ID prefixes.

## Phase 5: Update Metadata

If `bdd/bdd-metadata.json` is part of the workflow for the target:
- update freshness details
- update file lists and scenario counts
- mark the refreshed surfaces current

## Phase 6: Report Coverage

Summarize what was updated, what remains stale, and any gaps between the implemented behavior and existing behavioral documentation.
