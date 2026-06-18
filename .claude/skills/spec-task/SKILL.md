---
name: spec-task
description: Produce a self-contained functional spec through iterative clarification and codebase research. Use for defining what to build, acceptance criteria, edge cases, and implementation boundaries before planning or coding.
---

# /spec-task

Generic workflow skill for defining what to build.

## Phase 1: Load Context

Read:
1. `CLAUDE.md`
2. the project context documents that `CLAUDE.md` points to
3. `context/INDEX.md`
4. Relevant docs from `context/`
5. `project-delivery/decisions.md`
6. `project-delivery/learnings.md` when relevant

## Phase 2: Load Related Delivery Artifacts

1. Search `project-delivery/brainstorms/` for a matching brainstorm
2. Read `brainstorm.md` and `sprint.md` if present
3. Search `project-delivery/features/` for overlapping specs or plans

## Phase 3: Research Current State

Ground the spec in the actual codebase:
- types and interfaces
- frontend components and hooks
- backend endpoints, models, and schemas
- shared state or real-time behavior
- current user-visible behavior

## Phase 4: Iterative Requirements Conversation

Use structured rounds:

1. confirm current understanding
2. define expected behavior
3. identify boundaries and out-of-scope items
4. lock acceptance criteria and edge cases

Continue until there are no ambiguous or TBD items.

## Phase 5: Write The Spec

Create or reuse `project-delivery/features/NNN-[task-name]/spec.md`.

The spec should be self-contained and include:
- overview
- what exists today
- what changes
- functional requirements
- edge cases
- out of scope
- dependencies
- acceptance criteria
- glossary when domain language matters

Reference relevant context docs when they materially shape the feature.

## Phase 6: Approval

Summarize:
- what the task delivers
- FR count
- key edge cases
- acceptance criteria count
- remaining risks or open questions

Only move on once the spec is approved.
