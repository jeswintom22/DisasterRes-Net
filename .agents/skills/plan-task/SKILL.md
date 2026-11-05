---
name: plan-task
description: Create a phased implementation plan from an approved spec by combining project context, codebase pattern analysis, and supporting research. Use for defining how to build a feature, sequencing phases, documenting decisions, and preparing implementation status tracking.
---

# /plan-task

Generic workflow skill for defining how to build an approved spec.

## Phase 1: Load Context

Read:
1. `CLAUDE.md`
2. the project context documents that `CLAUDE.md` points to
3. `context/INDEX.md`
4. Relevant docs from `context/`
5. `project-delivery/decisions.md`
6. `project-delivery/learnings.md` if relevant

## Phase 2: Load Inputs

1. Locate the feature directory under `project-delivery/features/`
2. Read `spec.md` completely
3. Check for related brainstorm artifacts
4. Warn if the spec is still draft

## Phase 3: Supporting Research

Gather implementation intelligence through parallel research when useful:
- official docs or external references
- codebase pattern analysis
- API or integration research
- local reusable references

Synthesize findings before writing the plan.

## Phase 4: Design The Plan

Break implementation into independently testable phases.

For each phase define:
- goal
- risk
- dependencies
- files to create or modify
- implementation steps
- anti-patterns to avoid
- verification strategy

Also identify:
- the critical path
- the riskiest phase
- project-specific constraints that came from `context/`

## Phase 5: Write Outputs

Write these files alongside the spec:
- `plan.md`
- `decisions.md`
- `implementation-notes.md`
- `status.md`

`status.md` should be ready for recovery-oriented execution by `/implement-task`.

## Phase 6: Approval

Present:
- architecture overview
- phase order
- risky areas and mitigations
- key decisions
- verification strategy

Wait for approval before implementation.
