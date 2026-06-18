---
name: review-task
description: Audit an implementation against its spec, plan, project context, and quality gates. Use for implementation verification, pre-merge audits, spec compliance checks, and structured review reports with severity-based findings.
---

# /review-task

Generic workflow skill for structured implementation audits.

## Phase 1: Load Context

Read:
1. `CLAUDE.md`
2. the project context documents that `CLAUDE.md` points to
3. `context/INDEX.md`
4. Relevant docs from `context/`
5. feature `spec.md`
6. feature `plan.md`
7. feature `decisions.md`
8. feature `implementation-notes.md`
9. feature `status.md`

## Phase 2: Run Mechanical Checks

Run the checks relevant to the changed surface:
- backend lint or tests
- frontend lint, formatting, types, and build
- any task-specific verification called for in the plan

Capture failures exactly.

## Phase 3: Audit Spec Compliance

Systematically review:
- each functional requirement
- each edge case
- each acceptance criterion
- any out-of-scope guardrails

Mark outcomes clearly.

## Phase 4: Review Code Quality

Check changed files for:
- convention fit
- context fit
- adherence to decisions and implementation notes
- security and performance risks
- over-engineering or under-engineering

## Phase 5: Validate Findings

For non-trivial findings, use supporting review or validation agents where helpful.

## Phase 6: Write Audit

Write `project-delivery/features/NNN-[task-name]/audit.md` with:
- verdict
- mechanical check results
- spec compliance tables
- findings by severity
- documentation check
- next steps
