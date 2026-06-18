---
name: brainstorm
description: Explore a feature idea, assess complexity, research the codebase, ask clarifying questions, and recommend the right delivery path. Use for new features, meaningful changes, feature ideation, scoping, and any request that should start with why before implementation.
---

# /brainstorm

Generic workflow skill for exploring a feature idea before specs or code.

## Phase 0: Assess Complexity

Quickly inspect the codebase area mentioned by `$ARGUMENTS` and classify the work:

| Classification | Signals | Action |
|----------------|---------|--------|
| Trivial | typo, copy tweak, obvious one-file fix, mechanical config change | tell the user no brainstorm is needed and implement directly |
| Simple | clear scope, low risk, small file surface, no architectural choices | offer direct implementation or a lightweight workflow |
| Needs brainstorming | multiple files, unclear scope, architecture choices, dependencies, product ambiguity | continue with the full brainstorm flow |

Present the classification before going deeper.

## Phase 1: Load Project Context

Read:
1. `CLAUDE.md`
2. the project context documents that `CLAUDE.md` points to
3. `context/INDEX.md`
4. The specific docs from `context/` that matter for the request
5. `project-delivery/decisions.md`
6. `project-delivery/learnings.md` if it exists and is relevant

## Phase 2: Deep Research

Research the current state before proposing anything:

1. Search related code paths, types, models, schemas, hooks, services, and UI surfaces
2. Check `project-delivery/brainstorms/` and `project-delivery/features/` for overlap
3. Identify dependencies, existing patterns, and missing building blocks
4. Note which context docs shape the decision

Use parallel research when independent questions can be answered separately.

## Phase 3: Clarify The Problem

Present what you found, then ask focused questions covering:
- intent
- scope and exclusions
- actors and user journey
- dependencies
- constraints
- rollout risk

Base questions on codebase findings and project context, not generic templates.

## Phase 4: Propose Approaches

Present 2-3 approaches with:
- summary
- how it works
- pros
- cons
- risk: low, medium, or high
- effort: S, M, L, or XL
- likely files or systems touched

Give a clear recommendation.

## Phase 5: Recommend Delivery Path

Recommend one of:

- Just Do It
- Light pipeline: `/spec-task -> /implement-task`
- Standard pipeline: `/spec-task -> /plan-task -> /implement-task -> /review-task`
- Full pipeline: `/spec-task -> /plan-task -> /implement-task -> /review-task -> /capture-behavior`

## Phase 6: Write Outputs

Create `project-delivery/brainstorms/NNN-[feature-name]/`.

Always write `brainstorm.md`.
Write `sprint.md` when the work is multi-task or has a dependency chain.

Capture:
- problem statement
- research findings
- chosen approach
- rejected alternatives
- open questions
- recommended path

## Phase 7: Next Step

If direct implementation is appropriate, say so and continue.
Otherwise direct the user to the next canonical step using the generic command names.
