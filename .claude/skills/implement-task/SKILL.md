---
name: implement-task
description: Execute an approved implementation plan phase by phase, updating status at each boundary and leaving the codebase in a working state after every phase. Use for non-trivial feature execution, resumed long-running work, and plan-driven implementation.
---

# /implement-task

Generic workflow skill for executing a plan safely and recoverably.

## Phase 0: Load Context And Determine Starting Point

Read in order:
1. `CLAUDE.md`
2. the project context documents that `CLAUDE.md` points to
3. `context/INDEX.md`
4. Relevant docs from `context/`
5. feature `status.md`
6. feature `plan.md`
7. feature `decisions.md`
8. feature `implementation-notes.md`
9. feature `spec.md`

`status.md` is the source of truth for where to resume.

## Phase N: Execute The Current Phase

For each phase:

1. mark it in progress in `status.md`
2. re-read the phase instructions in `plan.md`
3. verify prior phase outputs if applicable
4. implement the scoped changes only
5. run the relevant verification commands
6. commit at the phase boundary when appropriate
7. mark the phase complete in `status.md`

## Implementation Rules

- keep the system working after every phase
- prefer the architecturally correct fix over a quick workaround
- respect `decisions.md`, `implementation-notes.md`, and relevant `context/` docs
- update `project-delivery/learnings.md` when a reusable lesson emerges

## Recovery Rule

If context compacts or implementation is resumed later, restart from `status.md` rather than memory.
