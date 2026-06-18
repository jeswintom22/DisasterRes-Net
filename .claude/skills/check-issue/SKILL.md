---
name: check-issue
description: Fetch a GitHub issue by number, assess complexity, brainstorm and implement if active, then close it. Use when you want to process a GitHub issue end-to-end in one command.
---

# /check-issue

End-to-end GitHub issue handler: fetch, assess, implement, close.

## Phase 0: Load Context

1. Read `CLAUDE.md` and the project context it points to
2. Read `context/INDEX.md` and load relevant docs
3. Read `project-delivery/decisions.md`
4. Read `project-delivery/learnings.md` if relevant

## Phase 1: Fetch & Triage

1. Run `gh issue view <issue-number>` to fetch the issue details
2. Check the issue state:
   - If **closed**: tell the user the issue is already closed and stop
   - If **open**: continue

3. Summarize the issue to the user: title, description, labels, and what it's asking for

## Phase 2: Assess Complexity

Inspect the codebase area mentioned by the issue and classify:

| Classification | Signals | Action |
|----------------|---------|--------|
| Trivial | typo, copy tweak, obvious one-file fix | implement directly, skip brainstorm |
| Simple | clear scope, low risk, small file surface, no architectural choices | present a short plan, then implement |
| Needs brainstorming | multiple files, unclear scope, architecture choices, dependencies | run the full brainstorm flow (Phase 3) |

Present the classification and get user confirmation before proceeding.

## Phase 3: Brainstorm (if needed)

For non-trivial issues, follow the brainstorm skill workflow:

1. **Research** the current state — search related code paths, types, models, schemas, hooks, services
2. **Check** `project-delivery/brainstorms/` and `project-delivery/features/` for overlap
3. **Propose** 2-3 approaches with pros, cons, risk, effort, and files touched
4. **Recommend** an approach and get user confirmation

Write brainstorm output to `project-delivery/brainstorms/NNN-[feature-name]/brainstorm.md` if the issue is non-trivial.

## Phase 4: Implement

1. Make the code changes
2. Run relevant lint/format checks:
   - Backend: `ruff check . && ruff format --check .` (from `app/backend/`)
   - Frontend: `npx eslint <files>` and `npx tsc --noEmit` (from `app/frontend/`)
3. Fix any lint errors before proceeding

## Phase 5: Close

1. Close the issue with a comment summarizing what was done:
   ```
   gh issue close <issue-number> --comment "Fixed: <summary of changes>"
   ```
2. Report the result to the user

## Guidelines

- Always read the issue first — don't assume what it says
- Check if related issues exist that could be addressed together (mention them to the user)
- Respect the project's code style rules (`.claude/rules/code-style.md`)
- Use shadcn components, shared components, and TeamAvatar per project conventions
- Record architectural decisions in `project-delivery/decisions.md` if applicable
- Record reusable lessons in `project-delivery/learnings.md` if applicable
- Do NOT commit or push unless the user asks — just make the changes and close the issue
