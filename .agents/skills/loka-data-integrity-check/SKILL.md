---
name: loka-data-integrity-check
description: Audit a Loka environment's database for hierarchy invariant violations and drive a guided code+data fix workflow. Always use this skill when the user mentions "data integrity", "hierarchy check", "rule violations", "orphan projects/missions/activities/time entries", "schema drift", or asks to audit/validate/check the prod or any other Loka database against the 5 hierarchy rules. Also use it when the user reports a hierarchy bug (e.g. "this project has no org", "this activity is orphaned") so that the root cause is found, not just patched. Source data is read-only; any DB or code change is gated behind an explicit human approval.
---

# /loka-data-integrity-check

Validate a Loka environment against the 5 hierarchy invariants and, for any
violation, drive a structured workflow that finds the root cause in code,
proposes a code-side fix to prevent recurrence, proposes a DB-side fix to
clean current state, and applies both only after human approval.

> **Default target is prod.** If no env is provided, this skill reads
> `app/backend/.env.prod`. The extract step is read-only; any mutation
> requires human approval and runs as a dry-run before committing.

## When to invoke

- The user types `/loka-data-integrity-check` or `/loka-data-integrity-check [env]`.
- The user reports a hierarchy bug ("project X has no organisation", "this
  activity isn't linked to a mission", "time entries are double-counted").
- The user wants to audit prod or a non-prod env before a release, after
  an import, or after a schema change.
- A migration just landed and you want to confirm it didn't violate any
  invariants.

If the user asks to *fix* a single hierarchy bug, still run the full
workflow — fixing without finding the root cause leaves the next bug
waiting in code.

## What it does (high level)

1. **Extract** — `scripts/extract.py` dumps a snapshot of the entity, time
   entry, and relationship tables to a JSON file. Read-only repeatable-read
   transaction. Defaults to `.env.prod`, output `/tmp/loka_integrity_snapshot.json`.
2. **Validate** — `scripts/validate.py` reads the snapshot, checks the 5
   rules, prints a human report, and writes a structured findings JSON to
   `/tmp/loka_integrity_findings.json`. Exit 0 = green, exit 1 = violations.
3. **Triage and fix (AI-driven)** — for every failing rule:
   - Locate where the rule could have been violated *at write time* in the
     backend code (`app/backend/app/`). This is the root cause.
   - Propose a code-side fix that tightens the rule at the source (e.g.,
     make a field required, add a uniqueness check, add a write-time guard,
     normalise a rel_type).
   - Propose a DB-side fix that cleans the existing violators, with a
     dry-run/commit pattern in a single transaction.
   - **Stop and ask the user** — show the violators, the proposed code
     diff, and the proposed DB script.
   - Only after explicit approval, apply code edits, run lint/format/build
     checks, then run the DB script in dry-run mode, then on second
     approval commit.
4. **Re-verify** — re-run extract + validate. Show before/after.

The 5 rules and their writers live in [rules.md](rules.md). The
step-by-step AI playbook for triage and fix lives in [workflow.md](workflow.md).
Read those before driving a session — they have the operational details
the body of this file deliberately keeps short.

## Inputs

| Argument | Default | Notes |
|---|---|---|
| `env` | `prod` | Resolves to `app/backend/.env.<env>`. Common: `prod`, `local`, `uat`. |

If the user just types `/loka-data-integrity-check`, default to `prod` and
say so up front so they can interrupt if they meant a different env.

## Procedure

### Phase 0 — Preconditions

1. Confirm we're inside a Loka repo checkout.
2. Resolve the env file: `app/backend/.env.<env>`. If missing, abort.
3. Confirm the backend venv exists (`app/backend/venv/`). If missing, ask
   the user to create it.
4. State the target env clearly to the user. Example:
   > Running data integrity check against **prod** (`app/backend/.env.prod`,
   > host `loka-prod-pgserver.postgres.database.azure.com`). The extract is
   > read-only.

### Phase 1 — Extract snapshot (read-only)

```bash
cd app/backend && source venv/bin/activate
python ../../.claude/skills/loka-data-integrity-check/scripts/extract.py --env .env.<env>
```

Or, equivalently, from the repo root:

```bash
.claude/skills/loka-data-integrity-check/scripts/extract.py --env .env.<env>
```

Report the row counts back to the user. If extract fails (firewall, auth,
network), stop and tell them what failed — common cause on Azure Postgres
is the developer's IP not being in the prod allow-list.

### Phase 2 — Validate

```bash
python .claude/skills/loka-data-integrity-check/scripts/validate.py /tmp/loka_integrity_snapshot.json
```

This prints a per-rule report and writes structured findings to
`/tmp/loka_integrity_findings.json`. Read that file and summarise to the
user: which rules passed, which failed, and how many violators per rule.

If `all_passed: true`, congratulate, stop here. The skill is done.

### Phase 3 — Triage each failing rule

Follow [workflow.md](workflow.md). For each failing rule, in order:

1. **Understand**: read the rule's "where it can be violated" section in
   [rules.md](rules.md). This tells you which API endpoints, importers, or
   scripts to grep.
2. **Locate root cause**: use Grep/Read on `app/backend/app/api/`,
   `app/backend/app/services/`, and `app/backend/scripts/` to find every
   place that writes the relationships or entity rows involved. List each
   call site with `file:line`.
3. **Propose code fix**: tighten the rule at the write site. Show the
   user the proposed diff(s) inline, explain the *why*, and wait for
   approval. Common patterns are in [workflow.md](workflow.md).
4. **Propose DB fix**: write a small Python script (model on the example
   in [workflow.md](workflow.md)) that fixes the violators in a single
   transaction with `--dry-run` (default) and `--commit` flags. Include
   row-count assertions so the script aborts if state has drifted since
   the snapshot. Show the dry-run output.
5. **Apply** in this order, each gated:
   - Code edits → run `ruff check` + `ruff format --check` on touched
     backend files; for frontend, `npm run lint && npx tsc --noEmit`.
   - DB script in `--dry-run` mode → human reviews row counts.
   - DB script in `--commit` mode → actually applies.
6. Move to the next failing rule.

### Phase 4 — Re-verify

Re-run extract + validate against the same env. Show the before/after
distribution per rule. Confirm `all_passed: true`. If anything is still
red, loop back to Phase 3 for the remaining rule.

### Phase 5 — Report

Summarise to the user:
- Which rules went red → green.
- Which code files changed (with line numbers).
- Which DB rows changed (commit message-style summary).
- Anything still outstanding (e.g. a violator the user explicitly opted to
  leave alone).

Don't commit anything to git yourself. Tell the user the diff is ready
for them to review and commit when they're ready. The skill never opens
PRs or pushes branches without an explicit ask.

## Safety rules

- The extract step is the only thing that runs by default. Everything else
  requires explicit user approval per step.
- Never run a fix script in `--commit` mode without first showing the
  `--dry-run` output and getting an OK.
- Never bypass an assertion in a fix script. If a row count drifts, the
  state of the world is no longer what the plan assumed — re-extract and
  re-plan.
- Never modify the env files. Never log database passwords.
- For prod, prefer fix scripts that hard-close (`valid_to = now()`)
  relationships rather than `DELETE`, unless the user explicitly says hard
  delete. The relationship table is temporal — closing preserves history.
- For non-relationship rows (entity_activities, entity_missions, etc.),
  hard delete is the only option; confirm twice before doing it on prod.

## When to update this skill

- A new hierarchy rule is added → update [rules.md](rules.md), add a
  `rule6_*` function in `scripts/validate.py`, update the SKILL.md
  description if needed.
- The data model gains or renames a primitive (e.g. a new
  `entity_widget` table) → update `scripts/extract.py` to dump it.
- A rel_type synonym shows up in the wild → add it to the `RULE*_SYNONYMS`
  set in `validate.py` so the validator stops counting it as a violation.
- A common root cause keeps recurring → document it in
  [workflow.md](workflow.md) so future runs find it faster.
