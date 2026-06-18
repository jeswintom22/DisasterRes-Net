# AI Workflow — Triage and Fix

This file is the operational playbook the AI follows when validate.py
finds violations. Read [rules.md](rules.md) first for *what* the rules
mean; this file is *how* to fix them safely.

## Mental model

For every failing rule, the goal is **defence in depth**:

1. Fix the *current state* in the database (the violators that exist now).
2. Fix the *future state* in the code (so the same violation can't happen
   again from the next API call or import run).

Doing only one is bad:
- DB-only fix → next import / endpoint call recreates the bug.
- Code-only fix → existing violators continue to break reports.

So every rule that fails gets *two* proposed fixes, and both go through
the same dry-run/approve/commit gates.

## Standard sequence per failing rule

### Step 1: Understand the violation

Read the relevant section of [rules.md](rules.md). Look at the violators
in `/tmp/loka_integrity_findings.json` — `report['rules'][i]['violators']`.
For Rule 5, look at `violator_rels` and `rel_type_breakdown` — those tell
you which writer planted the edges.

If the count of violators is small (≤5), enumerate them to the user. If
large, summarise by group (tenant, importer prefix, name pattern) and
show 5 representative samples.

### Step 2: Find the root cause in code

The user expects every violation to map to a specific writer. Use these
search seeds:

| Rule | Where to grep first |
|---|---|
| 1 | `Relationship\(.*belongs_to.*organisation` and `EntityProjectCreate` |
| 2 | `Relationship\(.*has_mission` and any `EntityMission(...)` constructor call |
| 3 | `Relationship\(.*related_to|relates_to` and `capture_activity` |
| 4 | `Relationship\(.*logged_activity|logged_for` |
| 5 | `Relationship\(.*logged_against` and any other `source_type=.time_entry.` writers |

Always check three places:
1. `app/backend/app/api/` — synchronous endpoint writers
2. `app/backend/app/services/` — service-layer writers (e.g. `activity_capture.py`)
3. `app/backend/scripts/` — seed and import scripts (these are the ones that
   historically planted bad data with synonym rel_types)

When you find a writer, list it as `file:line` in your message to the
user. Don't propose a fix until you've named every writer; missing one is
how recurrence happens.

### Step 3: Propose the code fix

The shape of a code fix depends on the failure mode:

**Failure: writer didn't create the edge at all (orphan).** Fix: make the
field required in the request schema (Pydantic), add validation that the
referenced entity exists in the same tenant, and write the edge in the
same DB transaction as the parent row.

```python
# Example: Rule 1, EntityProjectCreate
class EntityProjectCreate(_EntityBaseCreate):
    organisation_id: str   # required, was previously absent
    ...

# in create_project_with_groups()
org = await db.get(EntityOrganisation, body.organisation_id)
if not org or org.tenant_id != project.tenant_id:
    raise HTTPException(400, "organisation_id not found in this tenant")
db.add(Relationship(
    source_id=project.id, source_type='entity',
    target_id=org.id, target_type='organisation',
    rel_type='belongs_to',
    valid_from=datetime.utcnow(),
    tenant_id=project.tenant_id,
    created_by=current_user.id,
))
```

**Failure: writer used a wrong rel_type or target_type (typo / synonym).**
Fix: change the literal to the canonical name. Don't add a synonym
shim — that's what got us into the mess in the first place. The validator
already accepts the synonym so old data still passes.

**Failure: writer created a forbidden direct edge (Rule 5).** Fix: delete
the writer entirely. If something downstream reads it, refactor that read
to traverse via the activity. Don't gate the writer behind a feature flag
— get rid of it.

**Failure: writer created multiple edges where one is allowed (multi-link).**
Fix: add a uniqueness check at write time — query for existing active edges
with the same `(source_id, target_id, rel_type)` shape and either skip,
update, or error.

For each proposed code change, show the user the diff inline (file:line +
before/after) with one or two sentences explaining *why*.

### Step 4: Propose the DB fix

Write a one-shot Python script under `app/backend/scripts/` (or a tmp
directory if it's a one-time prod patch). The script must:

1. Take `--dry-run` (default) and `--commit` flags.
2. Connect via `asyncpg`, parse `DATABASE_URL` from a designated `.env`.
3. Open `BEGIN`, do all mutations, then `ROLLBACK` (dry-run) or `COMMIT`.
4. **Preflight assertions**: count the rows it expects to change, abort
   if the count differs from the snapshot — state has drifted.
5. **Post-mutation verification**: re-run the relevant rule query and
   show before/after.
6. Print row counts at every step so the human can sanity-check.

The historical fix script for the 2026-04-12 prod hierarchy fix was
[scripts/fix_hierarchy.py](../../../app/backend/scripts/fix_hierarchy.py).
Treat it as a template, not a library — each new fix needs its own
purpose-built script with its own assertions.

Common DB-fix patterns:

| Failure | DB action | Why |
|---|---|---|
| Orphan parent edge | `INSERT INTO relationships` | Add the missing edge |
| Wrong rel_type / target_type | `UPDATE relationships SET rel_type=..., target_type=...` | Rename without losing history |
| Multi-link | `UPDATE relationships SET valid_to = now()` on losers | Close, don't delete (temporal preservation) |
| Forbidden edge (Rule 5) | `UPDATE relationships SET valid_to = now()` (default) or `DELETE` (if user says hard delete) | Same |
| Orphan entity row that should be deleted | `DELETE FROM entity_*` | Only when the user explicitly says delete |

### Step 5: Apply, gated

Order matters:

1. **Apply code edits.** Run `cd app/backend && source venv/bin/activate &&
   ruff check <changed files> && ruff format --check <changed files>`.
   For frontend, `cd app/frontend && npm run lint && npx tsc --noEmit &&
   npm run build`. Fix anything that breaks before moving on.
2. **Run the DB script in `--dry-run` mode** against the target env. Show
   the row counts to the user. **Wait for explicit approval.**
3. **Run the DB script in `--commit` mode**. Show the final row counts.
4. Move to the next failing rule.

Never collapse 2 and 3 into one step. The human's eye on the dry-run
output is how we caught the `belongs_to | entity → entity` shape that
the original Rule 1 query missed — gate every prod write on a real human
look at real numbers.

### Step 6: Re-verify

After the last failing rule is fixed, re-run extract + validate. Confirm
`all_passed: true`. If not, loop back to Step 1 for the still-red rule.

## Common pitfalls (lessons from past runs)

These are real things that happened — read them so the same trap doesn't
catch you twice.

- **Do not assume the canonical edge shape is the only shape that exists.**
  The original Rule 1 check missed 170 `belongs_to | entity → entity`
  edges and reported 31 false orphans because it only looked for
  `target_type='organisation'`. The validator now accepts both shapes;
  new validators for new rules should do the same audit before locking
  in their query.

- **The relationship table is temporal.** Closing an edge with
  `valid_to = now()` is not destructive — old reports that reference the
  rel still work. Hard `DELETE` should only be used on entity tables (and
  only with explicit user approval).

- **Synonym rel_types are real.** Importers wrote `relates_to` instead of
  `related_to` and `logged_for` instead of `logged_activity` for over a
  thousand rows. The fix is to *rename in place* via UPDATE — much
  cheaper than re-creating rows. Verify zero unique-key conflicts before
  the rename.

- **Asserting expected row counts is non-negotiable.** Every UPDATE or
  DELETE in a fix script should be wrapped in `assert_eq("step name",
  rowcount, EXPECTED_COUNT)`. The first time a count differs, it usually
  means the snapshot is stale; abort and re-extract.

- **`source='system'` activities are not work.** They're audit log events
  generated by `capture_activity()` hooks. Don't try to attach them to
  missions — exclude them from Rule 3 (already done) and link them to the
  tenant's home org via `related_to | activity → entity` instead.

- **The default tenant home-org is not always obvious.** For Vizlake
  tenant `7ea46e0f-…` use `org-vizlake`. For the Loka platform tenant
  `c6fe34f8-…` use `org-loka-platform`. If a new tenant appears, ask the
  user before guessing.

- **Multiple writers means multiple fixes.** Don't fix only the API
  endpoint and forget the import script — they're both writers.

## What "find the root cause" looks like in practice

A real example, from the 2026-04-12 prod fix:

> Rule 4 had 1,178 orphan time entries. Sample IDs were `te-import-*`.
> Grep for `te-import-` → only one writer:
> [app/backend/scripts/import_timelog_from_excel.py](../../../app/backend/scripts/import_timelog_from_excel.py).
> Read the file → it was creating a `time_entry → activity` edge with
> `rel_type='logged_for'` and `target_type='activity'`. The canonical
> writer in `app/backend/app/api/time_entries.py` uses
> `rel_type='logged_activity'` and `target_type='entity'`.
> Root cause: importer used the wrong literal strings. Code fix: change
> two strings in the importer. DB fix: `UPDATE relationships SET
> rel_type='logged_activity', target_type='entity' WHERE rel_type='logged_for'
> AND source_type='time_entry' AND target_type='activity' AND valid_to IS NULL`
> (1,095 rows). Both fixes applied; Rule 4 went from 1,178 orphans to 83
> orphans (the remaining 83 had a different writer and were a separate fix).

That's the level of specificity to aim for: a named writer at a
named line, a one-line explanation of the bug, a code change that
prevents recurrence, a DB change that cleans the existing damage, and
a row count that proves it worked.
