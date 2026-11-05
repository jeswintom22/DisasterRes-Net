---
name: loka-db-clone
description: Destroy a Loka non-prod environment (dev or uat) and replace it with a literal clone of prod — schema and data. Snapshots prod's DDL via pg_dump (which carries the alembic_version pin), wipes the target's user schemas, then applies the prod schema and loads the prod data. Source is always prod, target is never prod.
---

# /loka-db-clone

Replace a Loka **dev** or **uat** environment database with a literal clone
of prod.  This is destructive — every user schema on the target is dropped
before prod's schema and data are restored.

## Why a clone (not a refresh)

The earlier "refresh" approach ran `alembic upgrade head` against an empty
target before loading prod data.  That meant the target's schema was always
local-alembic-head, which sometimes diverged from prod (e.g. data
migrations that require pre-existing tenant rows would abort against an
empty DB).

The clone approach instead snapshots prod's actual DDL via `pg_dump
--schema-only` plus the `alembic_version` row, and replays that on the
target.  The target ends up pinned to whatever migration revision prod is
currently at — never to local alembic head.  **The clone process never
runs alembic.**

> **Source is always prod.  Target is never prod.**  Both the scripts and
> this skill enforce this; if either guard would be tripped, abort and tell
> the user instead of trying to bypass it.

## Inputs

The user picks **one** target environment:

| Target | Env file under `app/backend/` | Notes |
|--------|------------------------------|-------|
| `dev`  | `.env.local`                 | Local Postgres on the developer machine |
| `uat`  | `.env.uat`                   | Azure-hosted UAT Postgres |

If the user just types `/loka-db-clone` with no argument, ask them which
target they want.  If they type `/loka-db-clone prod` or anything that
resolves to prod, **refuse** and explain why.

## Workflow

The work is split across three Python scripts in `app/backend/scripts/`:

1. `db_refresh_snapshot.py` — read-only extract of prod into a timestamped
   directory under `/tmp/loka_db_clone_<UTC>/`.  Writes:
   - `schema.sql` — `pg_dump --schema-only` of prod plus the
     `alembic_version` data row, concatenated.
   - `01_reference.json`, `02_loka.json`, `03_vizlake.json` — categorised
     seed JSON the loader understands.
   - `seed_manifest.json` — table order, self-ref columns, schema file
     pointer, source host, exported-at timestamp.
2. `db_refresh_wipe.py --env <target>` — drops every non-system schema on
   the target (`public`, plus auxiliary schemas like `ai`) and recreates
   `public`.  Does NOT run alembic.
3. `db_refresh_load.py --env <target> --snapshot <dir>` — applies
   `schema.sql` via `psql` (single transaction, ON_ERROR_STOP=1), then
   loads the JSON buckets through the same async loader used by
   `load_seed_data.py`.

Drive the workflow yourself — do not invoke a Python orchestrator.  This
keeps each destructive step gated behind explicit user confirmation in the
chat.

### External tool dependencies

- `pg_dump` and `psql` (libpq client tools, version ≥ prod's server
  version) must be on PATH.  On macOS this comes from Homebrew's
  `postgresql@16` formula or similar.

## Procedure

### Phase 0 — Preconditions

1. Confirm the user is in the Loka repo and has `app/backend/venv` set up.
2. Read `app/backend/.env.prod` exists.  If not, abort: tell the user the
   prod env file is missing.
3. Read `app/backend/.env.local` (for `dev`) or `app/backend/.env.uat`
   (for `uat`) exists.  If not, abort.
4. Sanity check the target env file: `ENVIRONMENT` must NOT be `PROD` and
   `DATABASE_URL` must NOT contain `loka-prod`.  If either fails, refuse.
5. Verify `pg_dump` and `psql` are on PATH.  If not, tell the user to
   install the Postgres client tools and stop.

### Phase 1 — Pick the target

If the user did not pass a target as an argument, ask:

> Which environment do you want to clone prod into — `dev` or `uat`?
> This will destroy all existing data in that environment.

Echo back the resolved env file path so the user can sanity-check it.

### Phase 2 — Snapshot prod

Run the snapshot script.  This step is read-only — no confirmation needed.

```bash
cd app/backend && source venv/bin/activate
python scripts/db_refresh_snapshot.py
```

The script prints the output directory at the end (e.g.
`/tmp/loka_db_clone_20260412T120000Z`).  **Capture this path** — phase 5
needs it.  Also report the schema dump size, the row counts (reference /
loka / vizlake / total), and the alembic revision the snapshot pins to
(grep `COPY public.alembic_version` in `schema.sql` if the user asks).

If the snapshot fails (network, firewall, auth, missing pg_dump), do not
proceed.  Tell the user what failed and stop.  Common cause on Azure
Postgres: the developer's IP needs to be in the prod server's firewall
allow-list.

### Phase 3 — Confirm the destructive step

Before touching the target, summarise what is about to happen and ask
for an explicit confirmation.  Use the AskUserQuestion tool with options
that make the destructive choice unmistakable.  For example:

> I'm about to wipe **`<target>`** (`<env file name>`, host
> `<masked host>`).  This will:
>
> 1. drop every non-system schema on that database
> 2. recreate `public`
> 3. apply prod's schema dump (`schema.sql`, alembic pin `<rev>`)
> 4. load the prod snapshot at `<snapshot path>` (`<total>` rows)
>
> This is irreversible.  Continue?

Acceptable confirmation: an explicit "yes" / "proceed" choice.  If the
user says anything else, stop and leave the snapshot dir in place so they
can rerun later without re-snapshotting.

### Phase 4 — Wipe the target

```bash
cd app/backend && source venv/bin/activate
python scripts/db_refresh_wipe.py --env <target>
```

Report the dropped schema names and the recreated `public` owner.  Does
NOT run alembic.

### Phase 5 — Load the snapshot (schema + data)

```bash
cd app/backend && source venv/bin/activate
python scripts/db_refresh_load.py --env <target> --snapshot <snapshot path>
```

The loader applies `schema.sql` first (via `psql --single-transaction
--set ON_ERROR_STOP=1`), then loads the JSON buckets in one async
transaction.  Report inserted-row counts per bucket as the loader emits
them.

If the schema apply fails, the target is left in whatever state psql
rolled back to (single-transaction mode means clean rollback).  Do not
retry the load until you understand why the apply failed — usually a
left-over schema or extension on the target that the wipe didn't drop.

### Phase 6 — Verify and report

Connect to the target one more time and run:

```sql
SELECT version_num FROM alembic_version;
SELECT count(*) FROM users;
SELECT count(*) FROM entity_organisations;
```

Then report:

- target env, host, alembic revision (must match prod's)
- dropped schemas / counts during wipe
- snapshot directory used and where it is parked
- final row counts on the verified tables

## Safety rules

- The source is hard-coded to `.env.prod` in `db_refresh_snapshot.py`.
  Do not run a snapshot against any other env file.
- The target must be `dev` or `uat`.  Refuse any other value, including
  `prod`, `production`, `local-prod`, etc.  The shared
  `_db_refresh_common.assert_not_prod` helper double-checks this in each
  script as well.
- Never call `db_refresh_wipe.py` or `db_refresh_load.py` without an
  explicit `--env` argument.  Never edit those scripts to make the
  guard optional.
- Never re-run the snapshot step against the target file by mistake.
- Do not delete the snapshot directory once you finish — leave it for
  rollback / retry.
- Never run `alembic upgrade head` as part of this workflow.  The whole
  point is that the target's schema comes from prod's `pg_dump`, not from
  local migrations.
- If the schema apply fails, do NOT load the JSON.  Investigate the
  conflicting state on the target (extra schemas, extensions, leftover
  objects) and update `db_refresh_wipe.py` if needed.

## When to update this skill

- A new target env is added (e.g. a `staging`).  Update
  `_db_refresh_common.TARGET_ENVS` and add the env file mapping here.
- The seed file format changes.  The loader and snapshot share the
  manifest structure, so any change there must update both
  `export_seed_data.py` and `db_refresh_snapshot.py`.
- Prod adds a new auxiliary schema (e.g. another framework alongside
  `ai`).  The wipe already drops every non-system schema, so this is
  usually automatic — just verify the schema apply still works.
- The prod host changes.  Update the substring check in
  `_db_refresh_common.assert_not_prod` so the guard still bites.
