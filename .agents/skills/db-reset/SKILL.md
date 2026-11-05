---
name: db-reset
description: Reset the database to a clean baseline — export current data as seed, regenerate the initial migration from models, and/or reload from seed files. Use when the schema has drifted, seed data needs refreshing, or you need a clean slate.
---

# /db-reset

Database baseline management skill for the Loka project.

## Overview

This skill manages the database lifecycle:
- **Export** current DB data into categorised JSON seed files
- **Reset** the DB to a clean state using migrations + seed data
- **Full** cycle: export → regenerate migration → reset

## Commands

| Argument | What it does |
|----------|-------------|
| `export` | Export current DB data to `seed/` JSON files |
| `reset` | Drop all tables, run migration, load seed data |
| `full` | Export current data, regenerate initial migration, then reset |
| *(empty)* | Show available options and current state |

## Architecture

### File Layout

```
app/backend/
├── alembic/versions/0001_initial_schema.py   # Baseline migration (all tables)
├── seed/
│   ├── seed_manifest.json                     # Table ordering + metadata
│   ├── 01_reference.json                      # Platform + ontology data
│   ├── 02_loka.json                           # Loka Platform tenant data
│   └── 03_vizlake.json                        # Vizlake tenant data
└── scripts/
    ├── export_seed_data.py                    # DB → JSON export
    ├── load_seed_data.py                      # JSON → DB import
    └── reset_db.py                            # Full reset orchestrator
```

### Data Categories

Seed data is split into 3 files based on ownership:

| File | Contains | Classification logic |
|------|----------|---------------------|
| `01_reference.json` | Platform tables (no `tenant_id`): tenants, users, loka_modules. Ontology/config tables: core_categories, core_roles, entity_type_definitions, relationship_type_definitions, mission configs, organisation_types, ai_models, tenant_loka_modules | Tables in `PLATFORM_TABLES` or `REFERENCE_TABLES` sets in export script |
| `02_loka.json` | All tenant-scoped rows where `tenant_id` matches the Loka Platform tenant | Filtered by deterministic UUID: `uuid5(NAMESPACE_DNS, "loka-platform.loka.app")` |
| `03_vizlake.json` | All tenant-scoped rows where `tenant_id` matches the Vizlake tenant | Filtered by deterministic UUID: `uuid5(NAMESPACE_DNS, "vizlake.loka.app")` |

### Self-Referential FK Handling

Tables with self-referential foreign keys (e.g., `entity_missions.parent_id`, `artifacts.parent_id`) use a two-pass insert strategy:
1. Insert all rows with self-ref columns set to NULL
2. Update the self-ref columns to their actual values

The list of self-ref columns is maintained in `SELF_REF_COLS` in the export script and stored in `seed_manifest.json`.

### Initial Migration Generation

The initial migration is generated **offline** from SQLAlchemy model metadata (no running DB required):
1. Import `Base.metadata` from `app.models`
2. Use Alembic's `render_python_code()` to render `op.create_table()` calls for all `sorted_tables`
3. Add index creation ops
4. Write as `0001_initial_schema.py`

This avoids the autogenerate diff problem where Alembic compares against an existing DB and only emits changes.

## Procedures

### Export (`/db-reset export`)

```bash
cd app/backend && source venv/bin/activate
python scripts/export_seed_data.py
```

This connects to the live DB and writes 3 JSON files + manifest to `seed/`.

### Reset (`/db-reset reset`)

```bash
cd app/backend && source venv/bin/activate
python scripts/reset_db.py
```

This:
1. Drops all tables (CASCADE)
2. Runs `alembic upgrade head`
3. Loads seed data in order: reference → loka → vizlake

### Regenerate Migration (`/db-reset full`)

1. Delete existing migration files in `alembic/versions/`
2. Run the migration generator script:

```python
# Generate from model metadata (offline, no DB needed)
import sys
sys.path.insert(0, '.')
from app.models import Base
from alembic.autogenerate import render_python_code
from alembic.operations import ops

upgrade_ops = ops.UpgradeOps(ops=[])
for table in Base.metadata.sorted_tables:
    upgrade_ops.ops.append(
        ops.CreateTableOp(table.name, list(table.columns), schema=None)
    )
for table in Base.metadata.sorted_tables:
    for idx in table.indexes:
        kw = {}
        pg_opts = idx.dialect_options.get('postgresql', {})
        if pg_opts.get('where') is not None:
            kw['postgresql_where'] = pg_opts['where']
        upgrade_ops.ops.append(
            ops.CreateIndexOp(idx.name, table.name,
                [c.name for c in idx.columns], unique=idx.unique, **kw)
        )
code = render_python_code(upgrade_ops)
# Write as alembic/versions/0001_initial_schema.py with proper header
```

3. Then run the full reset

### Adding a New Tenant

To add a new tenant's data as a separate seed file:
1. Add the tenant UUID to `export_seed_data.py` (follow the `uuid5(NAMESPACE_DNS, ...)` pattern)
2. Add a new entry in the classification logic
3. Re-export

## Key Decisions

- **JSON over Python**: Seed data is stored as JSON (not generated Python code) for readability, diffability, and language-agnostic use
- **Deterministic UUIDs**: Tenant IDs use `uuid5(NAMESPACE_DNS, "<slug>.loka.app")` for reproducibility
- **ON CONFLICT DO NOTHING**: All inserts are idempotent — safe to re-run
- **Manifest-driven loading**: `seed_manifest.json` controls table order and self-ref metadata so the loader doesn't need to import model code
- **Offline migration generation**: Uses Alembic's rendering API against model metadata, not autogenerate-against-DB, to produce clean "create everything" migrations

## Updating Table Classification

If new models are added, check whether they should go into reference or tenant data:

- **Platform tables** (inherit from `Base` directly, no `tenant_id`): add to `PLATFORM_TABLES` in `export_seed_data.py`
- **Reference/ontology tables** (have `tenant_id` but are config data): add to `REFERENCE_TABLES`
- **Tenant data tables**: no change needed — default classification handles them
