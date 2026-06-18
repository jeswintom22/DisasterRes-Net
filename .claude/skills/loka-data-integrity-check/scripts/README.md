# scripts/

Two read-only scripts that drive the data integrity check.

## extract.py

Dumps the entity, time_entry, and relationship tables to a JSON file
inside a read-only repeatable-read transaction.

```bash
# default — uses app/backend/.env.prod, output /tmp/loka_integrity_snapshot.json
python .claude/skills/loka-data-integrity-check/scripts/extract.py

# different env file
python .claude/skills/loka-data-integrity-check/scripts/extract.py --env .env.local

# absolute env path (e.g. /tmp/scratch.env)
python .claude/skills/loka-data-integrity-check/scripts/extract.py --env-path /tmp/scratch.env

# custom output
python .claude/skills/loka-data-integrity-check/scripts/extract.py --out /tmp/snap.json
```

The script auto-discovers the repo root by walking up from its own
location looking for `app/backend/<env-file>`, so it works whether you
run it from the repo root, the backend dir, or the skill directory.

Requires the backend venv to be active (uses `asyncpg`).

## validate.py

Reads a snapshot from `extract.py` and runs the 5 rule checks. Prints a
human report and writes a structured findings JSON file.

```bash
python .claude/skills/loka-data-integrity-check/scripts/validate.py \
    /tmp/loka_integrity_snapshot.json

# custom findings path
python .claude/skills/loka-data-integrity-check/scripts/validate.py \
    /tmp/loka_integrity_snapshot.json \
    --findings-out /tmp/my_findings.json
```

Exit codes:
- `0` — all 5 rules passed
- `1` — at least one rule had violations
- `2` — invocation error (missing snapshot, etc.)

The findings JSON has shape:

```json
{
  "snapshot_exported_at": "...",
  "snapshot_env": ".env.prod",
  "all_passed": false,
  "rules": [
    {
      "rule": "rule1",
      "title": "project → 1 organisation",
      "passed": false,
      "active_total": 48,
      "distribution": {"0": 1, "1": 47},
      "violators": [{"id": "...", "name": "...", "tenant_id": "...", "count": 0}]
    },
    ...
  ]
}
```

The AI driving the integrity-check workflow consumes this file directly.
