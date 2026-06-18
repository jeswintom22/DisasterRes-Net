"""Read-only export of a Loka DB into a JSON snapshot for hierarchy analysis.

Dumps the entity tables, relationships, and time_entries to a single JSON
file so the validator (and the AI driving the integrity-check workflow)
can reason about the data offline without repeated round-trips. Runs the
export inside a read-only repeatable-read transaction — never writes.

Usage (run from anywhere inside the repo):
    python .claude/skills/loka-data-integrity-check/scripts/extract.py
    python .claude/skills/loka-data-integrity-check/scripts/extract.py --env .env.local
    python .claude/skills/loka-data-integrity-check/scripts/extract.py --out /tmp/snap.json

Default env is .env.prod (under app/backend/), default output is
/tmp/loka_integrity_snapshot.json.

The script discovers the repo root by walking upward from its own location
looking for `app/backend/.env.<env>` — so it works whether you run it from
the repo root, the backend dir, or the skill directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import asyncpg

# Tables to dump and which columns the validator needs from each.
# Keep this list narrow — the JSON file is committed alongside the skill
# only when an issue needs persistent reproduction.
TABLES = [
    ("entity_organisations", ["id", "name", "slug", "tenant_id", "status", "active"]),
    (
        "entity_projects",
        ["id", "name", "slug", "tenant_id", "status", "active", "created_at"],
    ),
    (
        "entity_missions",
        ["id", "title", "slug", "tenant_id", "parent_id", "created_at"],
    ),
    (
        "entity_activities",
        [
            "id",
            "name",
            "tenant_id",
            "status",
            "active",
            "source",
            "type",
            "started_at",
            "ended_at",
            "created_at",
        ],
    ),
    (
        "time_entries",
        ["id", "tenant_id", "person_id", "date", "minutes", "status", "created_at"],
    ),
    (
        "relationships",
        [
            "id",
            "source_id",
            "source_type",
            "target_id",
            "target_type",
            "rel_type",
            "valid_from",
            "valid_to",
            "tenant_id",
            "created_at",
        ],
    ),
]


def find_env_file(env_name: str) -> Path:
    """Walk upward from this script looking for app/backend/<env_name>.

    The skill lives at .claude/skills/loka-data-integrity-check/scripts/, but
    the env files live at app/backend/. We walk up until we find a directory
    that contains both `app/backend` and `.claude` (i.e. the repo root).
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "app" / "backend" / env_name
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"Could not find {env_name} in any app/backend/ dir walking up from {here}. "
        "Run from inside a Loka repo checkout, or pass --env-path with an absolute path."
    )


def parse_url(env_path: Path) -> dict[str, object]:
    """Pull DATABASE_URL out of an env file and split it into asyncpg kwargs.

    Handles both prod-style (`@host/db?ssl=require`) and local-style
    (`@localhost:5432/db`) URLs. SSL is auto-enabled for non-local hosts;
    local connections (localhost / 127.0.0.1) skip SSL since the local
    dev Postgres typically doesn't have it configured.
    """
    content = env_path.read_text()
    m = re.search(
        r"DATABASE_URL=postgresql\+asyncpg://([^:]+):([^@]+)@([^/]+)/([^?\s]+)",
        content,
    )
    if not m:
        raise SystemExit(f"Could not parse DATABASE_URL from {env_path}")
    user, pw, host_part, db = m.groups()
    if ":" in host_part:
        host, port_str = host_part.split(":", 1)
        port: int | None = int(port_str)
    else:
        host = host_part
        port = None
    is_local = host in ("localhost", "127.0.0.1")
    kwargs: dict[str, object] = {
        "user": user,
        "password": pw,
        "host": host,
        "database": db,
    }
    if port is not None:
        kwargs["port"] = port
    if not is_local:
        kwargs["ssl"] = "require"
    return kwargs


def to_jsonable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


async def dump(env_path: Path, out_path: Path) -> None:
    conn_kwargs = parse_url(env_path)
    print(
        f"# connecting host={conn_kwargs['host']} db={conn_kwargs['database']} "
        f"user={conn_kwargs['user']} env={env_path.name} (read-only)"
    )
    conn = await asyncpg.connect(**conn_kwargs)
    try:
        # Pin a single read-only snapshot so all tables come from the same
        # MVCC view — no torn reads if writes happen during the export.
        await conn.execute("BEGIN")
        await conn.execute("SET TRANSACTION READ ONLY, ISOLATION LEVEL REPEATABLE READ")
        snapshot: dict[str, object] = {
            "exported_at": datetime.utcnow().isoformat(),
            "env": env_path.name,
            "tables": {},
        }

        for table, cols in TABLES:
            col_list = ", ".join(cols)
            rows = await conn.fetch(f"SELECT {col_list} FROM {table}")  # noqa: S608
            data = [{col: to_jsonable(row[col]) for col in cols} for row in rows]
            snapshot["tables"][table] = data
            print(f"  {table:<22} {len(data):>7} rows")

        await conn.execute("ROLLBACK")
    finally:
        await conn.close()

    out_path.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env",
        default=".env.prod",
        help="env file name under app/backend/ (default: .env.prod)",
    )
    ap.add_argument(
        "--env-path",
        default=None,
        help="absolute path to an env file, overrides --env",
    )
    ap.add_argument(
        "--out",
        default="/tmp/loka_integrity_snapshot.json",
        help="output JSON path",
    )
    args = ap.parse_args()

    if args.env_path:
        env_path = Path(args.env_path).resolve()
        if not env_path.exists():
            print(f"env file not found: {env_path}", file=sys.stderr)
            sys.exit(2)
    else:
        env_path = find_env_file(args.env)
    out_path = Path(args.out).resolve()
    asyncio.run(dump(env_path, out_path))


if __name__ == "__main__":
    main()
