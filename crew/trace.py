from __future__ import annotations
import json
from pathlib import Path
from threading import Lock
from .context import RunContext

_lock = Lock()
_path = Path(".cache") / "crew-runs.jsonl"
def write_trace(run: RunContext, status: str) -> None:
    _path.parent.mkdir(exist_ok=True)
    record = {"run_id": run.run_id, "query": run.query, "status": status, "events": run.events}
    with _lock, _path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
