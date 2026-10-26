from __future__ import annotations
import json
import logging
from pathlib import Path
from threading import Lock
from .context import RunContext

_logger = logging.getLogger(__name__)
_lock = Lock()
_ROOT = Path(__file__).resolve().parent.parent
_path = _ROOT / ".cache" / "crew-runs.jsonl"

def write_trace(run: RunContext, status: str) -> None:
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        record = {"run_id": run.run_id, "query": run.query, "status": status, "events": run.events}
        with _lock, _path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        _logger.warning("Failed to write crew trace: %s", exc)

