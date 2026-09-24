"""Per-run state. Images stay server-side and never enter LLM prompts."""
from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

current_run: ContextVar["RunContext | None"] = ContextVar("current_run", default=None)

@dataclass
class RunContext:
    query: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    artifacts: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def put_image(self, image: Any) -> str:
        handle = f"img_{len(self.artifacts) + 1}"
        self.artifacts[handle] = image
        return handle

    def image(self, handle: str) -> Any:
        return self.artifacts[handle]

    def emit(self, stage: str, message: str, **payload: Any) -> None:
        self.events.append({"stage": stage, "message": message, "at": datetime.now(timezone.utc).isoformat(), **payload})
