"""Bounded background jobs and Server-Sent Events for the Flask surface."""
from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from threading import Lock
from typing import Any
import time

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat-job")
_jobs: dict[str, "ChatJob"] = {}
_lock = Lock()

@dataclass
class ChatJob:
    id: str
    query: str
    queue: Queue = field(default_factory=Queue)
    result: dict[str, Any] | None = None
    error: str | None = None
    future: Future | None = None
    created: float = field(default_factory=time.monotonic)
    def emit(self, event: dict[str, Any]) -> None: self.queue.put(event)

def start_job(query: str) -> ChatJob:
    from .crew import run_disaster_crew
    job = ChatJob("", query)
    def work():
        try:
            payload = run_disaster_crew(query, event_sink=job.emit, return_payload=True)
            job.result = payload; job.emit({"stage": "complete", "message": "Report ready", "result": payload})
        except Exception as exc:  # endpoint returns the controlled error through SSE
            job.error = str(exc); job.emit({"stage": "error", "message": job.error})
    # Use a generated id rather than pre-running expensive source/inference work.
    from uuid import uuid4
    job.id = uuid4().hex
    with _lock: _jobs[job.id] = job
    job.future = _pool.submit(work)
    return job

def get_job(job_id: str) -> ChatJob | None:
    with _lock: return _jobs.get(job_id)
