from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class ProvenanceRef:
    name: str
    url: str
    tier: str = "metadata"

@dataclass(frozen=True)
class EventCandidate:
    event_id: str
    event_type: str
    event_date: str
    description: str
    source: ProvenanceRef

class SourceAdapter(Protocol):
    def search(self, location: str) -> list[EventCandidate]: ...
