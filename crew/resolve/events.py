from __future__ import annotations
from crew.sources.reliefweb import ReliefWebSource
from crew.sources.gdacs import GDACSSource
from crew.sources.base import EventCandidate

def resolve_event(location: str) -> EventCandidate:
    candidates = ReliefWebSource().search(location) + GDACSSource().search(location)
    return candidates[0] if candidates else ReliefWebSource.fallback(location)
