from __future__ import annotations
from dataclasses import dataclass
import re

# Common location spellings seen in disaster queries.  Keep this deterministic so
# source lookup remains auditable; an LLM should not silently alter a location.
LOCATION_ALIASES = {
    "wayand": "Wayanad, Kerala, India",
    "wayanad": "Wayanad, Kerala, India",
}

@dataclass(frozen=True)
class ResolvedQuery:
    intent: str
    location: str | None
    time_hint: str | None = None

def resolve_query(query: str) -> ResolvedQuery:
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return ResolvedQuery("unsupported", None)
    match = re.search(r"\b(?:at|in|near|around)\s+([\w .,'-]+?)(?:\s+(?:in|during|after|from)\b|[?.!,]|$)", cleaned, re.I)
    location = match.group(1).strip(" .,?!") if match else None
    if not location and len(cleaned.split()) <= 5:
        location = cleaned
    if location:
        location = LOCATION_ALIASES.get(location.lower(), location)
    return ResolvedQuery("investigate", location)
