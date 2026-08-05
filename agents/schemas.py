"""Shared data contracts for the agentic monitoring layer."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DisasterEvent:
    event_id: str
    source: str
    disaster_class: str
    latitude: float
    longitude: float
    place_name: Optional[str]
    magnitude_or_severity: Optional[float]
    event_time: datetime
    title: str


@dataclass
class ImageCandidate:
    url: str
    source_tweet_id: str
    tweet_created_at: datetime
    disaster_class: str
    event_id: str
    event_source: str
    latitude: float
    longitude: float
    place_name: Optional[str]

