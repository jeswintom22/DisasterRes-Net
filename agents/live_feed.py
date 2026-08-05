"""Small persistent hand-off between the monitoring agent and the web UI."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterable

from agents.schemas import DisasterEvent


LIVE_EVENTS_PATH = os.path.join("results", "live_disaster_events.json")


def publish_events(events: Iterable[DisasterEvent]) -> None:
    """Atomically publish the latest observed incidents for dashboard readers."""
    os.makedirs(os.path.dirname(LIVE_EVENTS_PATH), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "events": [
            {
                "event_id": event.event_id,
                "source": event.source,
                "disaster_class": event.disaster_class,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "place_name": event.place_name,
                "magnitude_or_severity": event.magnitude_or_severity,
                "event_time": event.event_time.isoformat(),
                "title": event.title,
            }
            for event in events
        ],
    }
    temporary_path = f"{LIVE_EVENTS_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2)
    os.replace(temporary_path, LIVE_EVENTS_PATH)
