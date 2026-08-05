"""Poll free, keyless disaster APIs and normalize them into DisasterEvent objects."""

import math
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional

import requests

from agents.schemas import DisasterEvent
from step1_collect_data import DISASTER_CLASSES

try:
    import feedparser
except ImportError:
    feedparser = None


USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
GDACS_URL = "https://www.gdacs.org/xml/rss.xml"
RELIEFWEB_URL = "https://api.reliefweb.int/v1/disasters"

EONET_CATEGORY_MAP = {
    "wildfires": "wildfire",
    "severeStorms": "hurricane",
    "floods": "flood",
    "landslides": "landslide",
    "earthquakes": "earthquake",
}

KEYWORD_CLASS_MAP = {
    "earthquake": "earthquake",
    "quake": "earthquake",
    "flood": "flood",
    "cyclone": "hurricane",
    "hurricane": "hurricane",
    "storm": "hurricane",
    "typhoon": "hurricane",
    "wildfire": "wildfire",
    "fire": "wildfire",
    "landslide": "landslide",
    "mudslide": "landslide",
}

SOURCE_PRIORITY = {"usgs": 0, "eonet": 1, "gdacs": 2, "reliefweb": 3}


class DisasterMonitorAgent:
    def fetch_all(self) -> List[DisasterEvent]:
        events: List[DisasterEvent] = []
        for fetcher in (self.fetch_usgs, self.fetch_eonet, self.fetch_gdacs, self.fetch_reliefweb):
            try:
                events.extend(fetcher())
            except Exception as exc:
                print(f"[OBSERVATION] {fetcher.__name__} failed: {exc}")
        return self._deduplicate(events)

    def fetch_usgs(self, min_mag: float = 5.0, hours_back: int = 24) -> List[DisasterEvent]:
        end = datetime.utcnow()
        start = end - timedelta(hours=hours_back)
        params = {
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": min_mag,
            "orderby": "time",
            "limit": 50,
        }
        response = requests.get(USGS_URL, params=params, timeout=15)
        response.raise_for_status()
        events: List[DisasterEvent] = []
        for feature in response.json().get("features", []):
            props = feature.get("properties", {})
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            if len(coords) < 2:
                continue
            event_time = datetime.utcfromtimestamp((props.get("time") or 0) / 1000)
            events.append(
                DisasterEvent(
                    event_id=str(feature.get("id") or props.get("code") or props.get("time")),
                    source="usgs",
                    disaster_class="earthquake",
                    latitude=float(coords[1]),
                    longitude=float(coords[0]),
                    place_name=props.get("place"),
                    magnitude_or_severity=_float_or_none(props.get("mag")),
                    event_time=event_time,
                    title=str(props.get("place") or "USGS earthquake"),
                )
            )
        return events

    def fetch_eonet(self, days: int = 10) -> List[DisasterEvent]:
        events: List[DisasterEvent] = []
        for category, disaster_class in EONET_CATEGORY_MAP.items():
            params = {"category": category, "status": "open", "days": days, "limit": 50}
            response = requests.get(EONET_URL, params=params, timeout=15)
            response.raise_for_status()
            for item in response.json().get("events", []):
                geometry = item.get("geometry") or []
                if not geometry:
                    continue
                latest = geometry[-1]
                coords = latest.get("coordinates") or []
                if len(coords) < 2:
                    continue
                event_time = _parse_datetime(latest.get("date")) or datetime.utcnow()
                events.append(
                    DisasterEvent(
                        event_id=str(item.get("id")),
                        source="eonet",
                        disaster_class=disaster_class,
                        latitude=float(coords[1]),
                        longitude=float(coords[0]),
                        place_name=None,
                        magnitude_or_severity=None,
                        event_time=event_time,
                        title=str(item.get("title") or category),
                    )
                )
        return events

    def fetch_gdacs(self) -> List[DisasterEvent]:
        if feedparser is None:
            print("[OBSERVATION] feedparser is not installed; skipping GDACS RSS.")
            return []
        feed = feedparser.parse(GDACS_URL)
        events: List[DisasterEvent] = []
        for entry in feed.entries:
            title = str(getattr(entry, "title", "GDACS alert"))
            disaster_class = _class_from_text(title + " " + str(getattr(entry, "summary", "")))
            if disaster_class is None:
                continue
            lat = _float_or_none(entry.get("geo_lat") or entry.get("lat"))
            lon = _float_or_none(entry.get("geo_long") or entry.get("long"))
            if lat is None or lon is None:
                continue
            events.append(
                DisasterEvent(
                    event_id=str(entry.get("id") or entry.get("guid") or entry.get("link") or title),
                    source="gdacs",
                    disaster_class=disaster_class,
                    latitude=lat,
                    longitude=lon,
                    place_name=None,
                    magnitude_or_severity=_severity_from_text(title),
                    event_time=_parse_datetime(entry.get("published")) or datetime.utcnow(),
                    title=title,
                )
            )
        return events

    def fetch_reliefweb(self, limit: int = 20) -> List[DisasterEvent]:
        params = {"appname": "disasterresnet", "limit": limit, "sort[]": "date:desc"}
        response = requests.get(RELIEFWEB_URL, params=params, timeout=15)
        response.raise_for_status()
        events: List[DisasterEvent] = []
        for item in response.json().get("data", []):
            fields = item.get("fields", {})
            disaster_class = _class_from_text(
                " ".join(
                    [
                        str(fields.get("name", "")),
                        " ".join(str(t.get("name", "")) for t in fields.get("type", []) if isinstance(t, dict)),
                    ]
                )
            )
            if disaster_class is None:
                continue
            coords = _reliefweb_coordinates(fields)
            if coords is None:
                continue
            events.append(
                DisasterEvent(
                    event_id=str(item.get("id")),
                    source="reliefweb",
                    disaster_class=disaster_class,
                    latitude=coords[0],
                    longitude=coords[1],
                    place_name=_reliefweb_place(fields),
                    magnitude_or_severity=None,
                    event_time=_parse_datetime(fields.get("date", {}).get("created")) or datetime.utcnow(),
                    title=str(fields.get("name") or "ReliefWeb disaster"),
                )
            )
        return events

    def _deduplicate(self, events: Iterable[DisasterEvent]) -> List[DisasterEvent]:
        sorted_events = sorted(events, key=lambda ev: (SOURCE_PRIORITY.get(ev.source, 99), -ev.event_time.timestamp()))
        kept: List[DisasterEvent] = []
        for event in sorted_events:
            duplicate_idx = self._find_duplicate(kept, event)
            if duplicate_idx is None:
                kept.append(event)
                continue
            current = kept[duplicate_idx]
            if SOURCE_PRIORITY.get(event.source, 99) < SOURCE_PRIORITY.get(current.source, 99):
                kept[duplicate_idx] = event
        return sorted(kept, key=lambda ev: ev.event_time, reverse=True)

    def _find_duplicate(self, kept: List[DisasterEvent], event: DisasterEvent) -> Optional[int]:
        for idx, other in enumerate(kept):
            if event.disaster_class != other.disaster_class:
                continue
            if abs((event.event_time - other.event_time).total_seconds()) > 6 * 3600:
                continue
            if _haversine_km(event.latitude, event.longitude, other.latitude, other.longitude) <= 50:
                return idx
        return None


def _parse_datetime(value: object) -> Optional[datetime]:
    if not value:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        try:
            return parsedate_to_datetime(text).replace(tzinfo=None)
        except (TypeError, ValueError):
            return None


def _float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _class_from_text(text: str) -> Optional[str]:
    lower = text.lower()
    for keyword, disaster_class in KEYWORD_CLASS_MAP.items():
        if keyword in lower and disaster_class in DISASTER_CLASSES:
            return disaster_class
    return None


def _severity_from_text(text: str) -> Optional[float]:
    lower = text.lower()
    if "red" in lower:
        return 3.0
    if "orange" in lower:
        return 2.0
    if "green" in lower:
        return 1.0
    return None


def _reliefweb_coordinates(fields: Dict) -> Optional[tuple]:
    countries = fields.get("country") or []
    for country in countries:
        if not isinstance(country, dict):
            continue
        location = country.get("location") or {}
        lat = _float_or_none(location.get("lat"))
        lon = _float_or_none(location.get("lon"))
        if lat is not None and lon is not None:
            return lat, lon
    return None


def _reliefweb_place(fields: Dict) -> Optional[str]:
    countries = fields.get("country") or []
    if countries and isinstance(countries[0], dict):
        return countries[0].get("name")
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
