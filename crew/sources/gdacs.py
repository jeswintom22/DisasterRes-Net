from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
import math
from typing import Any
import xml.etree.ElementTree as ET
import httpx

from crew.cache import get as cache_get, set as cache_set
from .base import EventCandidate, ProvenanceRef

GDACS_NAMESPACES = {
    "gdacs": "http://www.gdacs.org",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "georss": "http://www.georss.org/georss",
}

EVENT_TYPE_MAP = {
    "EQ": "earthquake",
    "FL": "flood",
    "TC": "cyclone",
    "WF": "wildfire",
    "VO": "volcano",
    "DR": "drought",
    "TS": "tsunami",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 points."""
    r_earth = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def _parse_pub_date(raw_date: str) -> str:
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date.strip())
        return dt.date().isoformat()
    except Exception:
        return raw_date.strip()[:10]


def parse_gdacs_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse GDACS GeoRSS XML string into normalized alert dictionaries."""
    if not xml_text or not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _parse_pub_date(item.findtext("pubDate") or "")

        country = (item.findtext("gdacs:country", default="", namespaces=GDACS_NAMESPACES) or "").strip()
        raw_etype = (item.findtext("gdacs:eventtype", default="", namespaces=GDACS_NAMESPACES) or "").strip().upper()
        event_id = (item.findtext("gdacs:eventid", default="", namespaces=GDACS_NAMESPACES) or "").strip()
        alert_level = (item.findtext("gdacs:alertlevel", default="", namespaces=GDACS_NAMESPACES) or "").strip()

        lat_str = (item.findtext("geo:lat", default="", namespaces=GDACS_NAMESPACES) or "").strip()
        lon_str = (item.findtext("geo:long", default="", namespaces=GDACS_NAMESPACES) or "").strip()

        lat: float | None = None
        lon: float | None = None
        if lat_str and lon_str:
            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                lat, lon = None, None
        else:
            point_str = (item.findtext("georss:point", default="", namespaces=GDACS_NAMESPACES) or "").strip()
            if point_str and " " in point_str:
                parts = point_str.split()
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                except ValueError:
                    lat, lon = None, None

        event_type = EVENT_TYPE_MAP.get(raw_etype, raw_etype.lower() or "disaster")
        if not event_id:
            event_id = f"gdacs-{raw_etype.lower() or 'ev'}-{abs(hash(title or link)) % 100000}"
        else:
            event_id = f"gdacs-{raw_etype.lower()}-{event_id}"

        full_desc = title
        if description and description.lower() not in title.lower():
            full_desc = f"{title}. {description}".strip(". ")
        if alert_level:
            full_desc = f"[GDACS {alert_level} Alert] {full_desc}"

        items.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "event_date": pub_date,
                "title": title,
                "description": full_desc[:1000],
                "country": country,
                "link": link or "https://www.gdacs.org",
                "lat": lat,
                "lon": lon,
            }
        )
    return items


class GDACSSource:
    url = "https://www.gdacs.org/xml/rss.xml"

    def _fetch_items(self, offline: bool = False) -> list[dict[str, Any]]:
        cached = cache_get("gdacs_rss_v1")
        if isinstance(cached, list):
            return cached
        if offline:
            return []
        try:
            response = httpx.get(
                self.url,
                timeout=8.0,
                headers={"User-Agent": "DisasterRes-Net/1.0"},
            )
            if response.status_code != 200:
                return []
            items = parse_gdacs_xml(response.text)
            if items:
                cache_set("gdacs_rss_v1", items, ttl_s=3600)
            return items
        except httpx.HTTPError:
            return []

    def active_crisis_countries(self, offline: bool = False) -> set[str]:
        """Return lowercase country names currently having active GDACS alerts."""
        countries: set[str] = set()
        for item in self._fetch_items(offline=offline):
            raw_c = str(item.get("country") or "")
            for part in raw_c.split(","):
                clean_c = part.strip().lower()
                if clean_c:
                    countries.add(clean_c)
        return countries

    def search(
        self,
        location: str,
        *,
        locality: str | None = None,
        country: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        hazard_hint: str | None = None,
        max_distance_km: float = 300.0,
        limit: int = 5,
        offline: bool = False,
    ) -> list[EventCandidate]:
        items = self._fetch_items(offline=offline)
        if not items:
            return []

        tokens: list[str] = []
        for piece in (locality, country, *(location or "").split(",")):
            if piece and piece.strip():
                tok = piece.strip().lower()
                if len(tok) >= 3 and tok not in tokens:
                    tokens.append(tok)

        scored: list[tuple[float, EventCandidate]] = []
        for item in items:
            haystack = f"{item.get('title', '')} {item.get('description', '')} {item.get('country', '')}".lower()
            item_country = str(item.get("country") or "").lower()

            match_score = 0.0
            if locality and locality.lower() in haystack:
                match_score += 2.0
            for tok in tokens:
                if tok in item_country:
                    match_score += 1.5
                elif tok in haystack:
                    match_score += 1.0

            item_lat = item.get("lat")
            item_lon = item.get("lon")
            if lat is not None and lon is not None and item_lat is not None and item_lon is not None:
                dist_km = haversine_km(lat, lon, float(item_lat), float(item_lon))
                if dist_km <= max_distance_km:
                    # Closer epicenters score higher (up to +2.5 at 0 km)
                    proximity_bonus = 2.5 * (1.0 - (dist_km / max(1.0, max_distance_km)))
                    match_score += max(0.5, proximity_bonus)

            if match_score <= 0.0:
                continue

            if hazard_hint and item.get("event_type") == hazard_hint.lower():
                match_score += 1.0

            candidate = EventCandidate(
                event_id=str(item["event_id"]),
                event_type=str(item["event_type"]),
                event_date=str(item.get("event_date") or ""),
                description=str(item.get("description") or item.get("title") or ""),
                source=ProvenanceRef("GDACS", str(item.get("link") or "https://www.gdacs.org")),
            )
            scored.append((match_score, candidate))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cand for _, cand in scored[:limit]]


def search_gdacs(
    location: str,
    *,
    locality: str | None = None,
    country: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    hazard_hint: str | None = None,
    max_distance_km: float = 300.0,
    limit: int = 5,
    offline: bool = False,
) -> list[EventCandidate]:
    return GDACSSource().search(
        location,
        locality=locality,
        country=country,
        lat=lat,
        lon=lon,
        hazard_hint=hazard_hint,
        max_distance_km=max_distance_km,
        limit=limit,
        offline=offline,
    )
