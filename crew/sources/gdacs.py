from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import calendar
import math
import re
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

MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

GDACS_API_SEARCH_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"


def _parse_time_hint_bounds(time_hint: str | None) -> tuple[str, str, int] | None:
    if not time_hint:
        return None
    th = time_hint.strip().lower()
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", th)
    if not year_match:
        return None
    year = int(year_match.group(1))

    month = None
    for m_name, m_num in MONTH_NAMES.items():
        if re.search(r"\b" + m_name + r"\b", th):
            month = m_num
            break
    if month is None:
        m_dash = re.search(r"\b\d{4}-(\d{1,2})\b", th)
        if m_dash:
            m_val = int(m_dash.group(1))
            if 1 <= m_val <= 12:
                month = m_val

    if month is not None:
        last_day = calendar.monthrange(year, month)[1]
        from_date = f"{year:04d}-{month:02d}-01"
        to_date = f"{year:04d}-{month:02d}-{last_day:02d}"
    else:
        from_date = f"{year:04d}-01-01"
        to_date = f"{year:04d}-12-31"

    return from_date, to_date, year


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


def parse_gdacs_geojson(data: dict[str, Any]) -> list[dict[str, Any]]:
    features = data.get("features", [])
    if not isinstance(features, list):
        return []
    items: list[dict[str, Any]] = []
    for f in features:
        if not isinstance(f, dict):
            continue
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates")
        lat: float | None = None
        lon: float | None = None
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                lon = float(coords[0])
                lat = float(coords[1])
            except (ValueError, TypeError):
                lat, lon = None, None

        raw_etype = str(props.get("eventtype") or "").upper().strip()
        event_type = EVENT_TYPE_MAP.get(raw_etype, raw_etype.lower() or "disaster")
        raw_id = props.get("eventid") or abs(hash(str(props.get("name") or coords))) % 100000
        event_id = f"gdacs-{raw_etype.lower() or 'ev'}-{raw_id}"

        alert_level = str(props.get("alertlevel") or "").strip()
        from_date_raw = str(props.get("fromdate") or "").strip()
        event_date = from_date_raw[:10] if from_date_raw else ""

        name = str(props.get("name") or props.get("eventname") or props.get("description") or "").strip()
        htmldesc = str(props.get("htmldescription") or "").strip()
        clean_desc = re.sub(r"<[^>]+>", " ", htmldesc).strip()
        if not clean_desc:
            clean_desc = name
        full_desc = clean_desc
        if alert_level and f"gdacs {alert_level.lower()}" not in full_desc.lower():
            full_desc = f"[GDACS {alert_level} Alert] {full_desc}"

        country_val = str(props.get("country") or "").strip()
        if not country_val and isinstance(props.get("affectedcountries"), list):
            ac_names = [
                str(ac.get("countryname") or "").strip()
                for ac in props["affectedcountries"]
                if isinstance(ac, dict) and ac.get("countryname")
            ]
            country_val = ", ".join(n for n in ac_names if n)

        url_obj = props.get("url")
        link = ""
        if isinstance(url_obj, dict):
            link = str(url_obj.get("report") or url_obj.get("details") or "")
        if not link:
            link = f"https://www.gdacs.org/report.aspx?eventtype={raw_etype}&eventid={props.get('eventid', '')}"

        items.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "event_date": event_date,
                "title": name,
                "description": full_desc[:1000],
                "country": country_val,
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

    def _search_historical_api(
        self,
        from_date: str,
        to_date: str,
        country: str | None = None,
        offline: bool = False,
    ) -> list[dict[str, Any]]:
        clean_country = (country or "").strip()
        cache_key = f"gdacs_hist_{from_date}_{to_date}_{clean_country.lower()}"
        cached = cache_get(cache_key)
        if isinstance(cached, list):
            return cached
        if offline:
            return []

        params: dict[str, Any] = {
            "fromDate": from_date,
            "toDate": to_date,
        }
        if clean_country:
            params["country"] = clean_country

        try:
            response = httpx.get(
                GDACS_API_SEARCH_URL,
                params=params,
                timeout=12.0,
                headers={"User-Agent": "DisasterRes-Net/1.0"},
            )
            if response.status_code != 200:
                return []
            data = response.json()
            items = parse_gdacs_geojson(data)
            if items:
                cache_set(cache_key, items, ttl_s=86400)
            return items
        except (httpx.HTTPError, ValueError):
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
        time_hint: str | None = None,
        max_distance_km: float = 300.0,
        limit: int = 5,
        offline: bool = False,
    ) -> list[EventCandidate]:
        bounds = _parse_time_hint_bounds(time_hint)
        current_year = datetime.now(timezone.utc).year

        items: list[dict[str, Any]] = []
        if bounds is not None:
            from_date, to_date, req_year = bounds
            if req_year < current_year:
                # Historical search: exclusively use historical API
                items = self._search_historical_api(from_date, to_date, country=country, offline=offline)
                if not items and country:
                    # Retry without country restriction in case country name didn't match GDACS country spelling
                    items = self._search_historical_api(from_date, to_date, country=None, offline=offline)
            else:
                # Current year: query historical API and combine with live RSS
                items = self._search_historical_api(from_date, to_date, country=country, offline=offline)
                items += self._fetch_items(offline=offline)
        else:
            # No specific year bounds: fetch active alerts from live RSS
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
                    # Closer epicenters score higher (up to +3.5 at 0 km)
                    proximity_bonus = 3.5 * (1.0 - (dist_km / max(1.0, max_distance_km)))
                    match_score += max(0.5, proximity_bonus)
                elif locality and locality.lower() not in item_country and locality.lower() not in haystack:
                    # If user specified a sub-national locality (e.g. Kerala) and coordinates are > max_distance_km
                    # and the text doesn't mention locality, do not match distant events in other parts of the country!
                    continue

            if match_score <= 0.0:
                continue

            if hazard_hint and item.get("event_type") == hazard_hint.lower():
                match_score += 2.0

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
    time_hint: str | None = None,
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
        time_hint=time_hint,
        max_distance_km=max_distance_km,
        limit=limit,
        offline=offline,
    )
