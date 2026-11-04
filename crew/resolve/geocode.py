from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import httpx

from crew.cache import get as cache_get, set as cache_set
from .query import ResolvedQuery

_TTL_30_DAYS = 86400 * 30
_HEADERS = {"User-Agent": "DisasterRes-Net/1.0 (academic research)"}


@dataclass(frozen=True)
class GeoLocation:
    display_name: str
    latitude: float
    longitude: float
    bbox: tuple[float, float, float, float] | None = None
    country: str | None = None
    country_code: str | None = None
    admin_area: str | None = None
    importance: float = 0.0
    candidates_evaluated: int = 1
    disambiguation_note: str | None = None
    source: str = "nominatim"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": self.latitude,
            "lon": self.longitude,
            "bbox": list(self.bbox) if self.bbox else None,
            "country_code": (self.country_code or "").upper(),
            "country": self.country,
            "admin_area": self.admin_area,
            "display_name": self.display_name,
            "importance": self.importance,
            "candidates_evaluated": self.candidates_evaluated,
            "disambiguation_note": self.disambiguation_note,
            "source": self.source,
        }


def _score_candidate(
    item: dict[str, Any],
    *,
    admin_hint: str | None = None,
    country_hint: str | None = None,
    country_iso2: str | None = None,
    active_crisis_countries: set[str] | None = None,
) -> float:
    try:
        importance = float(item.get("importance") or 0.5)
    except (TypeError, ValueError):
        importance = 0.5
    importance = max(0.0, min(1.0, importance))

    address = item.get("address") or {}
    cand_iso2 = str(address.get("country_code") or "").lower()
    cand_country = str(address.get("country") or "").lower()
    cand_state = str(
        address.get("state")
        or address.get("region")
        or address.get("province")
        or address.get("county")
        or ""
    ).lower()
    display_lower = str(item.get("display_name") or "").lower()

    admin_match = 0.0
    if country_iso2 and cand_iso2 == country_iso2.lower():
        admin_match += 0.6
    elif country_hint and country_hint.lower() in (cand_country, display_lower):
        admin_match += 0.5

    if admin_hint and (admin_hint.lower() in cand_state or admin_hint.lower() in display_lower):
        admin_match += 0.4
    admin_match = min(1.0, admin_match)

    crisis_match = 0.0
    if active_crisis_countries:
        lower_crisis = {c.lower() for c in active_crisis_countries}
        if cand_iso2 in lower_crisis or (cand_country and cand_country in lower_crisis):
            crisis_match = 1.0

    return 0.45 * importance + 0.30 * admin_match + 0.25 * crisis_match


def _parse_nominatim_candidates(
    data: list[dict[str, Any]],
    *,
    fallback_name: str,
    admin_hint: str | None = None,
    country_hint: str | None = None,
    country_iso2: str | None = None,
    active_crisis_countries: set[str] | None = None,
) -> GeoLocation | None:
    if not data:
        return None

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, item in enumerate(data):
        if "lat" not in item or "lon" not in item:
            continue
        score = _score_candidate(
            item,
            admin_hint=admin_hint,
            country_hint=country_hint,
            country_iso2=country_iso2,
            active_crisis_countries=active_crisis_countries,
        )
        # Preserve original Nominatim order as tie-breaker via -idx
        scored.append((score, -idx, item))

    if not scored:
        return None

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_score, _, best = scored[0]

    bbox_raw = best.get("boundingbox") or []
    bbox: tuple[float, float, float, float] | None = None
    if len(bbox_raw) == 4:
        try:
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        except (TypeError, ValueError):
            bbox = None

    address = best.get("address") or {}
    cc = str(address.get("country_code") or "").upper()
    country_name = address.get("country")
    admin_area = (
        address.get("state")
        or address.get("region")
        or address.get("province")
        or address.get("county")
    )
    display_name = str(best.get("display_name") or fallback_name)
    try:
        importance = float(best.get("importance") or 0.5)
    except (TypeError, ValueError):
        importance = 0.5

    # Check if multiple distinct countries appeared in candidates without an explicit country hint
    distinct_ccs: list[str] = []
    for _, _, cand in scored:
        c_addr = cand.get("address") or {}
        c_code = str(c_addr.get("country_code") or "").upper()
        if c_code and c_code not in distinct_ccs:
            distinct_ccs.append(c_code)

    disambiguation_note: str | None = None
    if len(distinct_ccs) > 1 and not country_iso2 and not country_hint:
        others = ", ".join(distinct_ccs[1:4])
        disambiguation_note = (
            f"Ambiguous toponym '{fallback_name}' resolved to {display_name} ({cc}); "
            f"alternative candidates exist in {others}."
        )

    return GeoLocation(
        display_name=display_name,
        latitude=float(best["lat"]),
        longitude=float(best["lon"]),
        bbox=bbox,
        country=str(country_name) if country_name else None,
        country_code=cc,
        admin_area=str(admin_area) if admin_area else None,
        importance=importance,
        candidates_evaluated=len(scored),
        disambiguation_note=disambiguation_note,
        source="nominatim",
    )


def _fetch_nominatim(query_str: str, country_iso2: str | None = None) -> list[dict[str, Any]]:
    url = "https://nominatim.openstreetmap.org/search"
    params: dict[str, Any] = {
        "format": "jsonv2",
        "q": query_str,
        "limit": 5,
        "addressdetails": 1,
    }
    if country_iso2 and len(country_iso2.strip()) == 2:
        params["countrycodes"] = country_iso2.strip().lower()

    response = httpx.get(url, params=params, headers=_HEADERS, timeout=10.0)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def geocode_location(
    place: str,
    *,
    admin_hint: str | None = None,
    country_hint: str | None = None,
    country_iso2: str | None = None,
    active_crisis_countries: set[str] | None = None,
    offline: bool = False,
) -> GeoLocation | None:
    """Multi-candidate Nominatim geocoder with hierarchical retry and crisis disambiguation."""
    if not place or not place.strip():
        return None

    clean_place = place.strip()
    cache_key = f"geocode:{clean_place.lower()}"
    if country_iso2:
        cache_key += f":{country_iso2.lower()}"

    cached = cache_get(cache_key)
    if isinstance(cached, dict) and "lat" in cached and "lon" in cached:
        bbox_list = cached.get("bbox")
        bbox_tuple = tuple(float(x) for x in bbox_list) if isinstance(bbox_list, list) and len(bbox_list) == 4 else None
        return GeoLocation(
            display_name=str(cached.get("display_name") or clean_place),
            latitude=float(cached["lat"]),
            longitude=float(cached["lon"]),
            bbox=bbox_tuple,  # type: ignore[arg-type]
            country=cached.get("country"),
            country_code=str(cached.get("country_code") or "").upper(),
            admin_area=cached.get("admin_area"),
            importance=float(cached.get("importance") or 0.0),
            candidates_evaluated=int(cached.get("candidates_evaluated") or 1),
            disambiguation_note=cached.get("disambiguation_note"),
            source=str(cached.get("source") or "nominatim"),
        )

    if offline:
        return None

    try:
        data = _fetch_nominatim(clean_place, country_iso2=country_iso2)
        # EC-8: If hierarchical canonical_query returns 0 results, retry with simplified locality/country
        if not data and "," in clean_place:
            parts = [p.strip() for p in clean_place.split(",") if p.strip()]
            if len(parts) >= 2:
                simplified = f"{parts[0]}, {parts[-1]}"
                if simplified.lower() != clean_place.lower():
                    data = _fetch_nominatim(simplified, country_iso2=country_iso2)
            if not data and parts:
                data = _fetch_nominatim(parts[0], country_iso2=None)

        geo_loc = _parse_nominatim_candidates(
            data,
            fallback_name=clean_place,
            admin_hint=admin_hint,
            country_hint=country_hint,
            country_iso2=country_iso2,
            active_crisis_countries=active_crisis_countries,
        )
        if geo_loc is None:
            return None

        cache_set(cache_key, geo_loc.to_dict(), ttl_s=_TTL_30_DAYS)
        return geo_loc
    except Exception:
        return None


def geocode_resolved_query(
    rq: ResolvedQuery,
    *,
    active_crisis_countries: set[str] | None = None,
    offline: bool = False,
) -> GeoLocation | None:
    """Geocode an expanded ResolvedQuery using its locality, admin_area, and country_iso2."""
    target = rq.canonical_query or rq.location or rq.locality or rq.country
    if not target:
        return None
    return geocode_location(
        target,
        admin_hint=rq.admin_area,
        country_hint=rq.country,
        country_iso2=rq.country_iso2,
        active_crisis_countries=active_crisis_countries,
        offline=offline,
    )


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    offline: bool = False,
) -> GeoLocation | None:
    """Reverse-geocode WGS84 (lat, lon) coordinates via OpenStreetMap Nominatim."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    cache_key = f"rev_geocode:{lat:.4f},{lon:.4f}"
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and "lat" in cached and "lon" in cached:
        bbox_list = cached.get("bbox")
        bbox_tuple = tuple(float(x) for x in bbox_list) if isinstance(bbox_list, list) and len(bbox_list) == 4 else None
        return GeoLocation(
            display_name=str(cached.get("display_name") or f"{lat:.4f}, {lon:.4f}"),
            latitude=float(cached["lat"]),
            longitude=float(cached["lon"]),
            bbox=bbox_tuple,  # type: ignore[arg-type]
            country=cached.get("country"),
            country_code=str(cached.get("country_code") or "").upper(),
            admin_area=cached.get("admin_area"),
            importance=float(cached.get("importance") or 0.5),
            candidates_evaluated=1,
            source="exif_gps",
        )

    default_bbox = (lat - 0.05, lat + 0.05, lon - 0.05, lon + 0.05)
    if offline:
        return GeoLocation(
            display_name=f"GPS ({lat:.4f}, {lon:.4f})",
            latitude=lat,
            longitude=lon,
            bbox=default_bbox,
            source="exif_gps",
        )

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "zoom": 10,
        "addressdetails": 1,
    }
    try:
        response = httpx.get(url, params=params, headers=_HEADERS, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        address = data.get("address") or {}
        locality = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state_district")
        )
        admin_area = address.get("state") or address.get("region") or address.get("province")
        country = address.get("country")
        cc = str(address.get("country_code") or "").upper()
        parts = [p for p in (locality, admin_area, country) if p]
        display_name = ", ".join(parts) if parts else str(data.get("display_name") or f"GPS ({lat:.4f}, {lon:.4f})")

        bbox_raw = data.get("boundingbox") or []
        bbox = (
            (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
            if len(bbox_raw) == 4
            else default_bbox
        )
        geo_loc = GeoLocation(
            display_name=display_name,
            latitude=lat,
            longitude=lon,
            bbox=bbox,
            country=str(country) if country else None,
            country_code=cc,
            admin_area=str(admin_area) if admin_area else None,
            importance=0.8,
            candidates_evaluated=1,
            source="exif_gps",
        )
        cache_set(cache_key, geo_loc.to_dict(), ttl_s=_TTL_30_DAYS)
        return geo_loc
    except Exception:
        return GeoLocation(
            display_name=f"GPS ({lat:.4f}, {lon:.4f})",
            latitude=lat,
            longitude=lon,
            bbox=default_bbox,
            source="exif_gps",
        )


def geocode(
    location: str,
    *,
    country_iso2: str | None = None,
    admin_hint: str | None = None,
    country_hint: str | None = None,
    active_crisis_countries: set[str] | None = None,
) -> dict[str, Any] | None:
    """
    Geocodes a location string using Nominatim (multi-candidate disambiguation).
    Returns a dict with lat, lon, bbox, country_code, and metadata, or None if not found.
    Bbox format: [latMin, latMax, lonMin, lonMax]
    """
    geo_loc = geocode_location(
        location,
        admin_hint=admin_hint,
        country_hint=country_hint,
        country_iso2=country_iso2,
        active_crisis_countries=active_crisis_countries,
    )
    return geo_loc.to_dict() if geo_loc is not None else None
