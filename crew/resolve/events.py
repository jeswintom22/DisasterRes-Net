from __future__ import annotations

import inspect
import re
from typing import Any

from crew.sources.reliefweb import ReliefWebSource
from crew.sources.gdacs import GDACSSource
from crew.sources.wikipedia import WikipediaSource
from crew.sources.base import EventCandidate
from .geocode import GeoLocation
from .query import ResolvedQuery


def _extract_coords(geo: GeoLocation | dict[str, Any] | None) -> tuple[float | None, float | None]:
    if geo is None:
        return None, None
    if isinstance(geo, GeoLocation):
        return geo.latitude, geo.longitude
    if isinstance(geo, dict):
        lat = geo.get("lat")
        lon = geo.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    return None, None


def _score_event(
    cand: EventCandidate,
    *,
    locality: str | None = None,
    country: str | None = None,
    hazard_hint: str | None = None,
    time_hint: str | None = None,
) -> float:
    score = 0.0
    text = f"{cand.event_type} {cand.description} {cand.event_date}".lower()
    if locality and locality.lower() in text:
        score += 3.0
    if country and country.lower() in text:
        score += 1.0
    if hazard_hint:
        h_low = hazard_hint.lower()
        if h_low in cand.event_type.lower() or h_low in text:
            score += 2.5
    elif cand.event_type != "disaster":
        # Specific peril inferred on open/discovery query
        score += 1.5
    if time_hint:
        year_m = re.search(r"\b(19\d{2}|20\d{2})\b", time_hint)
        if year_m and year_m.group(1) in text:
            score += 3.0
    if len(cand.description) > 80:
        score += 1.0
    return score


def _is_temporally_compatible(cand: EventCandidate, time_hint: str | None) -> bool:
    if not time_hint:
        return True
    req_y = re.search(r"\b(19\d{2}|20\d{2})\b", time_hint)
    if not req_y:
        return True
    y_req = int(req_y.group(1))

    cand_text = f"{cand.event_date} {cand.description}"
    years_found = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", cand_text)]
    if not years_found:
        return True

    if y_req in years_found:
        return True

    diffs = [abs(y - y_req) for y in years_found]
    if min(diffs) > 0:
        return False

    return True


def _call_source_search(
    search_fn: Any,
    location_str: str,
    **kwargs: Any,
) -> list[EventCandidate]:
    """Call `.search(location_str, ...)` passing only supported kwargs (preserves mock compatibility)."""
    try:
        sig = inspect.signature(search_fn)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        filtered = {
            k: v
            for k, v in kwargs.items()
            if v is not None and (accepts_var_kw or k in sig.parameters)
        }
        return search_fn(location_str, **filtered)
    except (ValueError, TypeError):
        return search_fn(location_str)


def find_events(
    location: str | ResolvedQuery,
    *,
    geo: GeoLocation | dict[str, Any] | None = None,
    hazard_hint: str | None = None,
    time_hint: str | None = None,
    offline: bool = False,
) -> list[EventCandidate]:
    if isinstance(location, ResolvedQuery):
        rq = location
        loc_str = rq.canonical_query or rq.location or rq.locality or rq.country or ""
        locality = rq.locality
        country = rq.country
        eff_hazard = hazard_hint or rq.hazard_hint
        eff_time = time_hint or rq.time_hint
    else:
        loc_str = str(location or "")
        parts = [p.strip() for p in loc_str.split(",") if p.strip()]
        locality = parts[0] if parts else None
        country = parts[-1] if len(parts) >= 2 else None
        eff_hazard = hazard_hint
        eff_time = time_hint

    if isinstance(geo, GeoLocation) and not country and geo.country:
        country = geo.country
    elif isinstance(geo, dict) and not country and geo.get("country"):
        country = str(geo["country"])

    lat, lon = _extract_coords(geo)

    rw_candidates = (
        []
        if offline
        else _call_source_search(
            ReliefWebSource().search,
            loc_str,
            locality=locality,
            country=country,
            hazard_hint=eff_hazard,
            time_hint=eff_time,
        )
    )
    gdacs_candidates = _call_source_search(
        GDACSSource().search,
        loc_str,
        locality=locality,
        country=country,
        lat=lat,
        lon=lon,
        hazard_hint=eff_hazard,
        time_hint=eff_time,
        offline=offline,
    )
    wiki_candidates = (
        []
        if offline
        else _call_source_search(
            WikipediaSource().search,
            loc_str,
            locality=locality,
            country=country,
            hazard_hint=eff_hazard,
            time_hint=eff_time,
            offline=offline,
        )
    )

    combined = rw_candidates + gdacs_candidates + wiki_candidates
    if not combined:
        return []

    valid_candidates = [c for c in combined if _is_temporally_compatible(c, eff_time)]
    if not valid_candidates:
        return []

    scored = [
        (
            _score_event(
                c,
                locality=locality,
                country=country,
                hazard_hint=eff_hazard,
                time_hint=eff_time,
            ),
            -idx,
            c,
        )
        for idx, c in enumerate(valid_candidates)
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    results = [c for _, _, c in scored]
    top = results[0]
    if top.event_type == "disaster":
        for other in results[1:]:
            if other.event_type != "disaster" and (not eff_hazard or other.event_type == eff_hazard.lower()):
                top = EventCandidate(
                    event_id=top.event_id,
                    event_type=other.event_type,
                    event_date=top.event_date or other.event_date,
                    description=top.description,
                    source=top.source,
                )
                results[0] = top
                break

    return results


def resolve_event(
    location: str | ResolvedQuery,
    *,
    geo: GeoLocation | dict[str, Any] | None = None,
    hazard_hint: str | None = None,
    time_hint: str | None = None,
    offline: bool = False,
) -> EventCandidate:
    loc_str = (
        (location.canonical_query or location.location or "")
        if isinstance(location, ResolvedQuery)
        else str(location or "")
    )
    eff_hazard = (
        (hazard_hint or location.hazard_hint)
        if isinstance(location, ResolvedQuery)
        else hazard_hint
    )
    candidates = find_events(
        location,
        geo=geo,
        hazard_hint=eff_hazard,
        time_hint=time_hint,
        offline=offline,
    )
    return candidates[0] if candidates else ReliefWebSource.fallback(loc_str, hazard_hint=eff_hazard)
