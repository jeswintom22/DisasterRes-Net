from __future__ import annotations

from datetime import date
import re
import httpx
from .base import EventCandidate, ProvenanceRef


def _decompose_location(
    location: str,
    locality: str | None = None,
    country: str | None = None,
) -> tuple[str, str | None, str | None]:
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    eff_locality = locality or (parts[0] if parts else location.strip())
    eff_country = country or (parts[-1] if len(parts) >= 2 else None)
    return location.strip(), eff_locality, eff_country


def _rank_candidates(
    candidates: list[EventCandidate],
    *,
    locality: str | None = None,
    country: str | None = None,
    hazard_hint: str | None = None,
    time_hint: str | None = None,
) -> list[EventCandidate]:
    if not candidates:
        return []

    def _score(cand: EventCandidate) -> float:
        score = 0.0
        desc_lower = f"{cand.event_type} {cand.description} {cand.event_date}".lower()
        if locality and locality.lower() in desc_lower:
            score += 2.0
        if country and country.lower() in desc_lower:
            score += 1.0
        if hazard_hint:
            h_low = hazard_hint.lower()
            if h_low in cand.event_type.lower() or h_low in desc_lower:
                score += 1.5
        if time_hint:
            year_match = re.search(r"\b(20\d{2})\b", time_hint)
            if year_match and year_match.group(1) in desc_lower:
                score += 1.5
        return score

    # Sort descending by score while preserving original relative order on ties
    indexed = [( _score(c), -idx, c ) for idx, c in enumerate(candidates)]
    indexed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [c for _, _, c in indexed]


class ReliefWebSource:
    url = "https://api.reliefweb.int/v1/disasters"
    reports_url = "https://api.reliefweb.int/v1/reports"

    def search(
        self,
        location: str,
        *,
        locality: str | None = None,
        country: str | None = None,
        hazard_hint: str | None = None,
        time_hint: str | None = None,
    ) -> list[EventCandidate]:
        full_loc, eff_locality, eff_country = _decompose_location(location, locality, country)
        results: list[EventCandidate] = []

        # Build query strings from most specific (locality) to broader (country) so national
        # declarations like "India: Floods and Landslides - Jul 2024" are not missed when
        # full_loc is "Wayanad, Kerala, India".
        query_terms: list[str] = []
        for term in (eff_locality, eff_country, full_loc):
            if term and term not in query_terms:
                query_terms.append(term)

        seen_ids: set[str] = set()
        for q_term in query_terms:
            try:
                payload = {
                    "query": {"value": q_term, "fields": ["name", "country.name"]},
                    "limit": 5,
                    "preset": "latest",
                }
                response = httpx.post(
                    self.url,
                    json=payload,
                    timeout=8,
                    headers={"User-Agent": "DisasterRes-Net/1.0"},
                )
                response.raise_for_status()
                for row in response.json().get("data", []):
                    row_id = str(row.get("id"))
                    if row_id in seen_ids:
                        continue
                    seen_ids.add(row_id)
                    fields = row.get("fields", {})
                    types = fields.get("type", [])
                    event_type = (
                        types[0].get("name")
                        if types and isinstance(types[0], dict)
                        else "disaster"
                    )
                    results.append(
                        EventCandidate(
                            row_id,
                             str(event_type).lower(),
                            str(fields.get("date", {}).get("created", ""))[:10],
                            str(fields.get("name", ""))[:1000],
                            ProvenanceRef(
                                "ReliefWeb",
                                f"https://reliefweb.int/disaster/{row_id}",
                            ),
                        )
                    )
                if results:
                    break
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue

        if not results:
            results = self._search_reports(
                full_loc,
                locality=eff_locality,
                country=eff_country,
            )

        return _rank_candidates(
            results,
            locality=eff_locality,
            country=eff_country,
            hazard_hint=hazard_hint,
            time_hint=time_hint,
        )

    def _search_reports(
        self,
        location: str,
        *,
        locality: str | None = None,
        country: str | None = None,
    ) -> list[EventCandidate]:
        """Reports mention district-level locations that disaster titles often omit."""
        search_terms = [t for t in (locality, location, country) if t]
        for term in search_terms:
            try:
                payload = {
                    "query": {"value": term},
                    "fields": {"include": ["title", "body", "date", "disaster", "url"]},
                    "limit": 3,
                    "sort": ["date:desc"],
                }
                response = httpx.post(
                    self.reports_url,
                    json=payload,
                    timeout=10,
                    headers={"User-Agent": "DisasterRes-Net/1.0"},
                )
                response.raise_for_status()
                candidates: list[EventCandidate] = []
                for row in response.json().get("data", []):
                    fields = row.get("fields", {})
                    title = str(fields.get("title", ""))
                    body = re.sub(r"<[^>]+>", " ", str(fields.get("body", "")))
                    description = " ".join((title + ". " + body).split())[:1200]
                    disaster = fields.get("disaster") or []
                    event_type = "disaster"
                    if disaster and isinstance(disaster[0], dict):
                        event_type = str(disaster[0].get("name", event_type)).lower()
                    candidates.append(
                        EventCandidate(
                            str(row.get("id")),
                            event_type,
                            str((fields.get("date") or {}).get("created", ""))[:10],
                            description,
                            ProvenanceRef(
                                "ReliefWeb report",
                                str(
                                    fields.get("url")
                                    or f"https://reliefweb.int/report/{row.get('id')}"
                                ),
                            ),
                        )
                    )
                if candidates:
                    return candidates
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
        return []

    @staticmethod
    def fallback(location: str, hazard_hint: str | None = None) -> EventCandidate:
        ev_type = (hazard_hint or "disaster").lower()
        return EventCandidate(
            f"local-{location.lower().replace(' ', '-')}",
            ev_type,
            date.today().isoformat(),
            (
                f"No authoritative event record was available for {location}. "
                "This report is a location-level investigation, not a verified event bulletin."
            ),
            ProvenanceRef("Local fallback", ""),
        )
