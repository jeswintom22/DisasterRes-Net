from __future__ import annotations

import os
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
    url = "https://api.reliefweb.int/v2/disasters"
    reports_url = "https://api.reliefweb.int/v2/reports"

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

        query_terms: list[str] = []
        for term in (eff_locality, eff_country, full_loc):
            if term and term not in query_terms:
                query_terms.append(term)

        appname = os.getenv("RELIEFWEB_APPNAME", "DisasterRes-Net")
        seen_ids: set[str] = set()
        for q_term in query_terms:
            try:
                full_q = f"{q_term} {time_hint}".strip() if time_hint else q_term
                payload = {
                    "query": {"value": full_q, "fields": ["name", "country.name"]},
                    "limit": 5,
                    "preset": "latest",
                }
                api_url = f"{self.url}?appname={appname}" if appname else self.url
                response = httpx.post(
                    api_url,
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
                time_hint=time_hint,
            )

        if not results:
            results = self._search_via_web_api(
                full_loc,
                locality=eff_locality,
                country=eff_country,
                hazard_hint=hazard_hint,
                time_hint=time_hint,
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
        time_hint: str | None = None,
    ) -> list[EventCandidate]:
        """Reports mention district-level locations that disaster titles often omit."""
        search_terms = [t for t in (locality, location, country) if t]
        appname = os.getenv("RELIEFWEB_APPNAME", "DisasterRes-Net")
        api_url = f"{self.reports_url}?appname={appname}" if appname else self.reports_url

        for term in search_terms:
            try:
                full_q = f"{term} {time_hint}".strip() if time_hint else term
                payload = {
                    "query": {"value": full_q},
                    "fields": {"include": ["title", "body", "date", "disaster", "url"]},
                    "limit": 3,
                    "sort": ["date:desc"],
                }
                response = httpx.post(
                    api_url,
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

    def _search_via_web_api(
        self,
        location: str,
        *,
        locality: str | None = None,
        country: str | None = None,
        hazard_hint: str | None = None,
        time_hint: str | None = None,
    ) -> list[EventCandidate]:
        clean_loc = locality or location
        q_parts = ["site:reliefweb.int/disaster OR site:reliefweb.int/report", clean_loc]
        if hazard_hint:
            q_parts.append(hazard_hint)
        else:
            q_parts.append("disaster")
        if time_hint:
            q_parts.append(time_hint)
        query_str = " ".join(q_parts)

        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            try:
                res = httpx.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": query_str, "num": 5},
                    timeout=8.0,
                )
                if res.status_code == 200:
                    candidates = []
                    for item in res.json().get("organic", []):
                        title = str(item.get("title") or "").strip()
                        link = str(item.get("link") or "").strip()
                        snippet = str(item.get("snippet") or "").strip()
                        desc = f"{title}. {snippet}"[:1200]
                        etype = "disaster"
                        for c_type in ("flood", "landslide", "earthquake", "cyclone", "wildfire"):
                            if c_type in desc.lower():
                                etype = c_type
                                break
                        row_id_m = re.search(r"/(?:disaster|report)/([\w-]+)", link)
                        row_id = row_id_m.group(1) if row_id_m else f"rw-web-{abs(hash(link)) % 100000}"
                        ev_date = ""
                        date_str = item.get("date")
                        if date_str:
                            ev_date = str(date_str)
                        if not ev_date:
                            from .wikipedia import _extract_date_from_text
                            ev_date = _extract_date_from_text(desc)
                        if not ev_date and time_hint:
                            y_m = re.search(r"\b(19\d{2}|20\d{2})\b", time_hint)
                            if y_m:
                                ev_date = y_m.group(1)
                        candidates.append(
                            EventCandidate(
                                row_id,
                                etype,
                                ev_date,
                                desc,
                                ProvenanceRef("ReliefWeb", link),
                            )
                        )
                    if candidates:
                        return candidates
            except Exception:
                pass
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
