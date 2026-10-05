from __future__ import annotations
from datetime import date
import re
import httpx
from .base import EventCandidate, ProvenanceRef

class ReliefWebSource:
    url = "https://api.reliefweb.int/v1/disasters"
    reports_url = "https://api.reliefweb.int/v1/reports"
    def search(self, location: str) -> list[EventCandidate]:
        try:
            payload = {"query": {"value": location, "fields": ["name"]}, "limit": 5, "preset": "latest"}
            response = httpx.post(self.url, json=payload, timeout=8, headers={"User-Agent": "DisasterRes-Net/1.0"})
            response.raise_for_status()
            results = []
            for row in response.json().get("data", []):
                fields = row.get("fields", {})
                types = fields.get("type", [])
                event_type = (types[0].get("name") if types and isinstance(types[0], dict) else "disaster")
                results.append(EventCandidate(str(row.get("id")), event_type.lower(), str(fields.get("date", {}).get("created", ""))[:10], str(fields.get("name", ""))[:1000], ProvenanceRef("ReliefWeb", f"https://reliefweb.int/disaster/{row.get('id') }")))
            return results or self._search_reports(location)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return self._search_reports(location)

    def _search_reports(self, location: str) -> list[EventCandidate]:
        """Reports mention district-level locations that disaster titles often omit."""
        try:
            payload = {
                "query": {"value": location},
                "fields": {"include": ["title", "body", "date", "disaster", "url"]},
                "limit": 3,
                "sort": ["date:desc"],
            }
            response = httpx.post(self.reports_url, json=payload, timeout=10, headers={"User-Agent": "DisasterRes-Net/1.0"})
            response.raise_for_status()
            candidates = []
            for row in response.json().get("data", []):
                fields = row.get("fields", {})
                title = str(fields.get("title", ""))
                body = re.sub(r"<[^>]+>", " ", str(fields.get("body", "")))
                description = " ".join((title + ". " + body).split())[:1200]
                disaster = fields.get("disaster") or []
                event_type = "disaster"
                if disaster and isinstance(disaster[0], dict):
                    event_type = str(disaster[0].get("name", event_type)).lower()
                candidates.append(EventCandidate(str(row.get("id")), event_type, str((fields.get("date") or {}).get("created", ""))[:10], description, ProvenanceRef("ReliefWeb report", str(fields.get("url") or f"https://reliefweb.int/report/{row.get('id')}"))))
            return candidates
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    @staticmethod
    def fallback(location: str) -> EventCandidate:
        return EventCandidate(f"local-{location.lower().replace(' ', '-')}", "disaster", date.today().isoformat(), f"No authoritative event record was available for {location}. This report is a location-level investigation, not a verified event bulletin.", ProvenanceRef("Local fallback", ""))
