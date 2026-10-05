from __future__ import annotations
import httpx
from .base import EventCandidate, ProvenanceRef

class GDACSSource:
    url = "https://www.gdacs.org/xml/rss.xml"
    def search(self, location: str) -> list[EventCandidate]:
        # GDACS has no location-text search endpoint; retain a resilient adapter seam.
        try:
            response = httpx.get(self.url, timeout=8, headers={"User-Agent": "DisasterRes-Net/1.0"})
            return [] if response.status_code == 200 else []
        except httpx.HTTPError:
            return []
