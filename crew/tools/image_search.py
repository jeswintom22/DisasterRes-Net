"""Open-source image search adapter point. GDELT is preferred over commercial scraping."""
from __future__ import annotations
import os
import httpx

class GoogleImageScrapeTool:
    name = "image_search"
    def run(self, location: str, event_type: str) -> list[str]:
        # Kept compatible with the original sprint. It only activates with an explicitly supplied key.
        key = os.getenv("SERPAPI_API_KEY")
        if not key:
            return self._gdelt_images(location, event_type)
        try:
            response = httpx.get("https://serpapi.com/search.json", params={"engine": "google_images", "q": f"{location} {event_type} aerial damage satellite", "api_key": key}, timeout=10)
            return [row["original"] for row in response.json().get("images_results", [])[:5] if row.get("original")]
        except (httpx.HTTPError, ValueError, KeyError):
            return self._gdelt_images(location, event_type)

    @staticmethod
    def _gdelt_images(location: str, event_type: str) -> list[str]:
        """Free, keyless fallback using GDELT news-image metadata."""
        try:
            response = httpx.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": f'"{location}" {event_type} disaster', "mode": "artlist", "maxrecords": 10, "format": "json", "sort": "HybridRel"},
                timeout=12,
                headers={"User-Agent": "DisasterRes-Net/1.0"},
            )
            response.raise_for_status()
            rows = response.json().get("articles", [])
            return list(dict.fromkeys(row.get("socialimage") for row in rows if row.get("socialimage")))[:5]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []
