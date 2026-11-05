"""Open-source image search adapter point. GDELT is preferred over commercial scraping."""
from __future__ import annotations
import os
import httpx

class GoogleImageScrapeTool:
    name = "image_search"
    def run(
        self,
        location: str,
        event_type: str,
        satellite_mode: bool = False,
        year: str | None = None,
    ) -> list[str]:
        # Only append 'aerial damage satellite' if explicitly running in satellite sweep mode
        parts = [location, event_type]
        if year and str(year).strip():
            y = str(year).strip()
            if y not in location:
                parts.append(y)
        if satellite_mode:
            parts.append("aerial damage satellite")
        else:
            parts.append("damage disaster")

        q = " ".join(p for p in parts if p)
        all_urls = []

        # 1. Google Custom Search
        gcs_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        gcs_cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
        if gcs_key and gcs_cx:
            all_urls.extend(self._google_custom_search_images(q, gcs_key, gcs_cx))

        # 2. DataForSEO
        dfs_login = os.getenv("DATAFORSEO_LOGIN")
        dfs_password = os.getenv("DATAFORSEO_PASSWORD")
        dfs_key = os.getenv("DATAFORSEO_API_KEY")
        if dfs_login and dfs_password:
            import base64
            auth_token = base64.b64encode(f"{dfs_login}:{dfs_password}".encode()).decode("utf-8")
            all_urls.extend(self._dataforseo_images(q, auth_token))
        elif dfs_key:
            all_urls.extend(self._dataforseo_images(q, dfs_key))

        # 3. Serper.dev
        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            all_urls.extend(self._serper_images(q, serper_key))

        # 4. SerpAPI
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            all_urls.extend(self._serpapi_images(q, serpapi_key))

        # 5. Fallback (Wikimedia / GDELT) if we still have too few
        if len(all_urls) < 3:
            all_urls.extend(self._fallback_images(location, event_type))

        # Deduplicate while preserving order
        unique_urls = list(dict.fromkeys(all_urls))
        return unique_urls[:75] # Return up to 75 unique images total

    @staticmethod
    def _google_custom_search_images(q: str, key: str, cx: str) -> list[str]:
        try:
            print(f"\n[DEBUG] [GCS API] GET {q}", flush=True)
            res = httpx.get("https://customsearch.googleapis.com/customsearch/v1", params={"key": key, "cx": cx, "q": q, "searchType": "image", "num": 10}, timeout=10)
            res.raise_for_status()
            images = [item["link"] for item in res.json().get("items", [])[:10] if item.get("link")]
            print(f"[DEBUG] [GCS API] Found {len(images)} images.", flush=True)
            return images
        except Exception as e:
            print(f"[DEBUG] [GCS API] Failed: {e}", flush=True)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"[DEBUG] [GCS API] Response: {e.response.text}", flush=True)
            return []

    @staticmethod
    def _dataforseo_images(q: str, auth: str) -> list[str]:
        try:
            url = "https://api.dataforseo.com/v3/serp/google/images/live/advanced"
            payload = [{"keyword": q, "language_code": "en", "location_code": 2840, "depth": 50}]
            res = httpx.post(url, headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"}, json=payload, timeout=15)
            res.raise_for_status()
            data = res.json()
            tasks = data.get("tasks", [])
            images = []
            if tasks and "result" in tasks[0] and tasks[0]["result"]:
                items = tasks[0]["result"][0].get("items", [])
                for item in items:
                    if "source_url" in item:
                        images.append(item["source_url"])
            return images
        except Exception:
            return []

    @staticmethod
    def _serper_images(q: str, key: str) -> list[str]:
        try:
            res = httpx.post("https://google.serper.dev/images", headers={"X-API-KEY": key, "Content-Type": "application/json"}, json={"q": q, "num": 10}, timeout=10)
            res.raise_for_status()
            return [item["imageUrl"] for item in res.json().get("images", []) if item.get("imageUrl")]
        except Exception:
            return []

    @staticmethod
    def _serpapi_images(q: str, key: str) -> list[str]:
        try:
            res = httpx.get("https://serpapi.com/search.json", params={"engine": "google_images", "q": q, "api_key": key, "tbm": "isch"}, timeout=10)
            res.raise_for_status()
            return [item["original"] for item in res.json().get("images_results", []) if item.get("original")]
        except Exception:
            return []

    @classmethod
    def _fallback_images(cls, location: str, event_type: str) -> list[str]:
        urls = cls._wikimedia_images(location, event_type)
        if not urls:
            urls = cls._gdelt_images(location, event_type)
        return urls

    @staticmethod
    def _wikimedia_images(location: str, event_type: str) -> list[str]:
        """Highly reliable free image search using Wikimedia Commons."""
        try:
            response = httpx.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{location} {event_type}",
                    "gsrnamespace": "6",
                    "gsrlimit": "5",
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "format": "json"
                },
                headers={"User-Agent": "DisasterRes-Net/1.0 (academic research)"},
                timeout=12
            )
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            urls = []
            for page in pages.values():
                imageinfo = page.get("imageinfo", [])
                if imageinfo:
                    urls.append(imageinfo[0].get("url"))
            return [url for url in urls if url][:5]
        except Exception:
            return []

    @staticmethod
    def _gdelt_images(location: str, event_type: str) -> list[str]:
        """Free, keyless fallback using GDELT news-image metadata."""
        try:
            query = f"{location} {event_type}"
            response = httpx.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": query, "mode": "artlist", "maxrecords": "5", "format": "json"},
                headers={"User-Agent": "DisasterRes-Net/1.0 (academic research)"},
                timeout=12
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])
            images = [art.get("socialimage") for art in articles if art.get("socialimage")]
            return images[:5]
        except Exception:
            return []
