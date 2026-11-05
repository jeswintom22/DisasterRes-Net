from __future__ import annotations

import re
import urllib.parse
from typing import Any
import httpx

from crew.cache import get as cache_get, set as cache_set
from .base import EventCandidate, ProvenanceRef

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
USER_AGENT = "DisasterResNet/1.0 (academic research; contact@disasterres.org)"

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

HAZARD_KEYWORDS = [
    ("landslide", ["landslide", "mudslide", "debris flow", "rockfall", "slope failure"]),
    ("flood", ["flood", "flooding", "inundation", "deluge", "flash flood"]),
    ("earthquake", ["earthquake", "quake", "tremor", "seismic"]),
    ("cyclone", ["cyclone", "typhoon", "hurricane", "tropical storm"]),
    ("wildfire", ["wildfire", "bushfire", "forest fire", "brush fire"]),
    ("tsunami", ["tsunami", "tidal wave"]),
    ("drought", ["drought"]),
]

NON_DISASTER_EXCLUSIONS = re.compile(
    r"\b(?:film|movie|song|album|season|election|championship|cup|tournament|cricket|football|fc|premier league|player|politician|minister)\b",
    re.I,
)


def _infer_hazard(text: str) -> str:
    text_low = text.lower()
    for canonical, keywords in HAZARD_KEYWORDS:
        if any(kw in text_low for kw in keywords):
            return canonical
    return "disaster"


def _extract_date_from_text(text: str) -> str:
    # 1. ISO YYYY-MM-DD
    m_iso = re.search(r"\b(19\d{2}|20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if m_iso:
        return f"{int(m_iso.group(1)):04d}-{int(m_iso.group(2)):02d}-{int(m_iso.group(3)):02d}"

    # 2. "16 August 2018"
    m_dmy = re.search(r"\b(\d{1,2})\s+([a-zA-Z]+)\s+(19\d{2}|20\d{2})\b", text)
    if m_dmy:
        day = int(m_dmy.group(1))
        m_name = m_dmy.group(2).lower()
        year = int(m_dmy.group(3))
        if m_name in MONTH_MAP:
            return f"{year:04d}-{MONTH_MAP[m_name]:02d}-{day:02d}"

    # 3. "August 16, 2018"
    m_mdy = re.search(r"\b([a-zA-Z]+)\s+(\d{1,2}),?\s+(19\d{2}|20\d{2})\b", text)
    if m_mdy:
        m_name = m_mdy.group(1).lower()
        day = int(m_mdy.group(2))
        year = int(m_mdy.group(3))
        if m_name in MONTH_MAP:
            return f"{year:04d}-{MONTH_MAP[m_name]:02d}-{day:02d}"

    # 4. "August 2018"
    m_my = re.search(r"\b([a-zA-Z]+)\s+(19\d{2}|20\d{2})\b", text)
    if m_my:
        m_name = m_my.group(1).lower()
        year = int(m_my.group(2))
        if m_name in MONTH_MAP:
            return f"{year:04d}-{MONTH_MAP[m_name]:02d}-01"

    # 5. Just year
    m_y = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if m_y:
        return m_y.group(1)

    return ""


class WikipediaSource:
    """Keyless Wikipedia / MediaWiki Live Event Search & Summary extractor."""

    def search(
        self,
        location: str,
        *,
        locality: str | None = None,
        country: str | None = None,
        hazard_hint: str | None = None,
        time_hint: str | None = None,
        limit: int = 3,
        offline: bool = False,
    ) -> list[EventCandidate]:
        if offline:
            return []

        clean_loc = (locality or location.split(",")[0] or location).strip()
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", time_hint or "")
        year_str = year_match.group(1) if year_match else ""

        cache_key = f"wiki_search_{clean_loc}_{year_str}_{hazard_hint or ''}"
        cached = cache_get(cache_key)
        if isinstance(cached, list):
            return [
                EventCandidate(
                    event_id=item["event_id"],
                    event_type=item["event_type"],
                    event_date=item["event_date"],
                    description=item["description"],
                    source=ProvenanceRef("Wikipedia", item["url"]),
                )
                for item in cached
            ]

        # Construct candidate search queries from most specific to broad
        queries = []
        if clean_loc and year_str:
            queries.append(f"intitle:{clean_loc} intitle:{year_str}")
            queries.append(f"{clean_loc} {year_str} flood OR landslide OR earthquake OR cyclone OR wildfire OR disaster")
        elif clean_loc:
            queries.append(f"{clean_loc} {hazard_hint or 'disaster'}")

        matching_titles: list[str] = []
        seen_titles: set[str] = set()

        for q in queries:
            try:
                res = httpx.get(
                    WIKI_API_URL,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": q,
                        "format": "json",
                        "srlimit": 5,
                    },
                    headers={"User-Agent": USER_AGENT},
                    timeout=8.0,
                )
                if res.status_code != 200:
                    continue
                hits = res.json().get("query", {}).get("search", [])
                for h in hits:
                    title = str(h.get("title") or "").strip()
                    snippet = str(h.get("snippet") or "").strip()
                    comb = f"{title} {snippet}".lower()

                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # Check for non-disaster exclusions unless hazard explicitly in title
                    has_hazard_in_title = any(kw in title.lower() for _, kws in HAZARD_KEYWORDS for kw in kws)
                    if NON_DISASTER_EXCLUSIONS.search(title) and not has_hazard_in_title:
                        continue

                    # Require at least one disaster or peril keyword
                    has_disaster_kw = (
                        has_hazard_in_title
                        or any(kw in comb for _, kws in HAZARD_KEYWORDS for kw in kws)
                        or "disaster" in comb
                        or "calamity" in comb
                    )
                    if not has_disaster_kw:
                        continue

                    matching_titles.append(title)
                if matching_titles:
                    break
            except (httpx.HTTPError, ValueError):
                continue

        results: list[EventCandidate] = []
        cache_items: list[dict[str, str]] = []

        for title in matching_titles[:limit]:
            try:
                safe_title = urllib.parse.quote(title.replace(" ", "_"))
                summary_res = httpx.get(
                    f"{WIKI_SUMMARY_URL}/{safe_title}",
                    headers={"User-Agent": USER_AGENT},
                    timeout=8.0,
                )
                if summary_res.status_code != 200:
                    continue
                sdata = summary_res.json()
                extract = str(sdata.get("extract") or "").strip()
                page_url = (
                    sdata.get("content_urls", {}).get("desktop", {}).get("page")
                    or f"https://en.wikipedia.org/wiki/{safe_title}"
                )
                page_id = str(sdata.get("pageid") or abs(hash(title)) % 100000)

                combined_text = f"{title}. {extract}"
                inferred_type = _infer_hazard(combined_text)
                if hazard_hint and hazard_hint.lower() in combined_text.lower():
                    inferred_type = hazard_hint.lower()

                event_date = _extract_date_from_text(combined_text)
                if not event_date and year_str:
                    event_date = year_str

                event_id = f"wiki-{inferred_type}-{page_id}"
                desc = f"{title}: {extract}"[:1200]

                cand = EventCandidate(
                    event_id=event_id,
                    event_type=inferred_type,
                    event_date=event_date,
                    description=desc,
                    source=ProvenanceRef("Wikipedia", page_url),
                )
                results.append(cand)
                cache_items.append({
                    "event_id": event_id,
                    "event_type": inferred_type,
                    "event_date": event_date,
                    "description": desc,
                    "url": page_url,
                })
            except (httpx.HTTPError, ValueError):
                continue

        if cache_items:
            cache_set(cache_key, cache_items, ttl_s=86400 * 7)

        return results

