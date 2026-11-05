# Implementation Notes: Dynamic Historical & Multi-Source Disaster Event Resolution (`005-historical-disaster-resolution`)

## 1. Codebase Touchpoint Matrix

| File | Action | Purpose |
| :--- | :--- | :--- |
| [`crew/sources/gdacs.py`](file:///z:/DisasterRes-Net/crew/sources/gdacs.py) | Modify | Add `_search_historical_api()` calling `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`; parse GeoJSON features; isolate live 7-day RSS feed when `time_hint` contains a past year; sort matched events by haversine distance. |
| [`crew/sources/wikipedia.py`](file:///z:/DisasterRes-Net/crew/sources/wikipedia.py) | Create | Implement `WikipediaSource.search()` using MediaWiki search API (`/w/api.php`) and REST summary API (`/api/rest_v1/page/summary/{title}`) with non-disaster filtering and canonical peril inference. |
| [`crew/sources/reliefweb.py`](file:///z:/DisasterRes-Net/crew/sources/reliefweb.py) | Modify | Upgrade endpoints from `/v1/` to `/v2/` with `?appname=<RELIEFWEB_APPNAME>`, pass `time_hint`, and add `_search_via_web_api()` using `SERPER_API_KEY` / `GOOGLE_CUSTOM_SEARCH_API_KEY` / `SERPAPI_API_KEY` (`site:reliefweb.int`) when direct API returns `403`/`410`. |
| [`crew/sources/__init__.py`](file:///z:/DisasterRes-Net/crew/sources/__init__.py) | Modify | Export `WikipediaSource`. |
| [`crew/resolve/events.py`](file:///z:/DisasterRes-Net/crew/resolve/events.py) | Modify | Integrate `WikipediaSource`, pass `time_hint` to `GDACSSource`, enforce `_is_temporally_compatible()` strict temporal disqualification, infer specific hazards on open discovery queries, and enrich complementary descriptions. |
| [`crew/report/facts.py`](file:///z:/DisasterRes-Net/crew/report/facts.py) | Modify | Update `FactsBlock.values()` to extract numeric tokens (`r"\b\d+(?:\.\d+)?\b"`) embedded in `scout.event_description`, `scout.event_date`, and `analysis.caveats`. |
| [`crew/crew.py`](file:///z:/DisasterRes-Net/crew/crew.py) | Modify | Skip `GoogleImageScrapeTool` when an explicit `time_hint` yields no temporally matching event (`event.event_id == "rw-fallback"` or `temporal_notice` is active). |
| `tests/test_historical_resolution.py` | Create | Unit and integration tests covering `AC-1`–`AC-6` and `EC-1`–`EC-5`. |

---

## 2. GDACS Historical REST API Contract

- **Endpoint:** `GET https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`
- **Query Parameters:**
  - `fromDate`: `YYYY-MM-DD` (e.g., `2018-01-01`)
  - `toDate`: `YYYY-MM-DD` (e.g., `2018-12-31`)
  - `country`: Country name (e.g., `India`)
- **GeoJSON Feature Mapping:**
  - `geometry.coordinates`: `[lon, lat]` (Note: GeoJSON is `[longitude, latitude]`, so `lat = coords[1]`, `lon = coords[0]`).
  - `properties.eventtype`: `"FL"` (`flood`), `"EQ"` (`earthquake`), `"TC"` (`cyclone`), `"VO"` (`volcano`), `"WF"` (`wildfire`), `"DR"` (`drought`).
  - `properties.eventid`: Integer ID (e.g., `1000212`).
  - `properties.fromdate`: ISO timestamp (e.g., `"2018-08-10T00:00:00"` $\rightarrow$ `"2018-08-10"`).
  - `properties.alertlevel`: `"Green"`, `"Orange"`, `"Red"`.
  - `properties.htmldescription`: Summary string (e.g., `"Orange Flood in India from: 10 Aug 2018 00 to: 24 Sep 2018 00."`).
  - `properties.url.report`: Provenance URL (`https://www.gdacs.org/report.aspx?eventid=1000212&episodeid=2&eventtype=FL`).

---

## 3. Wikipedia MediaWiki & REST Summary Contract

- **Search Endpoint:** `GET https://en.wikipedia.org/w/api.php`
  - `params`: `{"action": "query", "list": "search", "srsearch": f"intitle:{locality} intitle:{year}", "format": "json", "srlimit": 5}`
  - Fallback search if `intitle:` returns no disaster page: `f"{locality} {year} {hazard_or_disaster} flood landslide earthquake cyclone"`
  - Required Header: `{"User-Agent": "DisasterResNet/1.0 (https://github.com/disasterres)"}`
- **Summary Endpoint:** `GET https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}`
  - Extracts `data["title"]`, `data["extract"]`, and `data["content_urls"]["desktop"]["page"]`.

