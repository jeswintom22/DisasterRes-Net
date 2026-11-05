# Status Tracker: Dynamic Historical & Multi-Source Disaster Event Resolution (`005-historical-disaster-resolution`)

- **Feature ID:** `005-historical-disaster-resolution`
- **Current Stage:** Implementation Complete (`Phases 1–5` implemented & verified)
- **Last Updated:** 2026-11-05
- **Branch:** `dev4`

---

## Delivery Artifacts Checklist

- [x] [`spec.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/005-historical-disaster-resolution/spec.md) — Functional requirements (`FR-1`–`FR-7`), edge cases (`EC-1`–`EC-5`), and testable acceptance criteria (`AC-1`–`AC-6`).
- [x] [`plan.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/005-historical-disaster-resolution/plan.md) — 5-phase 100% dynamic multi-source implementation plan (no offline catalog).
- [x] [`decisions.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/005-historical-disaster-resolution/decisions.md) — Architecture Decision Records (`ADR-001`–`ADR-003`).
- [x] [`implementation-notes.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/005-historical-disaster-resolution/implementation-notes.md) — Codebase touchpoint matrix and GDACS/Wikipedia REST API contracts.
- [x] [`status.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/005-historical-disaster-resolution/status.md) — Recovery-oriented phase checklist for `/implement-task`.

---

## Phase Execution Checklist

### [x] Phase 1: GDACS Historical REST API & Temporal Isolation
- [x] Add `_search_historical_api()` in [`crew/sources/gdacs.py`](file:///z:/DisasterRes-Net/crew/sources/gdacs.py) calling `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`.
- [x] Isolate the rolling 7-day GeoRSS feed (`rss.xml`) so it is not queried when `time_hint` contains a past year.
- [x] Rank matched GDACS features by haversine distance ($\le 300\text{ km}$).

### [x] Phase 2: Wikipedia Live Historical Source & ReliefWeb Web-Search Fallback
- [x] Create [`crew/sources/wikipedia.py`](file:///z:/DisasterRes-Net/crew/sources/wikipedia.py) querying MediaWiki search (`/w/api.php`) and REST summary (`/api/rest_v1/page/summary/{title}`) with non-disaster filtering and canonical peril inference.
- [x] Upgrade [`crew/sources/reliefweb.py`](file:///z:/DisasterRes-Net/crew/sources/reliefweb.py) to `/v2/` and add `_search_via_web_api()` (`site:reliefweb.int`) using `SERPER_API_KEY` / `GOOGLE_CUSTOM_SEARCH_API_KEY` / `SERPAPI_API_KEY` when direct API returns `403`/`410`.

### [x] Phase 3: Strict Temporal Disqualification & Multi-Source Candidate Fusion
- [x] Update [`crew/resolve/events.py`](file:///z:/DisasterRes-Net/crew/resolve/events.py) to integrate `WikipediaSource`, pass `time_hint` to `GDACSSource`, and enforce `_is_temporally_compatible()`.
- [x] Normalize generic `"disaster"` event types to specific inferred perils (`flood`, `landslide`, `earthquake`, `cyclone`, `wildfire`) and merge complementary descriptions.

### [x] Phase 4: Unmatched-Period Guard & `FactsBlock` Numeric Fidelity Fix
- [x] Update `FactsBlock.values()` in [`crew/report/facts.py`](file:///z:/DisasterRes-Net/crew/report/facts.py) to extract numeric tokens embedded in `scout.event_description`, `scout.event_date`, and `analysis.caveats`.
- [x] Update [`crew/crew.py`](file:///z:/DisasterRes-Net/crew/crew.py) to skip `GoogleImageScrapeTool` when an explicit `time_hint` yields no temporally matching event.

### [x] Phase 5: Regression & Integration Test Suite
- [x] Create `tests/test_historical_resolution.py` covering `AC-1`–`AC-6` and `EC-1`–`EC-5`.
- [x] Verify full pytest suite passes (`.\.venv\Scripts\python.exe -m pytest tests/ -v`).
- [ ] Add `_search_historical_api()` in [`crew/sources/gdacs.py`](file:///z:/DisasterRes-Net/crew/sources/gdacs.py) calling `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`.
- [ ] Isolate the rolling 7-day GeoRSS feed (`rss.xml`) so it is not queried when `time_hint` contains a past year.
- [ ] Rank matched GDACS features by haversine distance ($\le 300\text{ km}$).

### [ ] Phase 2: Wikipedia Live Historical Source & ReliefWeb Web-Search Fallback
- [ ] Create [`crew/sources/wikipedia.py`](file:///z:/DisasterRes-Net/crew/sources/wikipedia.py) querying MediaWiki search (`/w/api.php`) and REST summary (`/api/rest_v1/page/summary/{title}`) with non-disaster filtering and canonical peril inference.
- [ ] Upgrade [`crew/sources/reliefweb.py`](file:///z:/DisasterRes-Net/crew/sources/reliefweb.py) to `/v2/` and add `_search_via_web_api()` (`site:reliefweb.int`) using `SERPER_API_KEY` / `GOOGLE_CUSTOM_SEARCH_API_KEY` / `SERPAPI_API_KEY` when direct API returns `403`/`410`.

### [ ] Phase 3: Strict Temporal Disqualification & Multi-Source Candidate Fusion
- [ ] Update [`crew/resolve/events.py`](file:///z:/DisasterRes-Net/crew/resolve/events.py) to integrate `WikipediaSource`, pass `time_hint` to `GDACSSource`, and enforce `_is_temporally_compatible()`.
- [ ] Normalize generic `"disaster"` event types to specific inferred perils (`flood`, `landslide`, `earthquake`, `cyclone`, `wildfire`) and merge complementary descriptions.

### [ ] Phase 4: Unmatched-Period Guard & `FactsBlock` Numeric Fidelity Fix
- [ ] Update `FactsBlock.values()` in [`crew/report/facts.py`](file:///z:/DisasterRes-Net/crew/report/facts.py) to extract numeric tokens embedded in `scout.event_description`, `scout.event_date`, and `analysis.caveats`.
- [ ] Update [`crew/crew.py`](file:///z:/DisasterRes-Net/crew/crew.py) to skip `GoogleImageScrapeTool` when an explicit `time_hint` yields no temporally matching event.

### [ ] Phase 5: Regression & Integration Test Suite
- [ ] Create `tests/test_historical_resolution.py` covering `AC-1`–`AC-6` and `EC-1`–`EC-5`.
- [ ] Verify full pytest suite passes (`.\.venv\Scripts\python.exe -m pytest tests/ -v`).

