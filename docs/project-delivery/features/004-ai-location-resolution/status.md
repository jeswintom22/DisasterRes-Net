# Status Tracker: AI-Driven Geographic Location & Toponym Resolution (`004-ai-location-resolution`)

- **Feature ID:** `004-ai-location-resolution`
- **Current Stage:** Implementation Complete (`Phases 1–5` implemented & verified)
- **Last Updated:** 2026-10-28
- **Branch:** `dev4`

---

## Delivery Artifacts Checklist

- [x] [`spec.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/004-ai-location-resolution/spec.md) — Functional requirements (`FR-1`–`FR-8`), edge cases (`EC-1`–`EC-8`), and testable acceptance criteria (`AC-1`–`AC-8`).
- [x] [`plan.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/004-ai-location-resolution/plan.md) — 5-phase implementation plan with architecture flow, file touchpoints, anti-patterns, and verification strategy.
- [x] [`decisions.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/004-ai-location-resolution/decisions.md) — 5 Architecture Decision Records (`ADR-001`–`ADR-005`).
- [x] [`implementation-notes.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/004-ai-location-resolution/implementation-notes.md) — Exact codebase touchpoint matrix, sandboxed system prompt, GDACS XML namespace map, and EXIF DMS conversion formulas.
- [x] [`audit.md`](file:///z:/DisasterRes-Net/docs/project-delivery/features/004-ai-location-resolution/audit.md) — Post-implementation audit report.

---

## Phase Execution Checklist

### [x] Phase 1: Schema Upgrade, Prompt-Injection Sandbox & AI Toponym Extractor
- [x] Expand `ResolvedQuery` in [`crew/resolve/query.py`](file:///z:/DisasterRes-Net/crew/resolve/query.py) with `locality`, `admin_area`, `country`, `country_iso2`, `canonical_query`, `confidence`, and `resolution_method`.
- [x] Create [`crew/resolve/ai_resolver.py`](file:///z:/DisasterRes-Net/crew/resolve/ai_resolver.py) with `<untrusted_user_query>` sandboxing, strict post-parse validation, $3.0\text{ s}$ timeout, and L1 disk caching (`TTL = 7 days`).
- [x] Upgrade deterministic offline fallback in [`crew/resolve/query.py`](file:///z:/DisasterRes-Net/crew/resolve/query.py) to strip action verbs/disaster nouns and extract `time_hint`.

### [x] Phase 2: Multi-Candidate Nominatim Geocoder & Disambiguation Engine
- [x] Expand `GeoLocation` in [`crew/resolve/geocode.py`](file:///z:/DisasterRes-Net/crew/resolve/geocode.py) with `country`, `country_code`, `admin_area`, `importance`, `candidates_evaluated`, and `disambiguation_note`.
- [x] Upgrade Nominatim query to `limit=5`, `addressdetails=1`, and optional `countrycodes=<country_iso2>`.
- [x] Implement composite candidate scoring and hierarchical fallback retry (`locality, country` $\rightarrow$ `locality`).
- [x] Implement `reverse_geocode(lat, lon, offline=False)` in [`crew/resolve/geocode.py`](file:///z:/DisasterRes-Net/crew/resolve/geocode.py).

### [x] Phase 3: Structured ReliefWeb Search & Functional GDACS GeoRSS Parser
- [x] Implement XML/GeoRSS parsing and $300\text{ km}$ haversine proximity filtering in [`crew/sources/gdacs.py`](file:///z:/DisasterRes-Net/crew/sources/gdacs.py).
- [x] Upgrade [`crew/sources/reliefweb.py`](file:///z:/DisasterRes-Net/crew/sources/reliefweb.py) to query using decomposed `locality` and `country` fields.
- [x] Update [`crew/resolve/events.py`](file:///z:/DisasterRes-Net/crew/resolve/events.py) to pass `ResolvedQuery` and `GeoLocation` to both event sources and rank results by `hazard_hint` and `time_hint`.

### [x] Phase 4: EXIF GPS Extraction & `FactsBlock` Provenance Integration
- [x] Create [`crew/vision/exif_gps.py`](file:///z:/DisasterRes-Net/crew/vision/exif_gps.py) to extract EXIF `GPSInfo` (`0x8825`) and reject Null Island `(0.0, 0.0)`.
- [x] Wire EXIF GPS extraction, reverse geocoding, and $> 50\text{ km}$ prompt-vs-EXIF conflict detection into [`crew/crew.py`](file:///z:/DisasterRes-Net/crew/crew.py).
- [x] Surface `disambiguation_note` and EXIF conflict caveats in [`crew/report/facts.py`](file:///z:/DisasterRes-Net/crew/report/facts.py).

### [x] Phase 5: Golden Evaluation Benchmark & Regression Test Suite
- [x] Create [`tests/test_location_resolution.py`](file:///z:/DisasterRes-Net/tests/test_location_resolution.py) with the 25-query golden benchmark (`AC-1`) and tests for `EC-1`–`EC-8` and `AC-1`–`AC-8`.
- [x] Verify full pytest suite passes (`.\.venv\Scripts\python.exe -m pytest tests/ -v`).
