# Architecture Decision Records: AI-Driven Geographic Location & Toponym Resolution (`004-ai-location-resolution`)

- **Date:** 2026-10-28
- **Status:** Accepted
- **Feature:** `004-ai-location-resolution`

---

## ADR-001: Guarded Two-Stage Hybrid (AI Toponym Extraction + Deterministic Nominatim Grounding) over Pure LLM Coordinate Generation

### Context
The user asked whether an AI model should be used to determine *"where in the world"* a disaster query refers to instead of relying on regex normalization (`re.search(r"\b(?:at|in|near|around)\s+...")`) and hardcoded alias dictionaries (`LOCATION_ALIASES`). Two primary paradigms exist:
1. **Pure AI End-to-End Geocoding:** Ask the LLM to output both the canonical place name and the WGS84 `(latitude, longitude)` coordinates directly.
2. **Guarded Two-Stage Hybrid:** Use the LLM strictly for semantic entity extraction, typo correction, and administrative hierarchy expansion (`locality`, `admin_area`, `country`, `country_iso2`), then pass that structured hierarchy to a deterministic geocoder (OpenStreetMap Nominatim) to obtain authoritative coordinates and bounding boxes.

### Decision
Adopt the **Guarded Two-Stage Hybrid Architecture**. LLMs are strictly prohibited from emitting numerical latitude/longitude coordinates (`FR-2`).

### Consequences
- **Positive:**
  - Eliminates LLM coordinate hallucination (e.g., transposing digits or guessing centroids $200\text{ km}$ away from small towns like Chooralmala or Joshimath), which would otherwise fetch completely wrong Sentinel-2 satellite imagery and Open-Meteo weather series.
  - Combines the natural-language flexibility of LLMs (handling typos, verb-first prompts, and multilingual exonyms) with the cartographic precision and polygon bounding boxes of OpenStreetMap.
- **Negative / Trade-offs:**
  - Requires two network hops on a cold cache miss (one to `LLM_MODEL_SMALL`, one to Nominatim), mitigated by L1 disk caching (`7-day` and `30-day` TTLs).

---

## ADR-002: Direct Structured JSON Extraction via `LLM_MODEL_SMALL` over Multi-Turn CrewAI Agent Loop

### Context
Location resolution runs at the very beginning of Stage 1 (`Scout`) before satellite imagery, weather data, or humanitarian reports can be fetched. Running a full multi-turn CrewAI agent loop with tool calls just to parse a 6-word query would add $4\text{--}12\text{ s}$ of latency and significant token overhead.

### Decision
Perform Stage 1 entity extraction using a single-shot, low-temperature (`temperature=0.0`), structured JSON call to `LLM_MODEL_SMALL` (`gpt-4o-mini` by default) with a hard $3.0\text{ s}$ timeout and an immediate deterministic verb-stripping fallback on error or offline mode.

### Consequences
- **Positive:**
  - Keeps cold-cache extraction latency under $800\text{ ms}$ p95 and warm-cache latency under $5\text{ ms}$.
  - Sandboxes the raw user query inside `<untrusted_user_query>...</untrusted_user_query>` with zero tool-execution privileges, neutralizing indirect prompt-injection attacks (`EC-7`).
- **Negative / Trade-offs:**
  - Does not perform iterative web searches during extraction; instead, disambiguation is handled deterministically in Stage 2 using Nominatim's top-5 candidates and GDACS/ReliefWeb active alerts.

---

## ADR-003: Multi-Candidate Nominatim Geocoding (`limit=5`) with Crisis-Aware Disambiguation over Naive Top-1 Selection

### Context
Previously, [`crew/resolve/geocode.py`](file:///z:/DisasterRes-Net/crew/resolve/geocode.py#L46-L52) queried Nominatim with `limit=1` and returned `results[0]`. Many disaster-prone cities share names globally (e.g., *Tripoli* in Libya vs. Lebanon; *Valencia* in Spain vs. Venezuela; *Santiago* in Chile vs. Philippines).

### Decision
Query Nominatim with `limit=5`, `addressdetails=1`, and optional `countrycodes=<country_iso2>`, then rank candidates using a deterministic composite score combining OpenStreetMap `importance` ($0.45$), administrative/country match with `ResolvedQuery` ($0.30$), and active crisis presence in GDACS/ReliefWeb ($0.25$).

### Consequences
- **Positive:**
  - Automatically disambiguates shared toponyms when either the prompt context or active humanitarian feeds indicate which region is experiencing a crisis.
  - Surfaces a transparent `disambiguation_note` in `FactsBlock.caveats` whenever multiple countries match and no explicit country was specified in the query.
- **Negative / Trade-offs:**
  - Slightly larger JSON payload from Nominatim (`5` items vs. `1` item), which is negligible ($\approx 4\text{ KB}$).

---

## ADR-004: GeoRSS Haversine Proximity Filtering ($\le 300\text{ km}$) for GDACS Integration

### Context
[`crew/sources/gdacs.py`](file:///z:/DisasterRes-Net/crew/sources/gdacs.py) was previously left as a stub returning `[]` with the comment `"Stub — full XML parsing to be implemented as needed"`. However, GDACS (`https://www.gdacs.org/xml/rss.xml`) is the primary real-time global alert feed for sudden-onset disasters (earthquakes, cyclones, floods) and often titles alerts by epicentral region or country rather than every affected village.

### Decision
Implement full XML/GeoRSS parsing in [`crew/sources/gdacs.py`](file:///z:/DisasterRes-Net/crew/sources/gdacs.py) using Python's standard `xml.etree.ElementTree`, matching alerts by **either** country/locality substring match **or** great-circle haversine distance $\le 300\text{ km}$ between the geocoded query coordinates and the alert's `<geo:lat>` / `<geo:long>` centroid.

### Consequences
- **Positive:**
  - Sub-district queries (e.g., *"Chooralmala"* or *"Antakya"*) automatically match regional GDACS alerts whose epicenter is within $300\text{ km}$, even if the village name does not appear in the GDACS XML title.
  - Zero new third-party dependencies required (`xml.etree.ElementTree` and `math` are standard library).
- **Negative / Trade-offs:**
  - A $300\text{ km}$ radius can occasionally include neighboring countries near borders; mitigated by prioritizing alerts that also match `GeoLocation.country` or `ResolvedQuery.hazard_hint`.

---

## ADR-005: Prompt Precedence with Caveat Logging on Upload EXIF GPS Conflicts

### Context
When a user uploads a field photo in [`app.py`](file:///z:/DisasterRes-Net/app.py) or [`crew/crew.py`](file:///z:/DisasterRes-Net/crew/crew.py), the image may contain embedded EXIF GPS coordinates (`GPSLatitude`, `GPSLongitude`). Three cases arise:
1. The user uploads an image with no location in their text prompt (e.g., *"Assess the severity of this flood"*).
2. The user uploads an image whose EXIF GPS matches their text prompt location ($\le 50\text{ km}$).
3. The user uploads an image whose EXIF GPS conflicts with their text prompt location ($> 50\text{ km}$).

### Decision
- When the text prompt contains no location (`ResolvedQuery.location is None`), use the image's EXIF GPS coordinates via `reverse_geocode(lat, lon)` (`resolution_method = "exif_gps"`).
- When both are present and disagree by $> 50\text{ km}$, **prioritize the user's explicit text prompt location** for satellite/weather/event retrieval, and append a prominent warning caveat to `FactsBlock.caveats` reporting the exact EXIF coordinates and distance discrepancy (`EC-5`).

### Consequences
- **Positive:**
  - Prevents surprise overrides when an analyst tests an uploaded image against a specific named scenario, while alerting the analyst if an uploaded photo's geotag does not match the claimed disaster zone (crucial for misinformation/verification workflows).
  - Enables zero-text-location image triage for geotagged field photos (`AC-5`).
- **Negative / Trade-offs:**
  - Requires reading the image header via Pillow before running the `Scout` stage.

