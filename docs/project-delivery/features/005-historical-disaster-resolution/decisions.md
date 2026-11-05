# Architecture Decision Records: Dynamic Historical & Multi-Source Disaster Event Resolution (`005-historical-disaster-resolution`)

- **Date:** 2026-11-05
- **Status:** Proposed
- **Feature:** `005-historical-disaster-resolution`

---

## ADR-001: 100% Dynamic 3-Tier Live API Discovery over Hardcoded Offline Disaster Catalog

### Context
When resolving historical or exploratory queries such as *"Find Disaster happened in kerala in the year 2018"*, the system needs to identify the historical disaster (**2018 Kerala Floods**, August 2018) and its hazard class (`flood`). An initial proposal considered maintaining a static offline dictionary of major historical disasters (`historical_catalog.py`). However, the user rightly rejected an offline catalog as a brittle anti-pattern that cannot generalize to arbitrary global localities, sub-districts, or historical years.

### Decision
Reject any hardcoded offline disaster catalog. Instead, implement a **100% Dynamic 3-Tier Live Discovery Architecture**:
1. **Tier 1 — GDACS Historical Events REST API (`https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`):** Official, public, keyless UN/EC-JRC endpoint returning historical alerts as GeoJSON features with exact `[lon, lat]` coordinates, GLIDE numbers (`2018-000134`), alert levels, and date ranges, filtered by $300\text{ km}$ haversine proximity to the geocoded query location.
2. **Tier 2 — Live Web Search Grounding (`Serper.dev` / `Google Custom Search` / `SerpAPI` + `ReliefWeb v2`):** Uses the search API keys already configured in `.env` to dynamically search `site:reliefweb.int/disaster`, `site:reliefweb.int/report`, and `site:preventionweb.net` whenever direct ReliefWeb API calls return `403`/`410`.
3. **Tier 3 — Wikipedia / MediaWiki Live REST API (`en.wikipedia.org/w/api.php` + `/api/rest_v1/page/summary/{title}`):** Public, keyless encyclopedia search and summary extraction that resolves named historical disasters (e.g., `"2018 Kerala floods"`), infers specific hazards on open discovery queries, and provides verified multi-sentence abstracts.

### Consequences
- **Positive:**
  - Scales to **any** region, country, and year globally without manual maintenance or hardcoded lists.
  - Tiers 1 and 3 are 100% free and keyless; Tier 2 leverages search keys already present in `.env`.
  - Haversine coordinate filtering ($\le 300\text{ km}$) on GDACS GeoJSON features automatically disambiguates sub-national queries (e.g., isolating the August 2018 Kerala flood at `[76.403, 9.919]`, $49\text{ km}$ from Kerala's centroid, while rejecting cyclones in Andhra Pradesh and Odisha $> 900\text{ km}$ away).
- **Negative / Trade-offs:**
  - Requires live HTTP requests on a cold cache miss, mitigated by 24-hour L1 disk caching via [`crew/cache.py`](file:///z:/DisasterRes-Net/crew/cache.py) and short timeouts ($8\text{ s}$).

---

## ADR-002: Strict Temporal Disqualification over Soft Additive Year Scoring

### Context
Previously, [`crew/resolve/events.py`](file:///z:/DisasterRes-Net/crew/resolve/events.py#L27-L49) used a soft additive score where matching the requested year added `+1.5`, but mismatching the requested year by 8 years (`2026` vs. `2018`) incurred zero penalty (`0.0`). When ReliefWeb failed (`410 Gone`), the live 2026 GDACS Cyclone `MURASU-26` alert won `candidates[0]` with `score = 1.0` (country match).

### Decision
Enforce **Strict Temporal Disqualification** (`_is_temporally_compatible`) in [`crew/resolve/events.py`](file:///z:/DisasterRes-Net/crew/resolve/events.py) and isolate the rolling 7-day GDACS GeoRSS feed (`rss.xml`) from historical queries. Any candidate with an explicit 4-digit year $Y_{\text{cand}} \ne Y_{\text{req}}$ (where $Y_{\text{req}}$ does not appear in the candidate's date or title/description) is disqualified and removed before candidate selection.

### Consequences
- **Positive:**
  - Guarantees that a live 2026 alert can **never** be substituted for a historical query (e.g., `2018`).
  - If no historical record exists across any of the 3 live tiers for a requested year/month, the pipeline skips image scraping (`images_analysed = 0`) and honestly reports that no record was found for that period.
- **Negative / Trade-offs:**
  - Multi-year droughts or long-running crises must have either their onset year or active year mentioned in the date/description to match.

---

## ADR-003: Whitelisting `FactsBlock` String-Embedded Numerals in `NumericFidelityValidator`

### Context
[`crew/report/validator.py`](file:///z:/DisasterRes-Net/crew/report/validator.py) checks every decimal and percentage token (`r"\b\d+(?:\.\d+)?%?\b"`) in the LLM-generated SITREP against `facts.values()`. However, [`crew/report/facts.py`](file:///z:/DisasterRes-Net/crew/report/facts.py#L45-L60) only traversed `int` and `float` fields of `FactsBlock`, ignoring decimal numbers inside `facts.scout.event_description` (such as `"1.506 million"` or `"5.4 million"`). When OpenAI faithfully quoted those figures from the supplied JSON facts, `NumericFidelityValidator` rejected the narration as a hallucination and forced `Mode · deterministic (validation fallback)`.

### Decision
Extend `FactsBlock.values()` to also extract numeric tokens (`r"\b\d+(?:\.\d+)?\b"`) from `self.scout.event_description`, `self.scout.event_date`, and `self.analysis.caveats`.

### Consequences
- **Positive:**
  - Allows the LLM narrator to cite verified statistics present in `scout.event_description` without triggering false-positive fallbacks.
  - Preserves 100% of the anti-hallucination guarantee: any number *not* present in `FactsBlock` (neither in structured metrics nor in the verified source description) is still immediately rejected.

