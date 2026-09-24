# Brainstorm: CrewAI Multi-Agent Disaster Chat Interface

**Feature Name:** `crewai-chat-interface`
**Date:** 2026-09-17
**Author:** Brainstorm via `/brainstorm crewai-chat-interface`

---

## Phase 0: Complexity Classification

**Classification: Needs Brainstorming**

Signals:
- New external dependency (CrewAI, LLM provider, web-scraping tools)
- New frontend surface (chat window) alongside existing dashboard (`templates/index.html`)
- Three coordinated agents with defined handoff contracts
- Requires bridging the existing `HybridDisasterPipeline` + `DamageLocalizationAnalyzer` into
  an agent-callable API
- Touches `app.py`, model pipeline, templates, `requirements.txt`, and new module directories

---

## Phase 1: Project Context Summary

### Existing Architecture

```
User (browser) ──POST /api/predict (image + form)──▶ app.py (Flask)
                                                       │
                            ┌──────────────────────────┘
                            ▼
                  HybridDisasterPipeline.analyze(img_rgb)
                    ├── SaliencyAttention        (preprocessing/saliency.py)
                    ├── LBPFeatureExtractor      (preprocessing/lbp.py)
                    ├── CNN features             (step3_classify.py)
                    └── RF classifiers × 2      (step3_classify._load_rf_bundle)
                            │
                            ▼
                  DamageLocalizationAnalyzer.assess(...)
                    ├── Localization backend     (damage_assessment/localization_backends.py)
                    ├── DDM generation
                    └── DEM scoring → DamageAssessmentResult
                            │
                            ▼
                  JSON response → templates/index.html (dashboard)
```

### Key Outputs Already Available from the Pipeline

| Field | Source |
|---|---|
| `dem_score` (0–100) | `DamageAssessmentResult.dem_score` |
| `severity_level` | `DamageAssessmentResult.severity_level` |
| `affected_area_percentage` | `DamageAssessmentResult.affected_area_percentage` |
| `damaged_region_count` | `DamageAssessmentResult.damaged_region_count` |
| `emergency_recommendations` | `DamageAssessmentResult.emergency_recommendations` |
| `disaster_type` prediction | `HybridDisasterPipeline → RF classifier` |
| `damage` prediction confidence | `HybridDisasterPipeline → RF classifier` |

### Gaps That CrewAI Must Fill

1. **No location-to-image pipeline** — users currently upload an image manually. The new flow
   starts with a text query like `"Explain the disaster at Wayanad"`.
2. **No web scraping / news ingestion** — images and context must be fetched dynamically.
3. **No natural language summarisation** — all output is raw JSON; an LLM must narrate it.
4. **No chat UI** — the existing frontend is a single-page upload dashboard.

---

## Phase 2: Deep Research Findings

### What CrewAI Provides

- `crewai` orchestrates sequential or parallel **agents**, each with a **role**, **goal**, and
  **backstory**, plus a list of **tools**.
- An agent executes a **Task** using its tools and produces a string output consumed by the
  next agent.
- A **Crew** wires agents + tasks into a pipeline (`Process.sequential` or `Process.hierarchical`).
- Tools are plain Python callables wrapped with `@tool` or subclassing `BaseTool`.

### Existing Code Reuse Opportunities

| Existing Module | Re-use as |
|---|---|
| `HybridDisasterPipeline.analyze()` | Tool callable inside **Analysis Agent** |
| `DamageLocalizationAnalyzer.assess()` | Tool callable inside **Analysis Agent** |
| `app.py /api/predict` | Split into a dedicated internal helper so agents call Python directly (no HTTP) |
| `DISASTER_DESCRIPTIONS` dict | Summarisation agent context injection |
| `utils/image_io.py` | Image pre-processing before handing to the pipeline |

### Scraping Approach

- **ReliefWeb REST API** — free, authoritative, structured disaster event data with location,
  type, date, and description
- **SerpAPI Google Images** — reliable image search; requires API key (free tier available)
- **`httpx`** — lightweight async HTTP client for both tools

> **Recommendation:** ReliefWeb for event metadata + SerpAPI for images. Both provide Python
> SDKs or simple REST interfaces.

---

## Phase 3: Problem Statement

### User Journey

```
User types: "Explain the disaster at Wayanad"
                    │
                    ▼
         ┌─────────────────────────┐
         │  Chat Window (new UI)   │
         │  POST /api/chat         │
         └────────────┬────────────┘
                      │  query text
                      ▼
         ┌────────────────────────────────────────┐
         │  CrewAI Crew  (3 agents, sequential)   │
         └──┬─────────────────────────────────────┘
            │
   ┌────────▼──────────────┐
   │  Agent 1: Scout       │  Scrapes images + event metadata for the location
   └────────┬──────────────┘
            │  ScoutResult (event metadata + validated image arrays)
   ┌────────▼──────────────┐
   │  Agent 2: Analyst     │  Runs DisasterRes-Net pipeline → DEM + damage metrics
   └────────┬──────────────┘
            │  AnalysisResult (DEM score, severity, regions, recommendations)
   ┌────────▼──────────────┐
   │  Agent 3: Narrator    │  Writes natural-language situation report
   └────────┬──────────────┘
            │  Markdown summary string
            ▼
   Chat window renders the response to the user
```

### Scope

**In scope:**
- New `/api/chat` Flask endpoint consuming text queries
- Three CrewAI agents with well-defined tools and data contracts
- New `crew/` module (agents, tasks, tools, pipeline singleton)
- New chat panel in `templates/index.html` (or a separate `chat.html`)
- CrewAI + LLM provider added to `requirements.txt`

**Out of scope (this brainstorm):**
- Authentication / session management
- Persistent conversation history / memory across sessions
- Fine-tuning the LLM on disaster text
- Replacing the existing image-upload dashboard

---

## Phase 4: Proposed Architecture — Detailed Agent Definitions

---

### Agent 1 — Scout Agent (`DisasterScoutAgent`)

| Attribute | Value |
|---|---|
| **Role** | Disaster Intelligence Scout |
| **Goal** | Locate the most recent disaster event at the given location, fetch geo-context, and retrieve the best available aerial or damage imagery |
| **Backstory** | "You are a field intelligence officer for an emergency response team. Given a location name, you find real disaster imagery and structured event data using public APIs." |

#### Tools

| Tool | Purpose | Implementation |
|---|---|---|
| `ReliefWebEventSearchTool` | Search ReliefWeb `/v1/disasters` API for event type, date, country, and description | `httpx` GET + JSON parse |
| `GoogleImageScrapeTool` | Query SerpAPI `google_images` for `"{location} flood/earthquake aerial damage 2024"` and return top-5 image URLs | SerpAPI Python SDK |
| `ImageDownloaderTool` | Download image bytes from URL, validate with Pillow, resize via `resize_rgb_to_max_edge()` from `utils/image_io.py`, return numpy array | `httpx` + `PIL` + existing utility |

#### Task Definition

```
Task: Scout for "{location}" disaster
  1. Call ReliefWebEventSearchTool("{location}") → get event_type, event_date, event_description
  2. Call GoogleImageScrapeTool("{location} {event_type} aerial damage") → get 3–5 image URLs
  3. Call ImageDownloaderTool(url) for each URL → validate and pre-scale images
  4. Output: ScoutResult with event metadata + list of validated image arrays
```

#### Output Contract → passed to Agent 2

```python
@dataclass
class ScoutResult:
    location: str
    event_type: str         # "flood" | "earthquake" | "landslide" | …
    event_date: str         # ISO date string e.g. "2024-07-30"
    event_description: str  # 2–3 sentence summary from ReliefWeb
    images: list[np.ndarray]  # validated, pre-scaled RGB arrays (max edge 1280px)
    source_urls: list[str]
```

---

### Agent 2 — Analyst Agent (`DisasterAnalystAgent`)

| Attribute | Value |
|---|---|
| **Role** | Damage Analyst |
| **Goal** | Run DisasterRes-Net damage assessment on each scouted image and aggregate DEM scores, severity levels, and spatial metrics |
| **Backstory** | "You are a computer vision engineer embedded in a disaster response unit. You run deep-learning damage analysis on satellite imagery and produce precise numeric metrics for field commanders." |

#### Tools

| Tool | Purpose | Implementation |
|---|---|---|
| `DisasterResPipelineTool` | Wraps `HybridDisasterPipeline.analyze(img_rgb)` → returns predictions + saliency map | Direct Python call via `pipeline_singleton.py` |
| `DamageAssessmentTool` | Wraps `DamageLocalizationAnalyzer.assess(img_rgb, saliency_map, disaster_label)` → returns `DamageAssessmentResult` | Direct Python call via `pipeline_singleton.py` |
| `MetricAggregatorTool` | Takes list of `DamageAssessmentResult` objects → computes mean DEM score, modal severity, total regions, worst-case area% | Pure Python aggregation |

#### Task Definition

```
Task: Analyse images for "{location}" (event_type: "{event_type}")
  For each image in ScoutResult.images:
    a. Call DisasterResPipelineTool(image) → predictions, saliency_map
    b. Call DamageAssessmentTool(image, saliency_map, event_type) → DamageAssessmentResult
  Call MetricAggregatorTool(all_results) → aggregated metrics
  Output: AnalysisResult with consolidated damage metrics
```

#### Output Contract → passed to Agent 3

```python
@dataclass
class AnalysisResult:
    location: str
    event_type: str
    images_analysed: int
    mean_dem_score: float           # 0–100
    severity_level: str             # "Minor" | "Moderate" | "Severe" | "Critical"
    mean_affected_area_pct: float
    total_damaged_regions: int
    max_damage_confidence: float    # highest per-image damage class confidence
    emergency_recommendations: list[str]   # union of all per-image recommendations
    per_image_breakdown: list[dict]        # DEM score + severity per image
```

#### Important Implementation Notes

- Reuse the **cached** singleton from `crew/pipeline_singleton.py` to avoid re-loading
  InceptionResNetV2 weights on every agent run (model load ≈ 5–8 seconds).
- Windows multiprocessing restriction: agent tasks must run in the **main thread** only —
  do NOT use CrewAI's `async_execution=True` for this agent. See `learnings.md`.
- Use `torch.inference_mode()` (already in `step3_classify.py`) — do not regress to
  `torch.no_grad()`.

---

### Agent 3 — Narrator Agent (`DisasterNarratorAgent`)

| Attribute | Value |
|---|---|
| **Role** | Disaster Situation Report Writer |
| **Goal** | Write a clear, concise, and accurate natural-language situation report suitable for emergency coordinators and the public |
| **Backstory** | "You are a senior disaster risk communication officer. You translate technical damage metrics into plain-language situation reports that are accurate, empathetic, and actionable." |

#### Tools

| Tool | Purpose | Implementation |
|---|---|---|
| `MarkdownFormatterTool` | Converts `ScoutResult` + `AnalysisResult` fields into a structured Markdown template with numeric facts | Pure Python f-string templating |
| LLM (CrewAI built-in) | Expands the structured template into flowing narrative paragraphs, interprets DEM score meaning | Provider: OpenAI GPT-4o or Google Gemini via `litellm` |

#### Task Definition

```
Task: Summarise disaster analysis for "{location}"
  1. Call MarkdownFormatterTool(scout_result, analysis_result) → structured data block
  2. Write a 3-paragraph narrative using the LLM:
       § 1 — Event overview: what happened, where, when (from ScoutResult)
       § 2 — Damage assessment: DEM score interpretation, affected area, region count, severity
       § 3 — Emergency recommendations and next steps (from AnalysisResult.emergency_recommendations)
  3. Output: Final Markdown string for display in the chat window
```

#### Example Output

```markdown
## 🔴 Disaster Situation Report — Wayanad, Kerala

**Event:** Landslide / Flood | **Date:** August 2024 | **Severity:** Critical

### Overview
Heavy monsoon rains triggered a series of catastrophic landslides in Wayanad
district, Kerala, burying multiple villages under tonnes of debris and causing
severe flooding in low-lying areas...

### Damage Assessment
Analysis of 4 aerial images using DisasterRes-Net yielded a mean **DEM score of
82.4 / 100**, indicating **Critical** structural and terrain damage. Approximately
**63.2%** of the visible area shows damage signatures, with **17 distinct damaged
regions** identified. Damage classification confidence: **91.3%**.

### Emergency Recommendations
1. Immediate evacuation of downstream settlements in Chooralmala and Mundakkai
2. Deploy search-and-rescue teams with rope-access and thermal-imaging capability
3. Establish temporary shelters on higher ground (> 600 m elevation)
4. Monitor river levels for secondary flood surges
```

---

## Phase 5: System Integration

### New Flask Endpoint (`app.py`)

```python
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    try:
        from crew.crew import run_disaster_crew
        result = run_disaster_crew(query)   # returns Markdown string
        return jsonify({"response": result, "query": query})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
```

### New Module Structure

```
crew/
├── __init__.py
├── agents.py               # Agent definitions (Scout, Analyst, Narrator)
├── tasks.py                # Task definitions wired to agents
├── crew.py                 # Crew assembly → run_disaster_crew(query: str) → str
├── pipeline_singleton.py   # Shared HybridDisasterPipeline + DamageLocalizationAnalyzer
└── tools/
    ├── __init__.py
    ├── reliefweb.py        # ReliefWebEventSearchTool
    ├── image_search.py     # GoogleImageScrapeTool
    ├── image_dl.py         # ImageDownloaderTool
    ├── pipeline_tool.py    # DisasterResPipelineTool + DamageAssessmentTool
    ├── aggregator.py       # MetricAggregatorTool
    └── formatter.py        # MarkdownFormatterTool
```

### Chat UI Panel (new section in `templates/index.html` or `templates/chat.html`)

```
┌────────────────────────────────────────────────────┐
│  💬  DisasterRes-Net Chat                          │
│────────────────────────────────────────────────────│
│                                                    │
│  You: Explain the disaster at Wayanad              │
│                                                    │
│  Bot: ## 🔴 Disaster Situation Report — Wayanad   │
│       **Event:** Landslide | **Severity:** Critical│
│       ...                                          │
│                                                    │
│────────────────────────────────────────────────────│
│  [ Type a location query ...          ]  [ Send ]  │
└────────────────────────────────────────────────────┘
```

---

## Phase 6: New Dependencies

| Package | Purpose |
|---|---|
| `crewai>=0.80.0` | Multi-agent orchestration framework |
| `litellm>=1.40.0` | LLM provider abstraction (OpenAI / Gemini / Ollama) |
| `httpx>=0.27.0` | HTTP client for ReliefWeb + image download tools |
| `google-search-results>=2.4.2` | SerpAPI Python SDK (GoogleImageScrapeTool) |

> **Pydantic Warning:** `crewai >= 0.80` requires `pydantic v2`. Verify no conflict with
> existing `torch==2.8.0` or `scikit-learn==1.7.1` (both are pydantic-v2 compatible).

---

## Phase 7: Approaches Comparison

### Approach A — Full CrewAI Sequential ✅ Recommended

- **Summary:** Three agents in `Process.sequential`. Scout → Analyst → Narrator. Output of
  each task is a string / context passed to the next agent.
- **How it works:** `Crew(agents=[scout, analyst, narrator], tasks=[t1, t2, t3], process=Process.sequential).kickoff(inputs={"query": query})`
- **Pros:** Clean separation of concerns; each agent independently testable; CrewAI handles
  retry and logging; LLM swappable via `litellm`.
- **Cons:** Sequential latency (scout + analysis + narration ≈ 30–90s); requires LLM API key.
- **Risk:** Medium (LLM provider dependency; scraping fragility on changing APIs)
- **Effort:** L
- **Files touched:** `crew/` (new), `app.py`, `templates/`, `requirements.txt`

### Approach B — Thin Wrapper (no CrewAI)

- **Summary:** Single Python function: scrape → pipeline → GPT summarise. No agent abstraction.
- **Pros:** Simpler, fewer dependencies, easier to debug.
- **Cons:** No modularity; no retry; violates stated architecture requirement.
- **Risk:** Low
- **Effort:** M

### Approach C — CrewAI Hierarchical with Manager Agent

- **Summary:** A Manager LLM agent dynamically delegates to Scout, Analyst, Narrator as tools.
- **Pros:** More adaptive; manager can retry failed scrapes or request different image sources.
- **Cons:** Higher cost (extra LLM round-trips); significantly more complex setup; longer latency.
- **Risk:** High
- **Effort:** XL

**→ Recommendation: Approach A** directly matches the stated 3-agent architecture, is the
simplest CrewAI path, and provides the best balance of modularity and delivery speed.

---

## Phase 8: Open Questions

1. **LLM Provider** — OpenAI GPT-4o, Google Gemini 1.5 Pro, or local Ollama (`llama3`)?
   Affects API key management, cost, and offline viability.
2. **Image Source** — SerpAPI requires a paid account. Accept this, or restrict to
   ReliefWeb-attached media images only (free, fewer images)?
3. **Latency Budget** — Is a 30–90 second chat response acceptable? If not, should the
   Narrator stream tokens via Server-Sent Events (SSE) so the user sees text appearing live?
4. **Chat History** — Should each session retain prior messages (in-memory list passed as
   CrewAI context), or is every query fully stateless?
5. **UI Placement** — Separate `chat.html` page (cleaner URL routing), or a collapsible
   side panel inside the existing `index.html` dashboard?
6. **Fallback Behaviour** — If Scout finds no images for a location (scraping failure),
   should Narrator produce a text-only report from ReliefWeb metadata alone?
7. **Windows Multiprocessing** — CrewAI's async execution uses `asyncio`. Verify no conflict
   with `torch` CUDA context on Windows (see `learnings.md`).

---

## Recommended Delivery Path

```
/spec-task → /plan-task → /implement-task → /review-task
```

### Suggested Sprint Breakdown

| # | Task | Effort |
|---|---|---|
| 1 | `crew/pipeline_singleton.py` — shared model singletons | S |
| 2 | `crew/tools/reliefweb.py` — ReliefWebEventSearchTool | S |
| 3 | `crew/tools/image_search.py` + `image_dl.py` — scraping tools | M |
| 4 | `crew/tools/pipeline_tool.py` — DisasterResPipelineTool + DamageAssessmentTool | M |
| 5 | `crew/tools/aggregator.py` + `formatter.py` — metric aggregation + Markdown template | S |
| 6 | `crew/agents.py` + `crew/tasks.py` — agent and task definitions | M |
| 7 | `crew/crew.py` — Crew assembly, `run_disaster_crew()` entry point | S |
| 8 | `app.py` — `/api/chat` Flask endpoint | S |
| 9 | `templates/` — Chat UI panel (message list + input + send button) | M |
| 10 | Integration test with a known event (e.g. "Wayanad landslide") | M |
| 11 | `requirements.txt` + `.env.example` update | S |

---

## Rejected Alternatives

| Alternative | Reason Rejected |
|---|---|
| LangChain agents only | CrewAI explicitly specified; LangChain adds similar weight with less role clarity |
| Autogen multi-agent | Requires separate runtime; overkill for a linear 3-agent flow |
| Static hardcoded disaster database | Defeats the purpose of a live, location-aware query system |
| Replacing existing upload dashboard | Out of scope; chat is additive, not a replacement |

---

*Produced following `.claude/skills/brainstorm/SKILL.md` Phase 0–7 workflow.*
