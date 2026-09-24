# Sprint: CrewAI Multi-Agent Disaster Chat Interface

**Brainstorm:** `002-crewai-chat-interface`
**Delivery Path:** `/spec-task → /plan-task → /implement-task → /review-task`

---

## Dependency Chain

```
[1] pipeline_singleton.py
        │
        ├──▶ [4] pipeline_tool.py
        │          │
        │          └──▶ [6] agents.py + tasks.py
        │                       │
        ├──▶ [5] aggregator.py  └──▶ [7] crew.py
        │          │                       │
        └──▶ [2] reliefweb.py             └──▶ [8] /api/chat (app.py)
        │                                               │
        └──▶ [3] image_search.py + image_dl.py         └──▶ [9] Chat UI
                   │
                   └──▶ [6] agents.py

[10] Integration test (depends on all above)
[11] requirements.txt (can be done alongside any step)
```

---

## Tasks

### T1 — `crew/pipeline_singleton.py`
**Effort:** S | **Depends on:** nothing

Create a module-level singleton for `HybridDisasterPipeline` and `DamageLocalizationAnalyzer`
so model weights are loaded once per server process and reused across all agent runs.

```python
# Exposed interface
def get_pipeline() -> HybridDisasterPipeline: ...
def get_damage_analyzer() -> DamageLocalizationAnalyzer: ...
```

**Files:** `crew/pipeline_singleton.py`

---

### T2 — `crew/tools/reliefweb.py`
**Effort:** S | **Depends on:** nothing

Implement `ReliefWebEventSearchTool` using the ReliefWeb REST API v1.

- Endpoint: `https://api.reliefweb.int/v1/disasters`
- Filter by location name (fuzzy text match on `name` field)
- Return: `event_type`, `event_date`, `country`, `event_description` (2–3 sentences)
- Fallback: if no event found, return generic descriptor based on location name

**Files:** `crew/tools/reliefweb.py`

---

### T3 — `crew/tools/image_search.py` + `crew/tools/image_dl.py`
**Effort:** M | **Depends on:** nothing

`GoogleImageScrapeTool` (image_search.py):
- Uses SerpAPI `google_images` endpoint
- Query: `"{location} {event_type} aerial damage satellite"`
- Returns: list of 3–5 image URLs

`ImageDownloaderTool` (image_dl.py):
- Downloads image bytes via `httpx` with timeout
- Validates with `PIL.Image.open()`
- Rescales via `resize_rgb_to_max_edge(img_rgb, 1280)` from `utils/image_io.py`
- Returns: `np.ndarray` (RGB, uint8)
- Skips invalid or unreachable URLs silently

**Files:** `crew/tools/image_search.py`, `crew/tools/image_dl.py`

---

### T4 — `crew/tools/pipeline_tool.py`
**Effort:** M | **Depends on:** T1

`DisasterResPipelineTool`:
- Input: `np.ndarray` (RGB image)
- Calls `get_pipeline().analyze(img_rgb)`
- Returns: `predictions` dict + `saliency_map` + `lbp` output

`DamageAssessmentTool`:
- Input: `np.ndarray`, `saliency_map`, `disaster_label: str`
- Calls `get_damage_analyzer().assess(img_rgb, saliency_map, disaster_label, localization_backend="provided")`
- Returns: `DamageAssessmentResult`

**Files:** `crew/tools/pipeline_tool.py`

---

### T5 — `crew/tools/aggregator.py` + `crew/tools/formatter.py`
**Effort:** S | **Depends on:** T4

`MetricAggregatorTool` (aggregator.py):
- Input: `list[DamageAssessmentResult]`
- Computes: `mean_dem_score`, `modal_severity_level`, `sum(damaged_region_count)`, `max(affected_area_pct)`, union of `emergency_recommendations`
- Returns: `AnalysisResult` dataclass

`MarkdownFormatterTool` (formatter.py):
- Input: `ScoutResult`, `AnalysisResult`
- Returns: structured Markdown template with numeric facts for LLM to narrate

**Files:** `crew/tools/aggregator.py`, `crew/tools/formatter.py`

---

### T6 — `crew/agents.py` + `crew/tasks.py`
**Effort:** M | **Depends on:** T2, T3, T4, T5

Define the three agents with roles, goals, backstories, and tool lists.
Define the three tasks with descriptions, expected output format, and agent assignments.

```python
# agents.py
scout_agent   = Agent(role="Disaster Intelligence Scout",   tools=[reliefweb_tool, image_search_tool, image_dl_tool], ...)
analyst_agent = Agent(role="Damage Analyst",                tools=[pipeline_tool, assessment_tool, aggregator_tool], ...)
narrator_agent = Agent(role="Disaster Report Writer",       tools=[formatter_tool], llm=..., ...)

# tasks.py
scout_task   = Task(description="Scout for {location}...",  agent=scout_agent,   expected_output="ScoutResult JSON")
analyst_task = Task(description="Analyse images...",        agent=analyst_agent, expected_output="AnalysisResult JSON", context=[scout_task])
narrator_task = Task(description="Summarise disaster...",   agent=narrator_agent, expected_output="Markdown report",   context=[scout_task, analyst_task])
```

**Files:** `crew/agents.py`, `crew/tasks.py`

---

### T7 — `crew/crew.py`
**Effort:** S | **Depends on:** T6

Assemble the `Crew` and expose `run_disaster_crew(query: str) -> str`.

```python
def run_disaster_crew(query: str) -> str:
    location = _extract_location(query)   # simple regex or LLM extraction
    crew = Crew(
        agents=[scout_agent, analyst_agent, narrator_agent],
        tasks=[scout_task, analyst_task, narrator_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs={"location": location, "query": query})
    return str(result)
```

**Files:** `crew/crew.py`, `crew/__init__.py`

---

### T8 — `/api/chat` endpoint (`app.py`)
**Effort:** S | **Depends on:** T7

Add `/api/chat` POST endpoint to `app.py`.
- Accepts `{"query": "Explain the disaster at Wayanad"}`
- Returns `{"response": "<markdown>", "query": "<original>"}`
- 500 on crew failure with error message

**Files:** `app.py`

---

### T9 — Chat UI Panel
**Effort:** M | **Depends on:** T8

Add a chat panel to `templates/index.html` (or create `templates/chat.html`).

Components:
- Message list with user / bot bubbles
- Markdown rendering (use `marked.js` CDN)
- Text input + Send button
- Loading spinner while crew is running
- Auto-scroll to latest message

**Files:** `templates/index.html` or `templates/chat.html`, `static/`

---

### T10 — Integration Test
**Effort:** M | **Depends on:** T1–T9

Write `tests/test_crew_integration.py`:
- Mock SerpAPI + ReliefWeb HTTP calls using `responses` or `httpx` mock
- Use a real local test image (from `raw_dataset/`) as the "downloaded" image
- Assert that `run_disaster_crew("Explain the disaster at Wayanad")` returns a non-empty
  Markdown string containing expected sections (Overview, Damage Assessment, Recommendations)
- Assert DEM score is in range [0, 100]

**Files:** `tests/test_crew_integration.py`

---

### T11 — Dependencies + Environment Docs
**Effort:** S | **Depends on:** nothing

Update `requirements.txt`:
```
crewai>=0.80.0
litellm>=1.40.0
httpx>=0.27.0
google-search-results>=2.4.2
```

Create `.env.example`:
```
OPENAI_API_KEY=sk-...
SERPAPI_API_KEY=...
DISASTERRES_LLM_MODEL=gpt-4o
```

**Files:** `requirements.txt`, `.env.example`

---

## Summary

| Task | Effort | Blocked by |
|---|---|---|
| T1 pipeline_singleton | S | — |
| T2 reliefweb tool | S | — |
| T3 image scraping tools | M | — |
| T4 pipeline tools | M | T1 |
| T5 aggregator + formatter | S | T4 |
| T6 agents + tasks | M | T2, T3, T4, T5 |
| T7 crew assembly | S | T6 |
| T8 /api/chat endpoint | S | T7 |
| T9 chat UI | M | T8 |
| T10 integration test | M | T1–T9 |
| T11 deps + env docs | S | — |
| **Total** | **L** | |
