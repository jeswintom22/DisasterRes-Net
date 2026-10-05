# CrewAI Multi-Agent Disaster Chat Interface — Technical Execution Plan

**Location:** `.claude/plans/crewai_chat_interface_plan.md`
**Based on:** `project-delivery/brainstorms/002-crewai-chat-interface/` (`brainstorm.md`, `crewai-chat-interface-gap-analysis.md`, `sprint.md`)

---

## 1. Overview & Objective
Implement a multi-agent, research-grade disaster intelligence chat interface for DisasterRes-Net. The system accepts natural language queries (e.g., *"Explain the disaster at Wayanad"*), resolves location & event metadata, gathers satellite/aerial/news imagery, executes saliency + LBP + InceptionResNetV2 + Random Forest feature extraction & spatial damage assessment (`DamageLocalizationAnalyzer`), and delivers streamed, numerically validated situation reports via Server-Sent Events (SSE).

---

## 2. Dependency Chain & Execution Sequence

```
[0] Dependency Spike & Pins (crewai==1.15.*, httpx, diskcache, Flask-Limiter, imagehash)
        │
        ├──▶ [1] crew/context.py & crew/pipeline_singleton.py (Artifact store & Single-thread INFERENCE_POOL)
        │          │
        │          ├──▶ [2] crew/vision/fetch.py & prep.py (Hardened SSRF download & decode)
        │          │      │
        │          │      └──▶ [3] crew/vision/gate.py (CLIP Suitability Gate + phash dedup)
        │          │
        │          ├──▶ [4] crew/sources/ (SourceAdapter protocol, ReliefWeb, GDACS, USGS, Copernicus, NASA, GDELT)
        │          │
        │          └──▶ [7] crew/resolve/ (QueryResolver LLM + Geocoder + EventResolver)
        │                     │
        │                     └──▶ [8] Walking Skeleton (Plain Python end-to-end pipeline without CrewAI)
        │                            │
        │                            ├──▶ [9] crew/report/ (FactsBlock + NumericFidelityValidator + Fallback Template)
        │                            │
        │                            ├──▶ [10] crew/runner.py (Job Registry, SSE Event Bus, Timeout & Rate Limits)
        │                            │
        │                            ├──▶ [11] app.py API Routes (/api/chat, /api/chat/stream/<job_id>, /healthz)
        │                            │
        │                            ├──▶ [12] templates/chat.html (UI Stepper, Evidence Strip, Sanitized Markdown)
        │                            │
        │                            └──▶ [13] crew/ (Wrap into CrewAI agents, tasks, and crew execution)
        │                                   │
        │                                   ├──▶ [14] Intent Routing for Follow-up Turns
        │                                   └──▶ [15] Golden-set & Contract Test Suite
```

---

## 3. Sprint Tasks Breakdown (Revised Task Order)

### Task 0: Dependency Spike & Pinning
- **Goal:** Pin explicit versions (`crewai==1.15.*`, `crewai-tools`, `litellm`, `httpx`, `diskcache`, `imagehash`, `Flask-Limiter`, `bleach`).
- **Files:** `requirements.txt`, `.env.example`

### Task 1: Context & Model Singleton (`B2`, `B3`)
- **Goal:** Build `RunContext` with handle-based artifact store (`img://<run_id>/<idx>`). Build `pipeline_singleton.py` with `warm()` initialization and a 1-worker `ThreadPoolExecutor` (`INFERENCE_POOL`) to serialize PyTorch/GPU execution safely.
- **Files:** `crew/context.py`, `crew/pipeline_singleton.py`

### Task 2: Hardened Image Downloader (`B4`)
- **Goal:** Safe HTTP fetch preventing SSRF (validate IP addresses, block private/loopback/link-local IPs, scheme check, content-type check, max bytes limit, Pillow image verification & pixel decompression bomb protection).
- **Files:** `crew/vision/fetch.py`, `crew/vision/prep.py`

### Task 3: Image Suitability Gate (`B1`)
- **Goal:** Build `ImageSuitabilityGate` using zero-shot CLIP classification + perceptual hashing (`phash`, Hamming distance $\le 6$). Reject ground-level photos, map graphics, low-res images, and duplicate crops. Apply confidence penalties for oblique drone shots.
- **Files:** `crew/vision/gate.py`

### Task 4: Resilient Multi-Source Adapters (`A2`, `B6`)
- **Goal:** Implement `SourceAdapter` protocol with circuit breaking, explicit timeouts, two-tier disk caching (`diskcache`), and full `Provenance` tracking.
- **Files:** `crew/sources/base.py`, `crew/sources/reliefweb.py`, `crew/sources/gdacs.py`, `crew/sources/usgs.py`

### Task 5 & 6: Tiered Overhead Imagery & News Fallback (`B1`, `C2`)
- **Goal:** Implement Tier 1 (Copernicus EMS), Tier 3 (NASA GIBS), and Tier 5 (GDELT DOC 2.0 image collage) adapters.
- **Files:** `crew/sources/copernicus_ems.py`, `crew/sources/nasa_gibs.py`, `crew/sources/gdelt.py`

### Task 7: Query & Event Resolver (`A1`, `B8`)
- **Goal:** Implement `QueryResolver` (structured LLM call for intent, location, time hint), geocoding via Nominatim, and multi-candidate `EventResolver` ranking logic.
- **Files:** `crew/resolve/query.py`, `crew/resolve/geocode.py`, `crew/resolve/events.py`

### Task 8: Plain Python Walking Skeleton
- **Goal:** Build a complete end-to-end execution path without CrewAI abstractions first to de-risk pipeline integration and enable standalone testing.
- **Files:** `crew/walking_skeleton.py`

### Task 9: Fact Registry & Numeric Validator (`A3`, `B9`)
- **Goal:** Implement `FactsBlock`, `NumericFidelityValidator` (regex-based numeric check ensuring zero hallucinated metrics or casualties), deterministic report template rendering, and full 8-case degradation matrix.
- **Files:** `crew/report/facts.py`, `crew/report/validator.py`, `crew/report/template.py`, `crew/report/degrade.py`

### Task 10: Asynchronous Job Runner & SSE Bus (`A4`, `B7`, `B14`)
- **Goal:** Implement thread-safe job registry, SSE progress queue, wall-clock timeout (180 s), token budgeting, and JSONL tracing (`logs/runs/{run_id}.jsonl`).
- **Files:** `crew/runner.py`, `crew/trace.py`

### Task 11 & 12: Flask Web Integration & Chat UI (`A4`, `C5`)
- **Goal:** Add `/api/chat` (202 Accepted) and `/api/chat/stream/<job_id>` (SSE) routes to `app.py`. Create `templates/chat.html` with real-time stage progress stepper, live evidence thumbnail strip, and DOMPurify-sanitized markdown rendering.
- **Files:** `app.py`, `templates/chat.html`, `static/js/chat.js`

### Task 13: CrewAI Agent & Task Wrapping
- **Goal:** Wrap the validated pipeline into CrewAI `Agent`, `Task`, and `Crew` objects using `output_pydantic` and JSON handle passing (`image_handles: list[str]`).
- **Files:** `crew/agents.py`, `crew/tasks.py`, `crew/crew.py`

### Task 14: Intent Routing & Follow-up Turns (`B11`)
- **Goal:** Fast-path follow-up questions over cached `FactsBlock` without re-running full 90 s image fetching and vision inference loops.
- **Files:** `crew/resolve/query.py`

### Task 15 & 16: Verification & Final Polish (`B12`, `B13`)
- **Goal:** Build golden-set evaluation suite (`tests/golden/events.yaml`), contract tests, lock `requirements.txt`, update `.env.example`, and document learnings.
- **Files:** `tests/golden/`, `requirements.txt`, `.env.example`

---

## 4. Definition of Done
1. **Zero Hallucination:** No report contains a metric absent from its `FactsBlock` (validated by `NumericFidelityValidator`).
2. **Zero Casualty Claims:** Reports never report casualty/fatality figures.
3. **Mandatory Metadata:** Every report includes resolved event, date, confidence band, caveats, disclaimer, attribution, and `run_id`.
4. **Domain Enforcement:** Ground-level photos and maps are rejected with $\ge 90\%$ recall by `ImageSuitabilityGate`.
5. **First-Class Text Reports:** When zero suitable images exist, the system generates a high-quality text briefing without raising errors or outputting dummy DEM scores.
6. **Thread & GPU Safety:** Inference models loaded exactly once per process via `INFERENCE_POOL(max_workers=1)`. No CUDA concurrency crashes.
7. **Streaming UI:** Asynchronous SSE streaming with live progress stepper and evidence thumbnails.
