# CrewAI Multi-Agent Disaster Chat Interface — Detailed Technical Specification

**Location:** `.claude/crewai_chat_interface_spec.md`
**Based on:** `project-delivery/brainstorms/002-crewai-chat-interface/` (`brainstorm.md`, `crewai-chat-interface-gap-analysis.md`, `sprint.md`)

---

## 1. System Overview & Data Flow Architecture

The CrewAI Multi-Agent Disaster Chat Interface provides a natural language query interface for disaster event investigation and quantitative damage analysis.

```
User Query ("Explain disaster at Wayanad")
   │
   ▼
[1] QueryResolver (crew/resolve/query.py) ──▶ ResolvedQuery (Intent, Location, TimeHint)
   │
   ▼
[2] Geocoder & EventResolver (crew/resolve/) ──▶ Geocoded BBox + Candidate Event
   │
   ▼
[3] Multi-Source Image Retrieval (crew/sources/)
   ├── Tier 1: Copernicus EMS Rapid Mapping (Overhead)
   ├── Tier 2: Maxar Open Data Program (Overhead)
   ├── Tier 3: NASA GIBS / Worldview (Overhead)
   ├── Tier 4: ReliefWeb Report Attachments (Mixed)
   └── Tier 5: GDELT DOC 2.0 Image Collage (News Fallback)
   │
   ▼
[4] ImageSuitabilityGate (crew/vision/gate.py)
   ├── Zero-shot CLIP Classifier (Aerial/Oblique vs Ground/Graphic)
   ├── Perceptual Hash Dedup (phash, Hamming <= 6)
   └── Resolution & Aspect Ratio Filtering
   │
   ▼
[5] Artifact Store & Handle Passing (crew/context.py)
   └── Image Bytes stored in RunContext; handle strings ("img://<run_id>/<idx>") passed to agents
   │
   ▼
[6] Hybrid Disaster Pipeline & Damage Localization (crew/pipeline_singleton.py)
   ├── Saliency Attention (preprocessing/saliency.py)
   ├── LBP Texture Extraction (preprocessing/lbp.py)
   ├── CNN Feature Extraction & RF Fusion (step3_classify.py)
   └── M2 Spatial Damage Assessment (damage_assessment/localization.py)
   │
   ▼
[7] FactsBlock & NumericFidelityValidator (crew/report/)
   ├── Fact Registry creation
   ├── Constrained Prose Generation (LLM / gpt-4o-mini)
   ├── Regex-based Numeric Validation
   └── Deterministic Template Fallback on Validation Breach
   │
   ▼
[8] Asynchronous SSE Job Runner (crew/runner.py & app.py)
   ├── POST /api/chat -> 202 Accepted {"job_id": "..."}
   └── GET /api/chat/stream/<job_id> -> text/event-stream
```

---

## 2. Core Data Contracts (Pydantic & Python Data Structures)

### 2.1 Artifact Store & Run Context (`crew/context.py`)
```python
@dataclass(frozen=True)
class Provenance:
    source: str          # "copernicus" | "nasa_gibs" | "reliefweb" | "gdelt"
    tier: int            # 1 to 5
    url: str
    retrieved_at: datetime
    licence: str | None  # "CC-BY-NC-4.0" | "public-domain" | "copyright:news"
    attribution: str | None

class RunContext:
    run_id: str
    query: str
    artifacts: Dict[str, Any] = field(default_factory=dict)       # handle -> np.ndarray
    provenance: Dict[str, Provenance] = field(default_factory=dict)  # handle -> Provenance
    events: queue.Queue = field(default_factory=queue.Queue)
```

### 2.2 Agent Communication Payload (`crew/crew.py` & `crew/tasks.py`)
```python
class ScoutResult(BaseModel):
    location: str
    resolved_event_id: str
    event_type: str
    event_date: str
    event_description: str          # <= 500 tokens, sanitized text
    image_handles: List[str]        # ["img://<run_id>/0", "img://<run_id>/1"]
    sources: List[Dict[str, Any]]   # Provenance references
    rejected: List[Dict[str, str]]  # [{"url": "...", "reason": "ground-level"}]
    degraded_sources: List[str]     # ["Copernicus API timed out"]

class AnalysisResult(BaseModel):
    location: str
    event_type: str
    images_analysed: int
    images_rejected: int
    mean_dem_score: float | None    # None if zero suitable images survive
    severity_level: str | None      # "Minor" | "Moderate" | "Severe" | "Critical" | None
    mean_affected_area_pct: float | None
    total_damaged_regions: int | None
    max_damage_confidence: float | None
    confidence_band: Literal["high", "medium", "low", "none"]
    caveats: List[str]
    emergency_recommendations: List[str]
    per_image: List[Dict[str, Any]]
    model_version: str
```

### 2.3 Fact Registry & Validation (`crew/report/facts.py` & `validator.py`)
```python
@dataclass(frozen=True)
class Fact:
    value: float | str | int
    unit: str = ""
    source: str = ""

class FactsBlock:
    run_id: str
    scout: ScoutResult
    analysis: AnalysisResult
    disclaimer: str

    def numeric_values(self) -> Set[float]:
        """Extract all valid numbers present in the ground-truth analysis result."""
        ...
```

---

## 3. Subsystem Technical Requirements

### 3.1 Model Singleton & Inference Serialization (`crew/pipeline_singleton.py`)
1. **Model Pooling:** `HybridDisasterPipeline` and `DamageLocalizationAnalyzer` MUST be loaded once during process initialization via `warm()`.
2. **GPU Thread Safety:** All inference operations MUST run through a 1-worker thread pool executor (`INFERENCE_POOL = ThreadPoolExecutor(max_workers=1)`).
3. **PyTorch Mode:** All deep feature extraction MUST be wrapped in `torch.inference_mode()`.

### 3.2 Security & Safe Image Ingestion (`crew/vision/fetch.py`)
1. **SSRF Prevention:**
   - Enforce `https://` protocol only.
   - Perform DNS resolution (`socket.getaddrinfo`) and explicitly reject any IP addressing private (`10.x`, `172.16-31.x`, `192.168.x`), loopback (`127.x`), link-local (`169.254.x`), or reserved ranges.
2. **Decompression & Payload Protection:**
   - Enforce `Image.MAX_IMAGE_PIXELS = 50_000_000`.
   - Max download size cap at 10 MB (`Content-Length` header & streaming byte counter).
   - Convert images strictly to RGB, strip EXIF metadata, and downscale using `resize_rgb_to_max_edge(img, 1280)`.

### 3.3 Domain Enforcement (`crew/vision/gate.py`)
1. Zero-shot CLIP classifier evaluates candidate images against prompts for aerial/satellite vs ground-level/graphics.
2. Ground-level photos and map graphics MUST be rejected with $\ge 90\%$ recall.
3. Perceptual hashing (`imagehash.phash`) rejects duplicate crops (Hamming distance $\le 6$).
4. When zero suitable images pass the gate, `mean_dem_score` MUST be set to `None`, triggering a **text-only situation report**.

### 3.4 Strict Numeric Validation (`crew/report/validator.py`)
1. Every numeric figure present in LLM-generated prose is extracted via regex `\d+(?:\.\d+)?`.
2. Numbers are checked against the `FactsBlock` (allowing minor $\pm 0.5\%$ rounding tolerances).
3. If an unverified number (e.g. hallucinated casualty count or altered DEM score) is detected:
   - One retry prompt is issued to the LLM detailing the violation.
   - If the retry fails, the system IMMEDIATELY falls back to the deterministic Markdown report template ([template.py](file:///h:/DSR/DisasterRes-Net/crew/report/template.py)).

### 3.5 Web & Streaming Architecture (`app.py` & `crew/runner.py`)
1. **`POST /api/chat`**: Accepts `{"query": "..."}`, validates length ($\le 600$ chars), initializes background job in `RUNS` registry, and returns `202 Accepted` with `{"job_id": "..."}` in $<100$ ms.
2. **`GET /api/chat/stream/<job_id>`**: Returns SSE `text/event-stream` with headers `Cache-Control: no-cache` and `X-Accel-Buffering: no`. Emits event stages: `resolve`, `sources`, `analysis`, `narration`, `complete`, or `error`.
3. **Sanitization:** All markdown rendered in frontend MUST be purged using `DOMPurify` / `bleach` to block XSS attacks.

---

## 4. 8-Case Degradation & Fallback Matrix

| Case | Scenario | Execution Path | Generated Output |
| :--- | :--- | :--- | :--- |
| 1 | Unparseable / Non-Disaster Query | Reject early in `QueryResolver` | Refusal response with suggestion tips. |
| 2 | Geocoding Failure | `Geocoder` fails to resolve location | Clarification prompt with Nominatim suggestions. |
| 3 | No Recorded Disaster Event | No event found in ReliefWeb/GDACS | Text-only briefing: "No recorded disaster event found in location." |
| 4 | Event Found, Zero Images | News/scrapers return no URLs | Text-only event narrative from source reports; DEM set to `None`. |
| 5 | Images Found, All Rejected by Gate | Scraped images are ground-level/graphics | Text-only narrative + explicit notice detailing image rejection reasons. |
| 6 | Only Penalized Images Survive | Oblique drone shots survive | Assessment generated with `confidence_band = "low"` and prominent caveats. |
| 7 | Pipeline Inference Exception | PyTorch / OpenCV error during processing | Partial report with Scout facts + error note + `run_id` tracking code. |
| 8 | LLM Narration Error / Timeout | OpenAI request fails or fails numeric check | Deterministic Markdown template rendered directly from `FactsBlock`. |

---

## 5. Non-Negotiable Safety & Disclaimer Policy
Every situation report MUST end with the following hardcoded, non-editable disclaimer:

> *\*Disclaimer: This is a model-generated estimate from public imagery and open data sources. It is **not** an official damage assessment and must not be used as the sole basis for operational emergency decisions. Refer to official national disaster management authorities (e.g., NDMA, State EOC, ReliefWeb) for authoritative emergency response information.\**
