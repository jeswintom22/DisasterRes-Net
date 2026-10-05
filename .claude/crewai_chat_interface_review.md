# Architectural Design & Pre-Implementation Review Notes

**Location:** `.claude/crewai_chat_interface_review.md`
**Based on:** `project-delivery/brainstorms/002-crewai-chat-interface/` (`brainstorm.md`, `crewai-chat-interface-gap-analysis.md`, `sprint.md`)

---

## 1. Executive Summary & Design Evaluation

The initial CrewAI Chat Interface proposal (`brainstorm.md`) outlined a 3-agent architecture (Scout $\rightarrow$ Analyst $\rightarrow$ Narrator). However, a comprehensive architectural review revealed **4 major feature gaps** and **14 critical unstated gaps (B1–B14)** that directly compromise system stability, concurrency safety, data privacy, and mathematical correctness.

This document synthesizes the pre-implementation audit findings and establishes the mandatory design rules required before code audit and refactoring.

---

## 2. Summary of 14 Unstated Gaps & Required Architecture Adjustments

### B1. Domain Mismatch (P0 - Correctness & Hallucination Risk)
- **Problem:** `HybridDisasterPipeline` & `DamageLocalizationAnalyzer` were trained on overhead (aerial/satellite) imagery. Ground-level photos (e.g. news photos of rescue boats or street-level damage) yield planimetrically invalid DEM & affected area calculations.
- **Solution:** Implement `ImageSuitabilityGate` (zero-shot CLIP + `phash` dedup). Filter out ground-level & graphic images. Degrade confidence to `"low"` or produce **text-only reports** when overhead imagery is unavailable.

### B2. CrewAI Dataclass Serialization & Memory Bloat (P0 - Execution Blocker)
- **Problem:** CrewAI task outputs are text/JSON strings. Passing `np.ndarray` image arrays directly in `ScoutResult` breaks agent communication and context windows.
- **Solution:** Handle-passing architecture (`RunContext`). Store binary images in memory using handles (`img://<run_id>/<idx>`). Pass lightweight JSON Pydantic models containing handle strings between agents.

### B3. GPU Thread Safety & Model Concurrency (P0 - Runtime Crash Risk)
- **Problem:** Parallel web requests executing PyTorch / scikit-learn model inference across multiple threads cause CUDA memory fragmentation, non-deterministic hangs, and VRAM OOM.
- **Solution:** Singleton model initialization via `warm()`. Wrap all inference calls in a 1-worker thread pool (`INFERENCE_POOL = ThreadPoolExecutor(max_workers=1)`).

### B4. SSRF & Unsafe Remote Image Fetching (P1 - Security Vulnerability)
- **Problem:** Image scrapers fetch arbitrary LLM-suggested URLs, exposing internal network services to SSRF attacks and decompression bomb exploits.
- **Solution:** Enforce HTTPS-only, IP address validation (blocking private/loopback/link-local ranges), byte limiters (10 MB max), `Image.MAX_IMAGE_PIXELS = 50_000_000`, EXIF stripping, and magic-byte validation.

### B5. Prompt Injection via External Web Content (P1 - Security & Integrity)
- **Problem:** Scraped disaster reports (ReliefWeb, GDELT, Google snippet text) may contain adversarial instructions or prompt injections.
- **Solution:** Sanitize all incoming external text, delimit inside `<untrusted_source>` tags, enforce 500-token caps per source, and sanitize outputs with server-side `bleach` and client-side `DOMPurify`.

### B6. Lack of Caching Layers (P1 - Latency & Cost Overhead)
- **Problem:** Repeated user queries execute redundant multi-step scraping, HTTP downloads, and feature extraction, causing 30–90 s latencies and API rate-limit exhaustion.
- **Solution:** Two-tiered `diskcache`: L1 for raw source responses and image bytes (6 h / 30 d TTL); L2 for full run payloads keyed on normalized query and model version.

### B7. Unbounded Agent Iterations & Loop Controls (P1 - Resource Exhaustion)
- **Problem:** Unbounded agent retry loops can cause infinite execution cycles and excessive API token consumption.
- **Solution:** Enforce strict agent constraints: `max_iter` (Scout 6, Analyst 4, Narrator 2), `max_rpm=20`, `memory=False`, hard 180 s wall-clock job timeout, and Flask rate-limiting.

### B8. Event Ambiguity & Time Semantics (P1 - Semantic Accuracy)
- **Problem:** Ambiguous location queries (e.g. *"Wayanad"*) match multiple historical events (2018 floods, 2019 landslides, 2024 landslides).
- **Solution:** Candidate ranking by time hint, severity (GDACS level), and recency. Surface resolved event metadata in report headers and provide disambiguation UI chips.

### B9. Missing Graceful Degradation Matrix (P1 - Robustness)
- **Problem:** Unhandled external API failures or image fetch errors lead to HTTP 500 crashes.
- **Solution:** Implement explicit 8-case fallback matrix. Treat text-only disaster briefings as first-class valid responses.

### B10. Image Licensing & Rehosting Compliance (P2 - Legal Compliance)
- **Problem:** Scraped news photos (Tier 5–6) are copyrighted material. Rehosting full-resolution images creates copyright liability.
- **Solution:** Prioritize Tier 1–3 open-access imagery (Copernicus, NASA GIBS). For news photos, display thumbnails with mandatory attribution footers linked to original source URLs.

### B11. Follow-up Query Overhead (P2 - UX Optimization)
- **Problem:** Follow-up questions (e.g. *"What were the recommendations?"*) trigger a full 90 s multi-agent scouting and vision pipeline re-run.
- **Solution:** Intent routing in `QueryResolver`. Route follow-up queries directly to the Narrator using the cached `FactsBlock` (2–5 s response).

### B12. Lack of Automated Evaluation Suite (P2 - Quality Assurance)
- **Problem:** No benchmark exists to evaluate pipeline accuracy or measure suitabilty gate precision/recall.
- **Solution:** Create golden-set regression suite (`tests/golden/events.yaml`) with pinned event fixtures, threshold checks, and gate classification tests.

### B13. Dependency Resolution & Version Drift (P1 - Compatibility)
- **Problem:** Unpinned `crewai>=0.80.0` resolves to CrewAI 1.x, breaking 0.x API contracts, tool imports, and `kickoff()` return shapes.
- **Solution:** Pin exact versions (`crewai==1.15.*`, `crewai-tools`, `litellm`). Disable vector memory (`memory=False`) to avoid dependency conflicts.

### B14. System Observability & Tracing (P2 - Maintainability)
- **Problem:** Inability to inspect intermediate agent steps or debug failed pipeline runs.
- **Solution:** Emit structured JSONL execution logs (`logs/runs/{run_id}.jsonl`) tracking execution timings, rejected images, validator retries, and token usage.

---

## 3. Mandatory Implementation Directives
1. **Zero Direct Edits During Audit:** Audit current workspace code against this spec without making modifications to application code.
2. **Discrepancy Reporting:** Write all identified code discrepancies, bugs, missing contracts, or architectural drifts into an audit report artifact.
3. **Strict Validation Enforcement:** Require deterministic fallback whenever LLM narration alters numerical metrics or invents casualty statistics.
