from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Callable, Any
import httpx
from .cache import get as cache_get, set as cache_set
from .context import RunContext, current_run
from .resolve import resolve_event, resolve_query
from .report import FactsBlock, ScoutResult, render_report
from .report.validator import NumericFidelityValidator
from .tools.aggregator import aggregate_assessments
from .tools.image_dl import ImageDownloaderTool
from .tools.image_search import GoogleImageScrapeTool
from .tools.pipeline_tool import analyze_image
from .trace import write_trace
from .vision.gate import ImageSuitabilityGate

def _load_local_env() -> None:
    """Load the project's .env without requiring an additional package."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

def _openai_narrate(facts: FactsBlock, fallback: str, sink: Callable[[dict[str, Any]], None] | None, run: RunContext) -> tuple[str, str]:
    _load_local_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        _emit(run, sink, "narration", "OpenAI key not configured; using deterministic report")
        return fallback, "deterministic (no OpenAI key)"
    _emit(run, sink, "narration", "Generating constrained narration with OpenAI")
    facts_json = json.dumps({"event": facts.scout.event_description, "event_type": facts.scout.event_type, "date": facts.scout.event_date, "analysis": facts.analysis.__dict__, "disclaimer": facts.disclaimer}, ensure_ascii=False)
    prompt = ("Write a concise Markdown disaster situation report using ONLY the supplied JSON facts. "
              "Preserve every numeric value exactly, never invent casualties or facts, and include sections "
              "Overview, Damage Assessment, Emergency Recommendations, Caveats, Sources, and Run ID. "
              "If a metric is null, say it was unavailable. JSON facts:\n" + facts_json)
    try:
        response = httpx.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": os.getenv("LLM_MODEL_NARRATE", "gpt-4o-mini"), "temperature": 0.1, "messages": [{"role": "system", "content": "You are a careful emergency communications writer."}, {"role": "user", "content": prompt}]}, timeout=30)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        if not NumericFidelityValidator().validate(text, facts):
            _emit(run, sink, "narration", "OpenAI response failed numeric validation; using deterministic report")
            return fallback, "deterministic (validation fallback)"
        _emit(run, sink, "narration", "OpenAI narration generated and validated")
        return text, "OpenAI"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        detail = str(exc).replace("\n", " ")[:180]
        _emit(run, sink, "narration", f"OpenAI failed ({detail or type(exc).__name__}); using deterministic report")
        return fallback, "deterministic (OpenAI request failed)"

def _emit(run: RunContext, sink: Callable[[dict[str, Any]], None] | None, stage: str, message: str, **payload: Any) -> None:
    run.emit(stage, message, **payload)
    if sink: sink({"stage": stage, "message": message, **payload})

def run_disaster_crew(query: str, event_sink: Callable[[dict[str, Any]], None] | None = None, return_payload: bool = False):
    normalized = "v3:" + " ".join(query.lower().split())
    cached = cache_get(normalized)
    if cached:
        if event_sink:
            event_sink({"stage": "cache", "message": "Using a cached report"})
        return cached if return_payload else cached["response"]
    run = RunContext(query=query)
    token = current_run.set(run)
    try:
        resolved = resolve_query(query)
        if not resolved.location:
            raise ValueError("Please include a location, for example: 'Explain the disaster in Wayanad'.")
        _emit(run, event_sink, "resolve", f"Resolving event near {resolved.location}")
        event = resolve_event(resolved.location)
        scout = ScoutResult(resolved.location, event.event_id, event.event_type, event.event_date or "unknown", event.description, sources=[event.source], degraded_sources=[])
        _emit(run, event_sink, "sources", "Collecting candidate imagery from SerpAPI/GDELT")
        gate, downloader = ImageSuitabilityGate(), ImageDownloaderTool()
        assessments, confidences = [], []
        candidate_urls = GoogleImageScrapeTool().run(scout.location, scout.event_type)
        _emit(run, event_sink, "sources", f"Found {len(candidate_urls)} candidate image URL(s)")
        for url in candidate_urls:
            image = downloader.run(url)
            if image is None:
                scout.rejected.append({"url": url, "reason": "Image could not be safely downloaded."}); continue
            verdict = gate.evaluate(image)
            if not verdict.accepted:
                scout.rejected.append({"url": url, "reason": verdict.reason}); continue
            handle = run.put_image(image); scout.image_handles.append(handle)
            _emit(run, event_sink, "analysis", f"Analysing evidence image {len(scout.image_handles)}")
            analysis, assessment = analyze_image(image, scout.event_type)
            assessments.append(assessment)
            confidences.append(float((analysis["predictions"].get("damage") or {}).get("confidence", 0.0)))
        if not scout.image_handles:
            scout.degraded_sources.append("No source returned a suitable image; generated a text-only event briefing.")
        analysis_result = aggregate_assessments(scout.location, scout.event_type, assessments, confidences, len(scout.rejected))
        facts = FactsBlock(run.run_id, scout, analysis_result)
        deterministic_response = render_report(facts)
        response, narration_mode = _openai_narrate(facts, deterministic_response, event_sink, run)
        payload = {"response": response, "query": query, "run_id": run.run_id, "narration": narration_mode, "facts": {"images_analysed": analysis_result.images_analysed, "images_rejected": analysis_result.images_rejected, "confidence_band": analysis_result.confidence_band}, "events": run.events}
        cache_set(normalized, payload)
        write_trace(run, "complete")
        return payload if return_payload else response
    except Exception:
        write_trace(run, "error")
        raise
    finally:
        current_run.reset(token)
