from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
from crew.sources.base import ProvenanceRef

@dataclass(frozen=True)
class ScoutResult:
    location: str
    resolved_event_id: str
    event_type: str
    event_date: str
    event_description: str
    image_handles: list[str] = field(default_factory=list)
    sources: list[ProvenanceRef] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    degraded_sources: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class AnalysisResult:
    location: str
    event_type: str
    images_analysed: int
    images_rejected: int
    mean_dem_score: float | None
    severity_level: str | None
    mean_affected_area_pct: float | None
    total_damaged_regions: int | None
    max_damage_confidence: float | None
    confidence_band: str
    caveats: list[str]
    emergency_recommendations: list[str]
    per_image: list[dict[str, Any]]
    model_version: str = "DisasterRes-Net"

@dataclass(frozen=True)
class FactsBlock:
    run_id: str
    scout: ScoutResult
    analysis: AnalysisResult
    disclaimer: str = "Automated visual indicators are decision support only; verify them with local authorities and field assessment."

    def values(self) -> set[str]:
        data = asdict(self)
        found: set[str] = set()
        def visit(value: Any) -> None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.add(str(value)); found.add(f"{float(value):.1f}"); found.add(f"{float(value):.2f}")
            elif isinstance(value, dict):
                for item in value.values(): visit(item)
            elif isinstance(value, (list, tuple)): 
                for item in value: visit(item)
        visit(data)
        return found
