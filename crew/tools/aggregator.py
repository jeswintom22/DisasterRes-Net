from __future__ import annotations
from collections import Counter
from damage_assessment.localization import DamageAssessmentResult
from crew.report.degrade import caveats_for_images
from crew.report.facts import AnalysisResult

def aggregate_assessments(location: str, event_type: str, assessments: list[DamageAssessmentResult], confidences: list[float] | None = None, rejected: int = 0) -> AnalysisResult:
    if not assessments:
        return AnalysisResult(location, event_type, 0, rejected, None, None, None, None, None, "none", caveats_for_images(0), [], [])
    recommendations = list(dict.fromkeys(rec for item in assessments for rec in item.emergency_recommendations))
    severity = Counter(item.severity_level for item in assessments).most_common(1)[0][0]
    return AnalysisResult(location, event_type, len(assessments), rejected, round(sum(x.dem_score for x in assessments) / len(assessments), 2), severity, round(sum(x.affected_area_percentage for x in assessments) / len(assessments), 2), sum(x.damaged_region_count for x in assessments), max(confidences or [0.0]), "low", caveats_for_images(len(assessments)), recommendations, [{"dem_score": round(x.dem_score, 2), "severity": x.severity_level} for x in assessments])

class MetricAggregatorTool:
    def run(self, location: str, event_type: str, assessments): return aggregate_assessments(location, event_type, assessments)
