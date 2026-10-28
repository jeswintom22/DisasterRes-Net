from __future__ import annotations

from collections import Counter

from crew.report.degrade import caveats_for_images
from crew.report.facts import AnalysisResult
from damage_assessment.localization import DamageAssessmentResult


def aggregate_assessments(
    location: str,
    event_type: str,
    assessments: list[DamageAssessmentResult],
    confidences: list[float] | None = None,
    rejected: int = 0,
    extra_caveats: list[str] | None = None,
) -> AnalysisResult:
    if not assessments:
        base_caveats = list(caveats_for_images(0))
        if extra_caveats:
            base_caveats.extend(extra_caveats)
        return AnalysisResult(location, event_type, 0, rejected, None, None, None, None, None, "none", base_caveats, [], [])

    recommendations = list(dict.fromkeys(rec for item in assessments for rec in item.emergency_recommendations))
    severity = Counter(item.severity_level for item in assessments).most_common(1)[0][0]
    max_conf = max(confidences or [0.0])
    band = "high" if max_conf >= 0.70 else "medium" if max_conf >= 0.45 else "moderate"

    all_caveats = list(caveats_for_images(len(assessments)))
    if extra_caveats:
        for item in extra_caveats:
            if item not in all_caveats:
                all_caveats.append(item)

    return AnalysisResult(
        location,
        event_type,
        len(assessments),
        rejected,
        round(sum(x.dem_score for x in assessments) / len(assessments), 2),
        severity,
        round(sum(x.affected_area_percentage for x in assessments) / len(assessments), 2),
        sum(x.damaged_region_count for x in assessments),
        round(float(max_conf), 4),
        band,
        all_caveats,
        recommendations,
        [{"dem_score": round(x.dem_score, 2), "severity": x.severity_level} for x in assessments],
    )


class MetricAggregatorTool:
    def run(self, location: str, event_type: str, assessments):
        return aggregate_assessments(location, event_type, assessments)
