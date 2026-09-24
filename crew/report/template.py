from __future__ import annotations
from .facts import FactsBlock

def render_report(facts: FactsBlock) -> str:
    scout, analysis = facts.scout, facts.analysis
    sources = ", ".join(source.name for source in scout.sources if source.name) or "No external source"
    lines = [
        f"## Disaster Situation Report — {scout.location}",
        "",
        f"**Event:** {scout.event_type.title()} | **Date:** {scout.event_date} | **Confidence:** {analysis.confidence_band.title()}",
        "",
        "### Overview",
        scout.event_description,
        "",
        "### Damage Assessment",
    ]
    if analysis.mean_dem_score is None:
        lines.append("No validated overhead imagery was available for automated damage scoring. The report remains useful as an event-context briefing, but does not make a visual damage estimate.")
    else:
        lines.append(f"Analysis of {analysis.images_analysed} image(s) produced a mean DEM score of **{analysis.mean_dem_score:.1f} / 100** ({analysis.severity_level}). The mean affected area was **{analysis.mean_affected_area_pct:.1f}%**, with **{analysis.total_damaged_regions}** detected damage region(s).")
    lines.extend(["", "### Emergency Recommendations"])
    recommendations = analysis.emergency_recommendations or ["Confirm conditions through official emergency-management channels before deployment."]
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(["", "### Caveats", *(f"- {item}" for item in analysis.caveats), "", f"Sources: {sources}.", facts.disclaimer, f"Run ID: `{facts.run_id}`"])
    return "\n".join(lines)
