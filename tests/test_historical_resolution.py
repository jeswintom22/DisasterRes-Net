from __future__ import annotations

import pytest

from crew.report.facts import AnalysisResult, FactsBlock, ScoutResult
from crew.report.validator import NumericFidelityValidator
from crew.resolve.events import _is_temporally_compatible, find_events, resolve_event
from crew.resolve.query import resolve_query
from crew.sources.base import EventCandidate, ProvenanceRef
from crew.sources.gdacs import GDACSSource, _parse_time_hint_bounds
from crew.sources.wikipedia import WikipediaSource, _extract_date_from_text, _infer_hazard


def test_ac1_kerala_2018_discovery_resolution():
    """AC-1: Discovery query 'Find Disaster happened in kerala in the year 2018' resolves flood."""
    q = resolve_query("Find Disaster happened in kerala in the year 2018")
    assert q.time_hint == "2018"
    assert "kerala" in (q.location or "").lower()

    ev = resolve_event(q, time_hint=q.time_hint)
    assert ev.event_type == "flood", f"Expected flood, got: {ev.event_type}"
    assert "2018" in (ev.event_date or ""), f"Expected 2018 in event date, got: {ev.event_date}"
    assert "murasu" not in ev.description.lower()
    assert "murasu" not in ev.event_id.lower()
    assert ev.event_type != "cyclone"


def test_ac2_gdacs_historical_rest_and_bounds_parsing():
    """AC-2: GDACS historical API bounds parsing and coordinate proximity."""
    bounds = _parse_time_hint_bounds("in the year 2018")
    assert bounds is not None
    assert bounds[0] == "2018-01-01"
    assert bounds[1] == "2018-12-31"
    assert bounds[2] == 2018

    month_bounds = _parse_time_hint_bounds("August 2018")
    assert month_bounds is not None
    assert month_bounds[0] == "2018-08-01"
    assert month_bounds[1] == "2018-08-31"
    assert month_bounds[2] == 2018

    # Query GDACS historical API for Kerala (lat 10.35, lon 76.51)
    gdacs = GDACSSource()
    matches = gdacs.search(
        "Kerala, India",
        locality="Kerala",
        country="India",
        lat=10.3528,
        lon=76.5120,
        time_hint="2018",
    )
    assert len(matches) >= 1
    top = matches[0]
    assert top.event_type == "flood"
    assert "2018" in top.event_date
    assert "1000212" in top.event_id


def test_ac3_strict_temporal_disqualification():
    """AC-3: Candidates with conflicting explicit years are strictly disqualified."""
    mismatched = EventCandidate(
        "gdacs-tc-1001346",
        "cyclone",
        "2026-11-01",
        "Tropical Cyclone MURASU-26 active in North Indian Ocean.",
        ProvenanceRef("GDACS", "https://www.gdacs.org"),
    )
    assert not _is_temporally_compatible(mismatched, "2018")
    assert not _is_temporally_compatible(mismatched, "in the year 2018")

    matching = EventCandidate(
        "fl-2018-000134",
        "flood",
        "2018-08-16",
        "2018 Kerala floods in south India.",
        ProvenanceRef("ReliefWeb", "https://reliefweb.int"),
    )
    assert _is_temporally_compatible(matching, "2018")
    assert _is_temporally_compatible(matching, None)

    undated = EventCandidate(
        "local-fallback",
        "disaster",
        "",
        "No explicit date specified in this brief.",
        ProvenanceRef("Local", ""),
    )
    assert _is_temporally_compatible(undated, "2018")


def test_ac5_numeric_fidelity_whitelists_string_decimals():
    """AC-5: Numbers embedded in scout description (e.g. 1.506 million) pass validation."""
    scout = ScoutResult(
        location="Kerala, India",
        resolved_event_id="fl-2018-000134-ind",
        event_type="flood",
        event_date="2018-08-16",
        event_description="Catastrophic flooding where 1.506 million people were displaced and 5.4 million affected.",
        sources=[ProvenanceRef("ReliefWeb", "https://reliefweb.int")],
    )
    analysis = AnalysisResult(
        location="Kerala, India",
        event_type="flood",
        images_analysed=2,
        images_rejected=0,
        mean_dem_score=42.5,
        severity_level="Moderate",
        mean_affected_area_pct=6.8,
        total_damaged_regions=3,
        max_damage_confidence=0.88,
        confidence_band="High",
        caveats=[],
        emergency_recommendations=["Deploy flood response desks"],
        per_image=[],
    )
    facts = FactsBlock("test-run-1", scout, analysis)

    values = facts.values()
    assert "1.506" in values
    assert "5.4" in values
    assert "42.5" in values
    assert "6.8" in values

    validator = NumericFidelityValidator()

    # Valid report citing numbers from description and analysis
    valid_report = (
        "### Overview\n"
        "Flooding in Kerala in 2018 displaced 1.506 million people. "
        "Overall 5.4 million residents were affected.\n"
        "### Damage Assessment\n"
        "Analysis of images produced a mean DEM score of 42.5% with 6.8% affected area."
    )
    assert validator.validate(valid_report, facts)

    # Hallucinated report inventing a new decimal percentage not in facts
    invalid_report = (
        "### Overview\n"
        "Flooding in Kerala in 2018 was severe.\n"
        "### Damage Assessment\n"
        "Analysis of images produced a hallucinated metric of 99.9% damage."
    )
    assert not validator.validate(invalid_report, facts)


def test_wikipedia_hazard_and_date_inference():
    """Test Wikipedia source date extraction and hazard mapping helpers."""
    assert _infer_hazard("On 16 August 2018, severe floods affected Kerala") == "flood"
    assert _infer_hazard("Massive landslides struck Chooralmala and Mundakkai in Wayanad") == "landslide"
    assert _infer_hazard("A magnitude 7.8 earthquake struck southern Turkey") == "earthquake"

    assert _extract_date_from_text("On 16 August 2018, severe floods") == "2018-08-16"
    assert _extract_date_from_text("Started in July 2024 in Wayanad") == "2024-07-01"
    assert _extract_date_from_text("Disaster occurred on 2023-02-06 near Gaziantep") == "2023-02-06"

