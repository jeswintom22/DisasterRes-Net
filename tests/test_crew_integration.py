from __future__ import annotations
from unittest.mock import patch
from types import SimpleNamespace

from crew.crew import run_disaster_crew
from crew.resolve.query import resolve_query

def test_text_only_report_is_useful_when_no_image_is_available():
    with patch("crew.crew.resolve_event") as event, patch("crew.crew.GoogleImageScrapeTool.run", return_value=[]):
        from crew.sources.base import EventCandidate, ProvenanceRef
        event.return_value = EventCandidate("event-1", "flood", "2026-01-01", "A verified flood event affected the requested area.", ProvenanceRef("Test source", "https://example.test"))
        payload = run_disaster_crew("Explain the disaster at Test Wayanad", return_payload=True)
    assert "### Overview" in payload["response"]
    assert "### Damage Assessment" in payload["response"]
    assert "### Emergency Recommendations" in payload["response"]
    assert "No validated overhead imagery" in payload["response"]
    assert payload["facts"]["images_analysed"] == 0

def test_chat_endpoint_starts_a_job_and_rejects_empty_query():
    from app import app
    client = app.test_client()
    assert client.post("/api/chat", json={}).status_code == 400
    with patch("crew.runner.start_job", return_value=SimpleNamespace(id="job-1")):
        response = client.post("/api/chat", json={"query": "Explain the disaster at Wayanad"})
    assert response.status_code == 202
    assert response.get_json()["job_id"]


def test_wayand_alias_resolves_to_wayanad():
    assert resolve_query("Explain the disaster at wayand").location == "Wayanad, Kerala, India"
