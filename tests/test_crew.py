from crew.crew import run_disaster_crew
from crew.context import current_run

def test_crew_smalltalk():
    payload = run_disaster_crew("hello", return_payload=True)
    assert "run_id" in payload
    assert "Hi!" in payload["response"]

def test_crew_followup():
    payload = run_disaster_crew("remind me what was the damage", return_payload=True)
    assert "run_id" in payload
    assert "follow-up" in payload["response"]
