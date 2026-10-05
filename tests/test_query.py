from crew.resolve.query import resolve_query

def test_resolve_smalltalk():
    res = resolve_query("hello there")
    assert res.intent == "smalltalk"

def test_resolve_followup():
    res = resolve_query("remind me what was the damage")
    assert res.intent == "followup"

def test_resolve_investigate():
    res = resolve_query("Explain the disaster in Wayanad")
    assert res.intent == "investigate"
    assert res.location == "Wayanad, Kerala, India"
