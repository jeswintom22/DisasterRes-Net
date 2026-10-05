import pytest
from unittest.mock import patch
from crew.resolve.geocode import geocode

def test_geocode_empty():
    assert geocode("") is None
    assert geocode(None) is None

@patch("crew.resolve.geocode.httpx.get")
def test_geocode_success(mock_get):
    from httpx import Response, Request
    mock_get.return_value = Response(200, request=Request("GET", "http://test"), json=[{
        "lat": "10.0", "lon": "20.0",
        "boundingbox": ["9.0", "11.0", "19.0", "21.0"],
        "address": {"country_code": "in"}
    }])

    # Clear cache if any
    from crew.cache import _cache
    _cache.delete("geocode:wayanad_test_mock")

    res = geocode("Wayanad_test_mock")
    assert res is not None
    assert res["lat"] == 10.0
    assert res["lon"] == 20.0
    assert res["bbox"] == [9.0, 11.0, 19.0, 21.0]
    assert res["country_code"] == "IN"
