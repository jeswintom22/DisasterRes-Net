import os
from crew.cache import get, set as cache_set

def test_cache_set_get():
    cache_set("test_key", "test_value")
    assert get("test_key") == "test_value"

def test_cache_miss():
    assert get("missing_key_that_does_not_exist") is None
