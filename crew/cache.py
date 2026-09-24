"""Small bounded in-process cache for completed text reports."""
from __future__ import annotations
import time
from threading import Lock

_items: dict[str, tuple[float, object]] = {}
_lock = Lock()

def get(key: str, ttl_s: int = 3600):
    with _lock:
        value = _items.get(key)
        if value and time.monotonic() - value[0] < ttl_s:
            return value[1]
        _items.pop(key, None)
        return None

def set(key: str, value: object) -> None:
    with _lock:
        _items[key] = (time.monotonic(), value)
