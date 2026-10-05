import httpx
from crew.cache import get as cache_get, set as cache_set

def geocode(location: str) -> dict | None:
    """
    Geocodes a location string using Nominatim.
    Returns a dict with lat, lon, bbox, and country_code or None if failed/not found.
    Bbox format: [latMin, latMax, lonMin, lonMax]
    """
    if not location:
        return None

    cache_key = f"geocode:{location.strip().lower()}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "format": "jsonv2",
        "q": location,
        "limit": 1,
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "DisasterRes-Net/1.0 (academic research)"
    }

    try:
        response = httpx.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None

        result = data[0]
        bbox_raw = result.get("boundingbox", [])
        bbox = [float(c) for c in bbox_raw] if bbox_raw else None
        
        address = result.get("address", {})
        country_code = address.get("country_code", "").upper()

        out = {
            "lat": float(result["lat"]),
            "lon": float(result["lon"]),
            "bbox": bbox,
            "country_code": country_code
        }
        cache_set(cache_key, out, ttl_s=86400 * 30)  # cache for 30 days
        return out
    except Exception:
        return None
