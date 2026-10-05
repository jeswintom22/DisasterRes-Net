from __future__ import annotations

def caveats_for_images(count: int) -> list[str]:
    if count == 0:
        return ["No suitable overhead image was available, so no DEM score is reported.", "Event information may be incomplete when source services are unavailable."]
    return ["Image suitability is screened automatically and must be confirmed by a qualified analyst."]
