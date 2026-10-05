"""NASA GIBS image source adapter placeholder with an explicit public endpoint."""
from __future__ import annotations
from .base import ProvenanceRef

class NASAGIBSSource:
    def provenance(self) -> ProvenanceRef:
        return ProvenanceRef("NASA GIBS", "https://earthdata.nasa.gov/eosdis/daacs/gibs", "overhead")
