from __future__ import annotations
from .base import ProvenanceRef

class CopernicusEMSSource:
    def provenance(self) -> ProvenanceRef:
        return ProvenanceRef("Copernicus EMS", "https://emergency.copernicus.eu/mapping/", "overhead")
