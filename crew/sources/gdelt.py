from __future__ import annotations
from .base import ProvenanceRef

class GDELTSource:
    def provenance(self) -> ProvenanceRef:
        return ProvenanceRef("GDELT", "https://www.gdeltproject.org/", "news")
