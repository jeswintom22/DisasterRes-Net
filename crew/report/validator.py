from __future__ import annotations
import re
from .facts import FactsBlock

class NumericFidelityValidator:
    """Reject generated prose that invents numeric claims not present in facts."""
    def validate(self, report: str, facts: FactsBlock) -> bool:
        # Dates and report headings are intentionally ignored; only decimals/percentages are metrics.
        claimed = re.findall(r"\b\d+(?:\.\d+)?%?\b", report)
        values = facts.values()
        return all(token.rstrip("%") in values for token in claimed if "." in token or token.endswith("%"))
