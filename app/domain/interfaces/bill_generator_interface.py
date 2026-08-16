"""Contract for the Bill Generator service.

Accepts a MatchResult (output of MenuMatcherService) and produces a
BillResult — the complete, ready-to-display bill.

The interface deliberately knows nothing about:
  - Databases          (no storage is performed here)
  - Tax rules          (the implementation chooses the rate)
  - Discount logic     (may be extended in a future sprint)
  - Printing / PDF     (out of scope)
"""

from abc import ABC, abstractmethod

from app.domain.entities.bill import BillResult
from app.domain.entities.order import MatchResult


class BillGeneratorInterface(ABC):
    @abstractmethod
    def generate(self, match_result: MatchResult) -> BillResult:
        """Generate a complete bill from the Menu Matcher output.

        Args:
            match_result: The structured result from MenuMatcherService,
                          containing ``matched_items`` and ``unmatched_items``.

        Returns:
            A :class:`BillResult` ready to be serialised and returned to
            the caller.  No side-effects (no DB writes, no printing).
        """
        raise NotImplementedError
