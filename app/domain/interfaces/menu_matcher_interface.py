"""Contract for matching parsed order items against the shop's menu."""

from abc import ABC, abstractmethod

from app.domain.entities.order import MatchResult, ParsedOrderItem


class MenuMatcherInterface(ABC):
    @abstractmethod
    def match(self, parsed_items: list[ParsedOrderItem]) -> MatchResult:
        """Resolve parsed item names to real menu items with correct prices.

        Returns a MatchResult containing:
          - matched_items:   OrderItem list (with price, quantity, subtotal).
          - unmatched_items: list of spoken item names that could not be resolved.
        """
        raise NotImplementedError
