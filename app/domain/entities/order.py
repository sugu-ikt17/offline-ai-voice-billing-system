"""Order domain entities — a voice/manual order and its line items."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, TypedDict


class ParsedOrderItem(TypedDict):
    """Raw output of the order parser, before menu matching.

    Just an item name as spoken and a quantity — no price, no menu_item_id,
    since the parser has no knowledge of the menu.
    """

    item: str
    quantity: int


@dataclass
class ParseResult:
    """Structured output of OrderParserService.

    Attributes:
        recognized_items: List of successfully parsed ParsedOrderItem entries.
        unknown_items: List of item names or segments that could not be recognized.
        confidence: Parser confidence score (0.0 to 1.0).
        total_segments: Total number of order segments tokenized.
    """

    recognized_items: list[ParsedOrderItem] = field(default_factory=list)
    unknown_items: list[str] = field(default_factory=list)
    confidence: float = 1.0
    total_segments: int = 0


class OrderStatus(str, Enum):
    PENDING = "pending"
    BILLED = "billed"
    CANCELLED = "cancelled"


@dataclass
class OrderItem:
    menu_item_id: int
    name: str
    quantity: int
    unit_price: float
    id: Optional[int] = None

    @property
    def line_total(self) -> float:
        return round(self.unit_price * self.quantity, 2)

    def to_dict(self) -> dict:
        """Serialise to the canonical Bill Generator wire format."""
        return {
            "menu_id":  self.menu_item_id,
            "name":     self.name,
            "price":    self.unit_price,
            "quantity": self.quantity,
            "subtotal": self.line_total,
        }


@dataclass
class Order:
    items: list[OrderItem] = field(default_factory=list)
    id: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING
    raw_transcript: Optional[str] = None
    created_at: Optional[datetime] = None

    @property
    def subtotal(self) -> float:
        return round(sum(item.line_total for item in self.items), 2)


@dataclass
class MatchResult:
    """Structured output of MenuMatcherService.

    Attributes:
        matched_items:   Items successfully resolved to a menu entry, with
                         price and per-item subtotal filled in.
        unmatched_items: Spoken item names that could not be resolved to any
                         menu entry.  Preserved here so the caller can surface
                         them as warnings rather than silently dropping them.
    """

    matched_items:   list[OrderItem] = field(default_factory=list)
    unmatched_items: list[str]       = field(default_factory=list)

