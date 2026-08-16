"""Bill domain entities — the final computed bill for an order.

Hierarchy:
  BillLineItem   — one line on the receipt (item name, qty, price, subtotal).
  BillResult     — the complete bill: header, all lines, and all totals.

Design notes:
  - No database IDs here — bill storage is out of scope for now.
  - Currency values are always rounded to 2 decimal places at assignment
    so downstream code never has to worry about floating-point drift.
  - The Bill Generator is the ONLY place that populates BillResult.
    Callers treat it as read-only.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BillLineItem:
    """A single line on the receipt.

    Attributes:
        name:       Display name of the menu item.
        quantity:   How many units were ordered.
        unit_price: Price per single unit (from the menu).
        subtotal:   Line total = unit_price × quantity, rounded to 2 dp.
    """

    name:       str
    quantity:   int
    unit_price: float
    subtotal:   float

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "quantity":   self.quantity,
            "unit_price": self.unit_price,
            "subtotal":   self.subtotal,
        }


@dataclass
class BillResult:
    """The complete, computed bill returned by BillGeneratorService.

    Fields that may be extended in future sprints are already present
    with sensible defaults so callers don't need to change their code:
      - discount  : float  (default 0, Sprint N+1: voucher / manual)
      - tax       : float  (default 0, Sprint N+2: GST / VAT)

    Attributes:
        bill_number:     Unique identifier (UUID4 prefix).
        date_time:       UTC timestamp of generation.
        items:           Ordered list of BillLineItem.
        item_count:      Number of distinct item types.
        total_quantity:  Sum of all quantities across all lines.
        subtotal:        Sum of all line subtotals (before discount/tax).
        discount:        Discount amount in currency units (default 0).
        tax:             Tax amount in currency units (default 0).
        grand_total:     subtotal − discount + tax.
        warnings:        Human-readable messages about unresolved items etc.
        unmatched_items: Spoken item names that could not be resolved.
    """

    bill_number:     str
    date_time:       datetime
    items:           list[BillLineItem] = field(default_factory=list)
    item_count:      int   = 0
    total_quantity:  int   = 0
    subtotal:        float = 0.0
    discount:        float = 0.0
    tax:             float = 0.0
    grand_total:     float = 0.0
    warnings:        list[str] = field(default_factory=list)
    unmatched_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to the full JSON wire format consumed by the API layer."""
        return {
            "bill": {
                "bill_number":    self.bill_number,
                "date_time":      self.date_time.isoformat(),
                "items":          [i.to_dict() for i in self.items],
                "item_count":     self.item_count,
                "total_quantity": self.total_quantity,
                "subtotal":       self.subtotal,
                "discount":       self.discount,
                "tax":            self.tax,
                "grand_total":    self.grand_total,
            },
            "warnings":        self.warnings,
            "unmatched_items": self.unmatched_items,
        }
