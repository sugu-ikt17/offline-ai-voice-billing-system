"""Bill Generator Service.

Converts the structured output of MenuMatcherService into a complete,
ready-to-display bill.

Responsibilities (this service only):
  ✔ Generate a unique bill number.
  ✔ Capture the current UTC timestamp.
  ✔ Build BillLineItem records from matched OrderItems.
  ✔ Calculate item_count, total_quantity, subtotal.
  ✔ Apply discount and tax (both default to 0 — extension points).
  ✔ Calculate grand_total = subtotal - discount + tax.
  ✔ Carry unmatched item names through as warnings.
  ✔ Round all currency values to 2 decimal places.

Out of scope (intentionally):
  ✘ Database persistence
  ✘ Receipt printing / PDF
  ✘ Payment processing
  ✘ Inventory updates
  ✘ GST-specific rules (tax stays 0 until Sprint N+2)

Extension points:
  - Override ``discount`` and ``tax`` via constructor args or by
    subclassing and overriding ``_compute_discount`` / ``_compute_tax``.
  - Swap the bill-number strategy by overriding ``_new_bill_number``.
"""

import uuid
from datetime import datetime, timezone

from app.domain.entities.bill import BillLineItem, BillResult
from app.domain.entities.order import MatchResult, OrderItem
from app.domain.interfaces.bill_generator_interface import BillGeneratorInterface


class BillGeneratorService(BillGeneratorInterface):
    """Generates a complete BillResult from a MenuMatcher MatchResult.

    Args:
        discount: Fixed discount amount in currency units applied to every
                  bill.  Defaults to 0.  Will be clamped so it never
                  exceeds the subtotal.
        tax:      Fixed tax amount in currency units.  Defaults to 0.
                  (GST percentage logic will be added in a later sprint.)

    Usage::

        generator = BillGeneratorService()           # tax=0, discount=0
        bill      = generator.generate(match_result)
        response  = bill.to_dict()
    """

    def __init__(
        self,
        discount: float = 0.0,
        tax:      float = 0.0,
    ) -> None:
        self._fixed_discount = round(float(discount), 2)
        self._fixed_tax      = round(float(tax),      2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, match_result: MatchResult) -> BillResult:
        """Generate a complete bill from the Menu Matcher output.

        Args:
            match_result: Output of ``MenuMatcherService.match()``.

        Returns:
            A fully populated :class:`BillResult`.

        Raises:
            ValueError: If ``match_result`` is None.
        """
        if match_result is None:
            raise ValueError("match_result must not be None.")

        # Build line items ------------------------------------------------
        line_items = [
            self._make_line_item(order_item)
            for order_item in match_result.matched_items
        ]

        # Aggregate figures -----------------------------------------------
        item_count     = len(line_items)
        total_quantity = sum(li.quantity for li in line_items)
        subtotal       = round(sum(li.subtotal for li in line_items), 2)

        # Discount: clamp so we never go negative -------------------------
        discount   = round(min(self._fixed_discount, subtotal), 2)
        tax        = round(self._fixed_tax, 2)
        grand_total = round(subtotal - discount + tax, 2)

        # Warnings: one message per unmatched item ------------------------
        warnings = [
            f"'{name}' was not found in the menu and has been excluded from the bill."
            for name in match_result.unmatched_items
        ]

        return BillResult(
            bill_number     = self._new_bill_number(),
            date_time       = self._now(),
            items           = line_items,
            item_count      = item_count,
            total_quantity  = total_quantity,
            subtotal        = subtotal,
            discount        = discount,
            tax             = tax,
            grand_total     = grand_total,
            warnings        = warnings,
            unmatched_items = list(match_result.unmatched_items),
        )

    # ------------------------------------------------------------------
    # Extension hooks — override in subclasses for custom logic
    # ------------------------------------------------------------------

    def _compute_discount(self, subtotal: float) -> float:  # noqa: ARG002
        """Return the discount amount for a given subtotal.

        Override to implement percentage-based, voucher, or loyalty
        discounts in a future sprint.
        """
        return self._fixed_discount

    def _compute_tax(self, subtotal: float) -> float:  # noqa: ARG002
        """Return the tax amount for a given subtotal.

        Override to implement GST/VAT percentage logic in a future sprint.
        """
        return self._fixed_tax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_line_item(order_item: OrderItem) -> BillLineItem:
        """Convert an OrderItem (from the Menu Matcher) to a BillLineItem."""
        return BillLineItem(
            name       = order_item.name,
            quantity   = order_item.quantity,
            unit_price = round(float(order_item.unit_price), 2),
            subtotal   = order_item.line_total,   # already rounded in OrderItem
        )

    @staticmethod
    def _new_bill_number() -> str:
        """Generate a short, human-readable unique bill number.

        Format: BILL-<8 uppercase hex chars>
        Example: BILL-3F7A2C1E

        Using only 8 characters keeps it compact enough for a thermal
        receipt while remaining practically unique for a single-shop POS.
        Replace with a database sequence when bill persistence is added.
        """
        return f"BILL-{uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def _now() -> datetime:
        """Return the current UTC datetime.

        Isolated into a static method so tests can patch it easily.
        """
        return datetime.now(timezone.utc)
