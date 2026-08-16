"""Generate bill use case — converts a pending order into a final bill.

BUG-01 / BUG-08 FIX:
    The original code passed an `Order` entity directly to
    `bill_generator.generate()`, which expects a `MatchResult`. It then read
    non-existent attributes (`bill_entity.order_id`, `.tax_rate`, `.tax_amount`,
    `.total`) from the returned `BillResult`.

    Fix:
        1. Reconstruct a `MatchResult` from the order's saved items.
        2. Pass the `MatchResult` to `bill_generator.generate()`.
        3. Map the resulting `BillResult` fields correctly to `BillModel`:
               subtotal   → BillResult.subtotal
               tax_rate   → from settings (0.05 default)
               tax_amount → BillResult.tax
               total      → BillResult.grand_total
"""

from app.core.config import settings
from app.core.exceptions import NotFoundException, ValidationException
from app.domain.entities.order import MatchResult, Order, OrderItem, OrderStatus
from app.domain.interfaces.bill_generator_interface import BillGeneratorInterface
from app.infrastructure.database.models.bill_model import BillModel
from app.infrastructure.database.repositories.bill_repository import BillRepository
from app.infrastructure.database.repositories.order_repository import OrderRepository


class GenerateBillUseCase:
    def __init__(
        self,
        bill_generator: BillGeneratorInterface,
        order_repository: OrderRepository,
        bill_repository: BillRepository,
    ) -> None:
        self.bill_generator = bill_generator
        self.order_repository = order_repository
        self.bill_repository = bill_repository

    def execute(self, order_id: int) -> BillModel:
        # ── 1. Load & validate the order ──────────────────────────────────────
        order_model = self.order_repository.get_by_id(order_id)
        if order_model is None:
            raise NotFoundException(f"Order with id {order_id} not found.")
        if order_model.status == OrderStatus.BILLED.value:
            raise ValidationException(f"Order {order_id} has already been billed.")

        # ── 2. Rebuild domain order items from persisted ORM rows ─────────────
        domain_order_items: list[OrderItem] = [
            OrderItem(
                id=i.id,
                menu_item_id=i.menu_item_id,
                name=i.name,
                quantity=i.quantity,
                unit_price=i.unit_price,
            )
            for i in order_model.items
        ]

        # ── 3. Build a MatchResult so BillGeneratorService can consume it ──────
        #    Unmatched items are already excluded at order-save time, so the
        #    unmatched list is empty here.
        match_result = MatchResult(
            matched_items=domain_order_items,
            unmatched_items=[],
        )

        # ── 4. Generate the bill (returns BillResult, not a DB model) ──────────
        bill_result = self.bill_generator.generate(match_result)

        # ── 5. Persist bill: map BillResult → BillModel ────────────────────────
        #    BillModel columns:
        #        subtotal    ← bill_result.subtotal
        #        tax_rate    ← settings.tax_rate   (the configured rate, e.g. 0.05)
        #        tax_amount  ← bill_result.tax      (actual currency amount)
        #        total       ← bill_result.grand_total
        bill_model = BillModel(
            order_id=order_id,
            subtotal=bill_result.subtotal,
            tax_rate=settings.tax_rate,
            tax_amount=bill_result.tax,
            total=bill_result.grand_total,
        )
        created_bill = self.bill_repository.create(bill_model)

        # ── 6. Mark order as billed ────────────────────────────────────────────
        order_model.status = OrderStatus.BILLED.value
        self.order_repository.update(order_model)

        return created_bill
