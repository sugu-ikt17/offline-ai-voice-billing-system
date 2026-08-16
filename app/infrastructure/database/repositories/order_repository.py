"""Order repository — database operations for orders and their items."""

from typing import Optional

from sqlalchemy.orm import Session, joinedload

from app.infrastructure.database.models.order_model import OrderModel


class OrderRepository:
    """Data access layer for orders."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, order: OrderModel) -> OrderModel:
        """Persist a new order (with its items) and return it refreshed."""
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

    def get_all(self) -> list[OrderModel]:
        """Return all orders, most recent first."""
        return (
            self.db.query(OrderModel)
            .options(joinedload(OrderModel.items))
            .order_by(OrderModel.created_at.desc())
            .all()
        )

    def get_by_id(self, order_id: int) -> Optional[OrderModel]:
        """Return an order (with items) by id, or None if not found."""
        return (
            self.db.query(OrderModel)
            .options(joinedload(OrderModel.items))
            .filter(OrderModel.id == order_id)
            .first()
        )

    def update(self, order: OrderModel) -> OrderModel:
        """Commit changes made to an existing order and refresh it."""
        self.db.commit()
        self.db.refresh(order)
        return order

    def delete(self, order: OrderModel) -> None:
        """Delete an order (cascades to its items)."""
        self.db.delete(order)
        self.db.commit()
