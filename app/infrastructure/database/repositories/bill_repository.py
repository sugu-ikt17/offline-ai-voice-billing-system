"""Bill repository — database operations for bills."""

from typing import Optional

from sqlalchemy.orm import Session

from app.infrastructure.database.models.bill_model import BillModel


class BillRepository:
    """Data access layer for bills."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, bill: BillModel) -> BillModel:
        """Persist a new bill and return it refreshed."""
        self.db.add(bill)
        self.db.commit()
        self.db.refresh(bill)
        return bill

    def get_all(self) -> list[BillModel]:
        """Return all bills."""
        return self.db.query(BillModel).order_by(BillModel.generated_at.desc()).all()

    def get_by_id(self, bill_id: int) -> Optional[BillModel]:
        """Return a bill by id, or None if not found."""
        return self.db.query(BillModel).filter(BillModel.id == bill_id).first()

    def get_by_order_id(self, order_id: int) -> Optional[BillModel]:
        """Return the bill generated for a given order, or None."""
        return self.db.query(BillModel).filter(BillModel.order_id == order_id).first()
