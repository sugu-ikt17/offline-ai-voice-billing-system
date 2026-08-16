"""Pydantic schemas for the Bill API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BillRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    order_id: int
    subtotal: float
    tax_rate: float
    tax_amount: float
    total: float
    generated_at: datetime
