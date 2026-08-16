"""Pydantic schemas for the Order API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProcessTextRequest(BaseModel):
    """Request body for POST /orders/process.

    BUG-05 FIX: Moved here from the route file so all schemas live in one
    consistent location rather than being defined inline in the route module.
    """

    speech: str = Field(
        default="",
        description="Raw speech transcript to process through the full billing pipeline.",
        examples=["2 dosa 1 tea"],
    )


class OrderItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    menu_item_id: int
    name: str
    quantity: int
    unit_price: float


class OrderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    raw_transcript: str | None
    created_at: datetime
    items: list[OrderItemRead]
