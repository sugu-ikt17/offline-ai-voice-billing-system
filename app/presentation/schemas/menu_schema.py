"""Pydantic schemas for Menu request/response payloads."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MenuCreate(BaseModel):
    """Payload for creating a new menu item."""

    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0)


class MenuUpdate(BaseModel):
    """Payload for updating an existing menu item. All fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    price: float | None = Field(default=None, gt=0)


class MenuResponse(BaseModel):
    """Response representation of a menu item."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    price: float
    created_at: datetime
    updated_at: datetime
