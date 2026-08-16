"""MenuItem domain entity — framework-agnostic representation of a menu item."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MenuItem:
    name: str
    price: float
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
