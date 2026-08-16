"""
Menu repository.

Handles direct database access for MenuItemModel. Contains no
business logic or validation — only persistence operations.
"""

from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.infrastructure.database.models.menu_item_model import MenuItemModel


class MenuRepository:
    """Data access layer for menu items."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, menu_item: MenuItemModel) -> MenuItemModel:
        """Persist a new menu item and return it with refreshed fields."""
        self.db.add(menu_item)
        self.db.commit()
        self.db.refresh(menu_item)
        return menu_item

    def get_all(self) -> list[MenuItemModel]:
        """Return all menu items."""
        return self.db.query(MenuItemModel).all()

    def get_by_id(self, item_id: int) -> Optional[MenuItemModel]:
        """Return a menu item by its id, or None if not found."""
        return (
            self.db.query(MenuItemModel)
            .filter(MenuItemModel.id == item_id)
            .first()
        )

    def get_by_name(self, name: str) -> Optional[MenuItemModel]:
        """Return a menu item by its name (case-insensitive), or None if not found."""
        return (
            self.db.query(MenuItemModel)
            .filter(func.lower(MenuItemModel.name) == name.lower())
            .first()
        )

    def update(self, menu_item: MenuItemModel) -> MenuItemModel:
        """Commit changes made to an existing menu item and refresh it."""
        self.db.commit()
        self.db.refresh(menu_item)
        return menu_item

    def delete(self, menu_item: MenuItemModel) -> None:
        """Delete a menu item from the database."""
        self.db.delete(menu_item)
        self.db.commit()
