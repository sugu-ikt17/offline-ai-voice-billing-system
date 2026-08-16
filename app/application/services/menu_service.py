"""Menu service — business rules and validation around menu items.

Contains all menu-related business logic. Persistence is delegated
entirely to the injected MenuRepository.
"""

from app.core.exceptions import DuplicateException, NotFoundException, ValidationException
from app.infrastructure.database.models.menu_item_model import MenuItemModel
from app.infrastructure.database.repositories.menu_repository import MenuRepository


class MenuService:
    """Business logic layer for managing menu items."""

    def __init__(self, menu_repository: MenuRepository) -> None:
        self.menu_repository = menu_repository

    def create_menu(self, name: str, price: float) -> MenuItemModel:
        """Validate and create a new menu item.

        Raises:
            ValidationException: if name is empty or price is not > 0.
            DuplicateException: if a menu item with this name already exists.
        """
        name = name.strip() if name else ""
        if not name:
            raise ValidationException("Menu name must not be empty.")
        if price <= 0:
            raise ValidationException("Price must be greater than zero.")
        if self.menu_repository.get_by_name(name):
            raise DuplicateException(f"Menu item '{name}' already exists.")

        return self.menu_repository.create(MenuItemModel(name=name, price=price))

    def get_all_menu(self) -> list[MenuItemModel]:
        """Return all menu items."""
        return self.menu_repository.get_all()

    def get_menu_by_id(self, item_id: int) -> MenuItemModel:
        """Return a menu item by id.

        Raises:
            NotFoundException: if no menu item exists with this id.
        """
        item = self.menu_repository.get_by_id(item_id)
        if item is None:
            raise NotFoundException(f"Menu item with id {item_id} not found.")
        return item

    def update_menu(
        self, item_id: int, name: str | None = None, price: float | None = None
    ) -> MenuItemModel:
        """Validate and update an existing menu item's name and/or price.

        Raises:
            NotFoundException: if the item does not exist.
            ValidationException: if the new name is empty or price is not > 0.
            DuplicateException: if the new name collides with another item.
        """
        item = self.get_menu_by_id(item_id)

        if name is not None:
            name = name.strip()
            if not name:
                raise ValidationException("Menu name must not be empty.")
            if name != item.name:
                existing = self.menu_repository.get_by_name(name)
                if existing and existing.id != item_id:
                    raise DuplicateException(f"Menu item '{name}' already exists.")
                item.name = name

        if price is not None:
            if price <= 0:
                raise ValidationException("Price must be greater than zero.")
            item.price = price

        return self.menu_repository.update(item)

    def delete_menu(self, item_id: int) -> None:
        """Delete a menu item by id.

        Raises:
            NotFoundException: if the item does not exist.
        """
        item = self.get_menu_by_id(item_id)
        self.menu_repository.delete(item)
