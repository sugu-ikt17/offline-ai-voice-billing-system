"""Reusable FastAPI dependency providers.

Wires the object graph for routes: Session -> MenuRepository -> MenuService.
Routes should depend on get_menu_service (or get_menu_repository, if they
ever need repository-level access directly) rather than constructing
these objects themselves.
"""

from fastapi import Depends

from app.application.services.menu_service import MenuService
# Re-exported here rather than redefined, so there is a single source of
# truth for session creation/teardown.
from app.infrastructure.database.database import get_db  # noqa: F401
from app.infrastructure.database.repositories.menu_repository import MenuRepository
from sqlalchemy.orm import Session


def get_menu_repository(db: Session = Depends(get_db)) -> MenuRepository:
    """Provide a MenuRepository bound to the current request's session."""
    return MenuRepository(db)


def get_menu_service(
    menu_repository: MenuRepository = Depends(get_menu_repository),
) -> MenuService:
    """Provide a MenuService with its repository dependency injected."""
    return MenuService(menu_repository)
