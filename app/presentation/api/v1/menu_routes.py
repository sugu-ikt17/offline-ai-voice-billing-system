"""Menu API routes — CRUD endpoints for menu items.

Routes only handle HTTP concerns (status codes, exception translation).
All business logic lives in MenuService.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.application.services.menu_service import MenuService
from app.core.dependencies import get_menu_service
from app.core.exceptions import DuplicateException, NotFoundException, ValidationException
from app.presentation.schemas.menu_schema import MenuCreate, MenuResponse, MenuUpdate

router = APIRouter(prefix="/menu", tags=["Menu"])


@router.post("", response_model=MenuResponse, status_code=status.HTTP_201_CREATED)
def create_menu(
    payload: MenuCreate,
    service: MenuService = Depends(get_menu_service),
):
    try:
        return service.create_menu(payload.name, payload.price)
    except (ValidationException, DuplicateException) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("", response_model=list[MenuResponse])
def list_menu(service: MenuService = Depends(get_menu_service)):
    return service.get_all_menu()


@router.get("/{item_id}", response_model=MenuResponse)
def get_menu(item_id: int, service: MenuService = Depends(get_menu_service)):
    try:
        return service.get_menu_by_id(item_id)
    except NotFoundException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.put("/{item_id}", response_model=MenuResponse)
def update_menu(
    item_id: int,
    payload: MenuUpdate,
    service: MenuService = Depends(get_menu_service),
):
    try:
        return service.update_menu(item_id, payload.name, payload.price)
    except NotFoundException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except (ValidationException, DuplicateException) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_menu(item_id: int, service: MenuService = Depends(get_menu_service)):
    try:
        service.delete_menu(item_id)
    except NotFoundException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
