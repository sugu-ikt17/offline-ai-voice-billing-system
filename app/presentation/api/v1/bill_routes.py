"""Bill API routes — generate and retrieve bills.

Endpoints:
    POST /bills/{order_id}  — generate a bill for a pending order
    GET  /bills/{bill_id}   — retrieve a saved bill by ID
    GET  /bills             — list all bills (most recent first)
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.application.use_cases.generate_bill import GenerateBillUseCase
from app.core.exceptions import NotFoundException, ValidationException
from app.infrastructure.database.repositories.bill_repository import BillRepository
from app.presentation.api.schemas.bill_schema import BillRead
from app.presentation.dependencies import get_bill_repository, get_generate_bill_use_case

router = APIRouter(prefix="/bills", tags=["Bills"])


@router.get("", response_model=list[BillRead])
def list_bills(repository: BillRepository = Depends(get_bill_repository)):
    """Return all bills, most recent first.

    BUG-04 FIX: This endpoint was missing — the router only had
    GET /bills/{bill_id} and POST /bills/{order_id} with no list view.
    """
    return repository.get_all()


@router.post("/{order_id}", response_model=BillRead, status_code=status.HTTP_201_CREATED)
def generate_bill(
    order_id: int, use_case: GenerateBillUseCase = Depends(get_generate_bill_use_case)
):
    """Generate a bill for a pending order and persist it.

    The order must exist and must not already have been billed.
    Returns the persisted BillModel with computed totals.
    """
    try:
        return use_case.execute(order_id)
    except NotFoundException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValidationException as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/{bill_id}", response_model=BillRead)
def get_bill(
    bill_id: int, repository: BillRepository = Depends(get_bill_repository)
):
    """Retrieve a previously generated bill by its ID."""
    bill = repository.get_by_id(bill_id)
    if bill is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bill with id {bill_id} not found.",
        )
    return bill
