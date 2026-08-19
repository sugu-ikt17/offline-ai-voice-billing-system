"""Order processing routes.

Endpoint summary:
  GET  /orders              — list all saved orders
  GET  /orders/{id}         — retrieve a saved order by ID
  POST /orders/process      — parse speech → match menu → generate bill
"""

from datetime import datetime, timezone
import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.application.services.bill_generator_service import BillGeneratorService
from app.application.services.menu_matcher_service import MenuMatcherService
from app.application.services.order_parser_service import OrderParserService
from app.core.exceptions import NotFoundException
from app.core.logging import get_logger
from app.domain.entities.bill import BillResult
from app.infrastructure.database.database import get_db
from app.infrastructure.database.repositories.menu_repository import MenuRepository
from app.infrastructure.database.repositories.order_repository import OrderRepository
from app.presentation.api.schemas.order_schema import OrderRead, ProcessTextRequest  # BUG-05 FIX

logger = get_logger(__name__)

router = APIRouter(prefix="/orders", tags=["Orders"])


def get_order_repo(db: Session = Depends(get_db)) -> OrderRepository:
    return OrderRepository(db)


@router.get("", response_model=list[OrderRead])
def list_orders(repository: OrderRepository = Depends(get_order_repo)):
    """Return all saved orders, most recent first."""
    return repository.get_all()


@router.get("/{order_id}", response_model=OrderRead)
def get_order(order_id: int, repository: OrderRepository = Depends(get_order_repo)):
    """Return a single saved order by ID."""
    order = repository.get_by_id(order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with id {order_id} not found.",
        )
    return order


@router.post("/process", status_code=status.HTTP_200_OK)
def process_order_text(
    payload: ProcessTextRequest,
    db: Session = Depends(get_db),
):
    """Full voice-to-bill pipeline in a single call.

    Request body:
        { "speech": "2 dosa 1 tea" }

    Pipeline steps:
        1. OrderParserService  — tokenise speech into {item, quantity} pairs.
        2. MenuMatcherService  — resolve each token to a real menu entry.
        3. BillGeneratorService — compute line totals, aggregates, bill number.

    Response shape (see BillResult.to_dict()):
        {
          "bill": {
            "bill_number":    "BILL-3F7A2C1E",
            "date_time":      "2025-07-23T06:30:00+00:00",
            "items":          [ {name, quantity, unit_price, subtotal}, … ],
            "item_count":     2,
            "total_quantity": 3,
            "subtotal":       95.0,
            "discount":       0.0,
            "tax":            0.0,
            "grand_total":    95.0
          },
          "warnings":        [],
          "unmatched_items": []
        }
    """
    text = (payload.speech or "").strip()

    if not text:
        # Return an empty-bill structure without spinning up the pipeline.
        return BillResult(
            bill_number     = "BILL-00000000",
            date_time       = datetime.now(timezone.utc),
            items           = [],
            item_count      = 0,
            total_quantity  = 0,
            subtotal        = 0.0,
            discount        = 0.0,
            tax             = 0.0,
            grand_total     = 0.0,
            warnings        = [],
            unmatched_items = [],
        ).to_dict()

    from app.core.config import settings  # noqa: PLC0415
    from app.core.profiler import PipelineProfiler  # noqa: PLC0415

    profiler = PipelineProfiler(enabled=settings.debug)

    request_start = time.perf_counter()

    # Stage 9: Parser
    profiler.start_stage("Parser")
    t0 = time.perf_counter()

    from app.application.services.speech_service import _log_transcript_stage  # noqa: PLC0415
    _log_transcript_stage("PARSER INPUT", text)

    # Pass vocabulary to parser for multiword and item boundary detection
    menu_repo = MenuRepository(db)
    menu_items = menu_repo.get_all()
    vocab_list = [m.name for m in menu_items] if menu_items else None

    parsed_result = OrderParserService().parse_with_details(text, vocabulary=vocab_list)
    parsed_items = parsed_result.recognized_items
    t_parser = time.perf_counter() - t0
    profiler.end_stage("Parser")

    _log_transcript_stage("PARSER OUTPUT", str(parsed_items))

    # Step 2 — match tokens against the menu database.
    t0 = time.perf_counter()
    match_result = MenuMatcherService(menu_repo).match(parsed_items)
    t_matcher = time.perf_counter() - t0

    _log_transcript_stage("MATCHER OUTPUT", f"Matched: {match_result.matched_items}\nUnmatched: {match_result.unmatched_items}")

    # Step 3 — generate the final bill.
    t0 = time.perf_counter()
    bill_result = BillGeneratorService().generate(match_result)
    t_generator = time.perf_counter() - t0

    res_dict = bill_result.to_dict()
    _log_transcript_stage("FINAL BILL", str(res_dict["bill"]))

    # Stage 10: Response
    profiler.start_stage("Response")
    profiler.end_stage("Response")

    total_order_time = time.perf_counter() - request_start
    logger.info(
        "\n========================================\n"
        "ORDER PROCESSING TIMINGS\n"
        "========================================\n"
        "PARSER : %.3fs\n"
        "MATCHER: %.3fs\n"
        "BILL   : %.3fs\n"
        "TOTAL  : %.3fs\n"
        "========================================",
        t_parser,
        t_matcher,
        t_generator,
        total_order_time,
    )

    return res_dict
