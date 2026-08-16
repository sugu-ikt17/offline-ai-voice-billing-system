"""Process voice order use case.

Orchestrates the full pipeline: audio -> transcript -> parsed items ->
matched menu items -> persisted order. This is the only place that
coordinates all four AI-pipeline services together.

BUG-03 FIX:
    The original code treated `matched_items` (which is a `MatchResult`
    dataclass) as if it were a plain list:
        - `if not matched_items:` — a MatchResult instance is always truthy
          regardless of whether it has items, so this guard never fired.
        - `for item in matched_items:` — MatchResult is not iterable;
          this would raise TypeError at runtime.

    Fix: use `match_result.matched_items` (the actual list inside MatchResult).
"""

from app.core.exceptions import ValidationException
from app.domain.entities.order import OrderStatus
from app.domain.interfaces.menu_matcher_interface import MenuMatcherInterface
from app.domain.interfaces.order_parser_interface import OrderParserInterface
from app.domain.interfaces.speech_service_interface import SpeechServiceInterface
from app.infrastructure.database.models.order_model import OrderItemModel, OrderModel
from app.infrastructure.database.repositories.order_repository import OrderRepository


class ProcessVoiceOrderUseCase:
    def __init__(
        self,
        speech_service: SpeechServiceInterface,
        order_parser: OrderParserInterface,
        menu_matcher: MenuMatcherInterface,
        order_repository: OrderRepository,
    ) -> None:
        self.speech_service = speech_service
        self.order_parser = order_parser
        self.menu_matcher = menu_matcher
        self.order_repository = order_repository

    def execute(self, audio_file_path: str) -> OrderModel:
        # Step 1 — transcribe audio to text
        transcript = self.speech_service.transcribe(audio_file_path)

        # Step 2 — parse transcript into {item, quantity} tokens
        parsed_items = self.order_parser.parse(transcript)
        if not parsed_items:
            raise ValidationException("Could not understand any items in the recording.")

        # Step 3 — match tokens against the menu DB; returns a MatchResult
        match_result = self.menu_matcher.match(parsed_items)

        # BUG-03 FIX: `match_result` is a MatchResult dataclass; access the
        # inner list `.matched_items` rather than iterating MatchResult itself.
        if not match_result.matched_items:
            raise ValidationException("None of the spoken items match the menu.")

        order = OrderModel(
            status=OrderStatus.PENDING.value,
            raw_transcript=transcript,
            items=[
                OrderItemModel(
                    menu_item_id=item.menu_item_id,
                    name=item.name,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                )
                for item in match_result.matched_items  # BUG-03 FIX: was `matched_items`
            ],
        )
        return self.order_repository.create(order)
