"""FastAPI dependency providers.

Central place where the object graph is wired together: sessions ->
repositories -> services -> use cases. Routes only ever depend on
functions from this module, never construct these objects themselves.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.bill_generator_service import BillGeneratorService
from app.application.services.menu_matcher_service import MenuMatcherService
from app.application.services.order_parser_service import OrderParserService
from app.application.services.speech_service import SpeechService
from app.application.use_cases.generate_bill import GenerateBillUseCase
from app.application.use_cases.process_voice_order import ProcessVoiceOrderUseCase
from app.core.config import settings
from app.core.dependencies import get_menu_repository  # single source of truth for menu DI
from app.infrastructure.database.database import get_db
from app.infrastructure.database.repositories.bill_repository import BillRepository
from app.infrastructure.database.repositories.menu_repository import MenuRepository
from app.infrastructure.database.repositories.order_repository import OrderRepository
from app.infrastructure.speech_engine.faster_whisper_engine import FasterWhisperEngine


# --- Repositories -----------------------------------------------------

def get_order_repository(db: Session = Depends(get_db)) -> OrderRepository:
    return OrderRepository(db)


def get_bill_repository(db: Session = Depends(get_db)) -> BillRepository:
    return BillRepository(db)


# --- Services -----------------------------------------------------------

_SPEECH_SERVICE_SINGLETON: SpeechService | None = None


def init_speech_service(menu_repository: MenuRepository | None = None) -> SpeechService:
    """Pre-initialize and pre-load the singleton SpeechService during startup."""
    global _SPEECH_SERVICE_SINGLETON
    if _SPEECH_SERVICE_SINGLETON is None:
        engine = FasterWhisperEngine(
            model_name=settings.whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language or None,
        )
        _SPEECH_SERVICE_SINGLETON = SpeechService(engine=engine, menu_repository=menu_repository)
    _SPEECH_SERVICE_SINGLETON.load_model()
    return _SPEECH_SERVICE_SINGLETON


def get_speech_service(
    menu_repository: MenuRepository = Depends(get_menu_repository),
) -> SpeechService:
    global _SPEECH_SERVICE_SINGLETON
    if _SPEECH_SERVICE_SINGLETON is None:
        engine = FasterWhisperEngine(
            model_name=settings.whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language or None,
        )
        _SPEECH_SERVICE_SINGLETON = SpeechService(engine=engine, menu_repository=menu_repository)
    else:
        _SPEECH_SERVICE_SINGLETON._menu_repository = menu_repository
    return _SPEECH_SERVICE_SINGLETON


def get_order_parser_service() -> OrderParserService:
    return OrderParserService()


def get_menu_matcher_service(
    menu_repository: MenuRepository = Depends(get_menu_repository),
) -> MenuMatcherService:
    return MenuMatcherService(menu_repository)


def get_bill_generator_service() -> BillGeneratorService:
    # BUG-02 FIX: BillGeneratorService accepts `discount` and `tax` (currency
    # amounts), not `tax_rate`. Tax is 0 per current sprint scope; GST
    # percentage logic will be added in a later sprint.
    return BillGeneratorService()


# --- Use cases ------------------------------------------------------------

def get_process_voice_order_use_case(
    speech_service: SpeechService = Depends(get_speech_service),
    order_parser: OrderParserService = Depends(get_order_parser_service),
    menu_matcher: MenuMatcherService = Depends(get_menu_matcher_service),
    order_repository: OrderRepository = Depends(get_order_repository),
) -> ProcessVoiceOrderUseCase:
    return ProcessVoiceOrderUseCase(
        speech_service, order_parser, menu_matcher, order_repository
    )


def get_generate_bill_use_case(
    bill_generator: BillGeneratorService = Depends(get_bill_generator_service),
    order_repository: OrderRepository = Depends(get_order_repository),
    bill_repository: BillRepository = Depends(get_bill_repository),
) -> GenerateBillUseCase:
    return GenerateBillUseCase(bill_generator, order_repository, bill_repository)
