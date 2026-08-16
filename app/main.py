"""Application entry point — FastAPI app assembly."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.application.services.speech_service import SpeechService
from app.core.config import settings
from app.core.exceptions import SpeechRecognitionException
from app.core.logging import configure_logging, get_logger
from app.infrastructure.database.database import Base, engine
# Models must be imported so they register on Base.metadata before create_all().
from app.infrastructure.database.models import bill_model, menu_item_model, order_model  # noqa: F401
from app.presentation.api.v1.router import api_router

configure_logging()
logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creating database tables (if not present)...")
    Base.metadata.create_all(bind=engine)

    # Fail fast (but don't crash the app) if whisper.cpp isn't set up yet —
    # menu management and every other route should keep working even if
    # voice ordering isn't ready on this device.
    try:
        from app.presentation.dependencies import init_speech_service  # noqa: PLC0415
        init_speech_service()
        logger.info("Speech engine ready.")
    except SpeechRecognitionException as exc:
        logger.error("[Speech Engine Startup Validation Failed] %s", exc)

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Offline AI Voice Billing System",
    description=(
        "An offline, voice-driven billing API for small shops. Runs fully "
        "on-device (Raspberry Pi Zero 2 W) — menu management, voice-based "
        "ordering via Whisper.cpp, and bill generation, with no internet "
        "dependency."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allows a separately-served frontend (e.g. a dev server on another port)
# to call this API during development. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/")
def read_root() -> dict[str, str]:
    """Simple health/identity check for the API."""
    return {"message": "Offline AI Voice Billing API"}


# Static frontend is served under /app so it doesn't shadow the root endpoint above.
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
