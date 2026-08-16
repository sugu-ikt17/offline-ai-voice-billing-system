"""Speech Transcription API — POST /api/v1/speech/transcribe.

Accepts a multipart audio upload, sends it through the offline Whisper.cpp
speech engine, and returns a clean transcript.

Response contract (always JSON):
  Success  →  200  {"success": true,  "transcript": "<text>"}
  Unavail  →  503  {"success": false, "message": "Speech engine unavailable"}
  BadInput →  400  {"success": false, "message": "<reason>"}
  Error    →  500  {"success": false, "message": "<reason>"}

No other modules (Menu, Order, Bill, Receipt) are touched by this router.
"""

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.application.services.speech_service import SpeechService
from app.core.exceptions import SpeechRecognitionException
from app.core.logging import get_logger
from app.presentation.dependencies import get_speech_service

logger = get_logger(__name__)

router = APIRouter(prefix="/speech", tags=["Speech"])

# Accepted audio MIME types / extensions from the browser or direct API callers
_ALLOWED_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp3", ".m4a"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB safety cap


# ──────────────────────────────── helpers ─────────────────────────────────────

def _error_response(message: str, http_status: int) -> JSONResponse:
    """Return a JSON envelope that always contains ``success: false``."""
    return JSONResponse(
        status_code=http_status,
        content={"success": False, "message": message},
    )


# ──────────────────────────────── endpoint ────────────────────────────────────

@router.post(
    "/transcribe",
    status_code=status.HTTP_200_OK,
    summary="Transcribe uploaded audio via Faster-Whisper",
    response_description='{"success": true, "transcript": "<text>"}',
)
async def transcribe_audio(
    audio: UploadFile,
    speech_service: SpeechService = Depends(get_speech_service),
) -> JSONResponse:
    """Accept a multipart audio file, run it through Faster-Whisper offline STT,
    and return the recognized text.

    **Field name**: ``audio``  (multipart/form-data)

    **Accepted formats**: wav, webm, ogg, mp3

    **Responses**:
    - ``200`` — ``{"success": true, "transcript": "2 coffee 2 dosa"}``
    - ``400`` — missing file / unsupported format / empty upload
    - ``503`` — Faster-Whisper model not available on this device
    - ``500`` — unexpected server error
    """
    # ── Validate filename / extension ─────────────────────────────────────
    filename = (audio.filename or "").strip()
    if not filename:
        return _error_response("No audio file was provided.", status.HTTP_400_BAD_REQUEST)

    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return _error_response(
            f"Unsupported audio format '{ext or 'unknown'}'. "
            f"Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}.",
            status.HTTP_400_BAD_REQUEST,
        )

    from app.core.config import settings  # noqa: PLC0415
    from app.core.profiler import PipelineProfiler  # noqa: PLC0415

    profiler = PipelineProfiler(enabled=settings.debug)

    # ── Stage 1: Receive Upload ───────────────────────────────────────────
    profiler.start_stage("Receive Upload")
    audio_bytes = await audio.read()
    profiler.end_stage("Receive Upload")

    if not audio_bytes:
        return _error_response(
            "Uploaded audio file is empty.", status.HTTP_400_BAD_REQUEST
        )
    if len(audio_bytes) > _MAX_UPLOAD_BYTES:
        return _error_response(
            f"File too large ({len(audio_bytes) // (1024*1024)} MB). Maximum is 50 MB.",
            status.HTTP_400_BAD_REQUEST,
        )

    logger.info(
        "POST /speech/transcribe — file=%r ext=%s size=%d bytes",
        filename, ext, len(audio_bytes),
    )

    # ── Check engine availability ─────────────────────────────────────────
    engine = speech_service._engine
    if hasattr(engine, "is_available") and not engine.is_available():
        logger.warning("Speech engine unavailable")
        return _error_response(
            "Speech engine unavailable",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # ── Transcribe ────────────────────────────────────────────────────────
    request_start = time.perf_counter()
    try:
        transcript = speech_service.transcribe_upload(audio_bytes, filename, profiler=profiler)
    except SpeechRecognitionException as exc:
        msg = str(exc)
        if "unavailable" in msg.lower() or "not found" in msg.lower():
            logger.warning("Speech engine unavailable: %s", msg)
            return _error_response(
                "Speech engine unavailable",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        logger.error("Transcription error: %s", msg)
        return _error_response(
            f"Transcription failed: {msg}",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    except Exception as exc:
        logger.error("Unexpected error in /speech/transcribe: %s", exc, exc_info=True)
        return _error_response(
            "An unexpected error occurred during transcription.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    total_elapsed = time.perf_counter() - request_start
    logger.info(
        "POST /speech/transcribe complete — transcript=%r total_time=%.2fs",
        transcript[:80],
        total_elapsed,
    )

    # ── Stage 10: Response ────────────────────────────────────────────────
    profiler.start_stage("Response")
    res = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "transcript": transcript},
    )
    profiler.end_stage("Response")

    # Output formatted 10-stage profiling summary
    profiler.print_summary()

    return res
