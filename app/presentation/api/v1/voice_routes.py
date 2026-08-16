"""Voice Upload API — transcribes an uploaded audio file to text.

This endpoint only performs speech-to-text. It does not parse the
transcript into order items, match menu items, or generate a bill —
those are separate concerns handled elsewhere in the pipeline.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.application.services.speech_service import SpeechService
from app.core.config import settings
from app.core.exceptions import SpeechRecognitionException

router = APIRouter(prefix="/voice", tags=["Voice"])

_ALLOWED_EXTENSIONS = {".wav"}


def get_speech_service() -> SpeechService:
    """Local dependency provider — builds a SpeechService from configuration."""
    return SpeechService()


@router.post("/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_audio(
    audio_file: UploadFile,
    speech_service: SpeechService = Depends(get_speech_service),
):
    """Accept a WAV file, transcribe it, and return the recognized text."""
    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file was provided.",
        )

    extension = Path(audio_file.filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{extension or 'unknown'}'. Only .wav is supported.",
        )

    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    upload_dir = Path(settings.audio_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{uuid.uuid4().hex}{extension}"

    try:
        saved_path.write_bytes(audio_bytes)
        recognized_text = speech_service.transcribe(str(saved_path))
        return {"recognized_text": recognized_text}

    except SpeechRecognitionException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Speech recognition failed: {exc}",
        )
    finally:
        saved_path.unlink(missing_ok=True)
