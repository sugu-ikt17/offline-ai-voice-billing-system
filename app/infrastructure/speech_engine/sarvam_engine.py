"""Sarvam AI Speech-to-Text Engine Integration.

Uses Sarvam AI SaaS Speech-to-Text API (Model: saaras:v3) for runtime transcription.

IMPORTANT ARCHITECTURE & NETWORK NOTE:
-------------------------------------
Unlike local Faster-Whisper, Sarvam AI is an online cloud API endpoint.
It REQUIRES an active internet connection to transcribe audio. If network or API
credentials fail, SpeechRecognitionException will be raised.

Security:
---------
The Sarvam API key is read from configuration (SARVAM_API_KEY) and MUST NEVER
be printed, logged, or returned in API responses.
"""

import os
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import SpeechRecognitionException
from app.core.logging import get_logger

logger = get_logger(__name__)


class SarvamSpeechEngine:
    """Speech recognition engine powered by Sarvam AI Speech-to-Text API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language_code: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        """Initialize Sarvam AI Speech Engine.

        Args:
            api_key: Sarvam API subscription key. If None, read from settings/environment.
            model: STT model ID (default: saaras:v3).
            language_code: Language code for transcription (default: ta-IN).
            mode: Operating mode (default: transcribe).
        """
        self.api_key = api_key if api_key is not None else (getattr(settings, "sarvam_api_key", "") or os.getenv("SARVAM_API_KEY", ""))
        self.model = model or getattr(settings, "sarvam_model", "saaras:v3")
        self.language_code = language_code or getattr(settings, "sarvam_language_code", "ta-IN")
        self.mode = mode or getattr(settings, "sarvam_mode", "transcribe")
        self._client: Any = None

    def is_available(self) -> bool:
        """Return True if API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def is_model_loaded(self) -> bool:
        """Return True if SarvamAI client has been initialized."""
        return self._client is not None

    def load_model(self) -> None:
        """Initialize the SarvamAI API client.

        Raises:
            SpeechRecognitionException: If API key is missing or client instantiation fails.
        """
        if not self.api_key or not self.api_key.strip():
            err_msg = "Sarvam API key (SARVAM_API_KEY) is missing or empty."
            logger.error("Sarvam engine startup failed: API key missing.")
            raise SpeechRecognitionException(err_msg)

        if self.is_model_loaded():
            logger.debug("SarvamAI client is already initialized.")
            return

        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            logger.info(
                "Initializing SarvamAI engine client (model=%s, language=%s, mode=%s)...",
                self.model,
                self.language_code,
                self.mode,
            )
            self._client = SarvamAI(api_subscription_key=self.api_key)
            logger.info("SarvamAI engine client successfully initialized.")
        except Exception as exc:
            logger.error("Failed to initialize SarvamAI client: %s", exc)
            raise SpeechRecognitionException(f"Failed to initialize SarvamAI client: {exc}") from exc

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe an audio file using Sarvam AI Speech-to-Text API.

        Args:
            audio_file_path: Absolute or relative path to the preprocessed audio file.

        Returns:
            str: Raw recognized transcript text.

        Raises:
            SpeechRecognitionException: If audio file is missing or API request fails.
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise SpeechRecognitionException(f"Audio file not found at: {audio_file_path}")

        if not self.is_model_loaded():
            self.load_model()

        if self._client is None:
            raise SpeechRecognitionException("SarvamAI client is not initialized.")

        try:
            logger.info(
                "Sending audio to Sarvam AI STT (file=%s, model=%s, language=%s, mode=%s)...",
                audio_path.name,
                self.model,
                self.language_code,
                self.mode,
            )
            with open(audio_path, "rb") as audio_file:
                response = self._client.speech_to_text.transcribe(
                    file=audio_file,
                    model=self.model,
                    mode=self.mode,
                    language_code=self.language_code,
                )

            # Support both Pydantic model response and dict response
            if hasattr(response, "transcript"):
                transcript = response.transcript
            elif isinstance(response, dict):
                transcript = response.get("transcript", "")
            else:
                transcript = str(response)

            transcript_str = transcript.strip() if transcript else ""
            return transcript_str

        except SpeechRecognitionException:
            raise
        except Exception as exc:
            # Ensure API key is never logged or exposed
            err_str = str(exc)
            if self.api_key and self.api_key in err_str:
                err_str = err_str.replace(self.api_key, "[REDACTED]")
            logger.error("Sarvam AI transcription failed: %s", err_str)
            raise SpeechRecognitionException(f"Sarvam AI transcription failed: {err_str}") from exc
