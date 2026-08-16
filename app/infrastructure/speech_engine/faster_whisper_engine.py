"""Faster-Whisper integration.

Wraps the Faster-Whisper Python API (CTranslate2 backend).
Runs fully offline with GPU (CUDA) support and automatic CPU fallback.
Loads the model only once during startup and reuses it across requests.
"""

import time
from pathlib import Path
from typing import Optional

import ctranslate2
from faster_whisper import WhisperModel

from app.core.exceptions import SpeechRecognitionException
from app.core.logging import get_logger

logger = get_logger(__name__)


class FasterWhisperEngine:
    """Offline STT engine using Faster-Whisper Python API."""

    # Class-level model cache to guarantee the model is loaded only once
    _shared_model: Optional[WhisperModel] = None
    _loaded_model_name: Optional[str] = None
    _active_device: Optional[str] = None
    _active_compute_type: Optional[str] = None

    def __init__(
        self,
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        language: str | None = "ta",
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.beam_size = beam_size
        # Map empty string / 'auto' to None (Faster-Whisper auto-detect)
        self.language = language if language and language != "auto" else None

    def is_available(self) -> bool:
        """Return True if model is already loaded or can be loaded."""
        return True

    def is_model_loaded(self) -> bool:
        """Return whether the WhisperModel is currently loaded in memory."""
        return (
            FasterWhisperEngine._shared_model is not None
            and FasterWhisperEngine._loaded_model_name == self.model_name
        )

    def load_model(self) -> None:
        """Load the model into memory only once.

        Tries GPU (CUDA, float16) first if requested/available, falling back to CPU.
        Uses local_files_only=True to ensure no internet downloads occur.

        Raises:
            SpeechRecognitionException: if loading fails on both CUDA and CPU.
        """
        if self.is_model_loaded():
            logger.debug("Faster-Whisper model '%s' is already loaded.", self.model_name)
            return

        target_device = self.requested_device
        target_compute_type = self.requested_compute_type

        # Check CUDA availability
        if target_device == "cuda":
            try:
                cuda_count = ctranslate2.get_cuda_device_count()
            except Exception:
                cuda_count = 0

            if cuda_count == 0:
                logger.warning(
                    "CUDA requested for Faster-Whisper but CUDA device count is 0. Falling back to CPU."
                )
                target_device = "cpu"
                target_compute_type = "default"

        logger.info(
            "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s)...",
            self.model_name,
            target_device,
            target_compute_type,
        )

        model = None
        # Primary load attempt (CUDA or requested device)
        try:
            model = WhisperModel(
                self.model_name,
                device=target_device,
                compute_type=target_compute_type,
                local_files_only=True,
            )
        except Exception as exc:
            if target_device == "cuda":
                logger.warning(
                    "Failed to load Faster-Whisper model on CUDA (%s). Retrying fallback to CPU...",
                    exc,
                )
                target_device = "cpu"
                target_compute_type = "default"
                try:
                    model = WhisperModel(
                        self.model_name,
                        device=target_device,
                        compute_type=target_compute_type,
                        local_files_only=True,
                    )
                except Exception as cpu_exc:
                    err_msg = f"Failed to load Faster-Whisper model '{self.model_name}' on CPU fallback: {cpu_exc}"
                    logger.error("%s", err_msg)
                    raise SpeechRecognitionException(err_msg) from cpu_exc
            else:
                err_msg = f"Failed to load Faster-Whisper model '{self.model_name}': {exc}"
                logger.error("%s", err_msg)
                raise SpeechRecognitionException(err_msg) from exc

        FasterWhisperEngine._shared_model = model
        FasterWhisperEngine._loaded_model_name = self.model_name
        FasterWhisperEngine._active_device = target_device
        FasterWhisperEngine._active_compute_type = target_compute_type

        logger.info(
            "Faster-Whisper model '%s' successfully loaded on %s (%s).",
            self.model_name,
            target_device,
            target_compute_type,
        )

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe an audio file using Faster-Whisper Python API.

        Automatic language detection is enabled (language=None).

        Returns:
            str: clean recognized text

        Raises:
            SpeechRecognitionException: if audio file does not exist or inference fails.
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise SpeechRecognitionException(f"Audio file not found at: {audio_file_path}")

        if not self.is_model_loaded():
            self.load_model()

        model = FasterWhisperEngine._shared_model
        if model is None:
            raise SpeechRecognitionException("Faster-Whisper model is not initialized.")

        try:
            # Runtime Execution Device Verification
            cuda_count = 0
            try:
                cuda_count = ctranslate2.get_cuda_device_count()
            except Exception:
                pass
            cuda_available = cuda_count > 0

            inner_model = getattr(model, "model", None)
            model_device = getattr(inner_model, "device", FasterWhisperEngine._active_device or "unknown")
            exec_device = str(model_device)

            device_log_before = (
                f"\n========================================\n"
                f"FASTER-WHISPER RUNTIME EXECUTION VERIFICATION (BEFORE INFERENCE)\n"
                f"========================================\n"
                f"- Model Device      : {model_device}\n"
                f"- CUDA Available    : {cuda_available} (device_count={cuda_count})\n"
                f"- CUDA Device Name  : N/A (Driver error / CPU fallback)\n"
                f"- GPU Memory Before : N/A (CUDA unavailable)\n"
                f"- Execution Device  : {exec_device}\n"
                f"========================================"
            )
            print(device_log_before)
            logger.info(device_log_before)

            # Use configured beam_size and language for optimized inference.
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=self.beam_size,
                language=self.language,
            )
            text_parts = [segment.text.strip() for segment in segments if segment.text and segment.text.strip()]
            transcript = " ".join(text_parts).strip()

            device_log_after = (
                f"\n========================================\n"
                f"FASTER-WHISPER RUNTIME EXECUTION VERIFICATION (AFTER INFERENCE)\n"
                f"========================================\n"
                f"- Model Device      : {model_device}\n"
                f"- CUDA Available    : {cuda_available} (device_count={cuda_count})\n"
                f"- CUDA Device Name  : N/A (Driver error / CPU fallback)\n"
                f"- GPU Memory After  : N/A (CUDA unavailable)\n"
                f"- Execution Device  : {exec_device}\n"
                f"========================================"
            )
            print(device_log_after)
            logger.info(device_log_after)

            logger.debug(
                "Faster-Whisper inference: beam_size=%d, language=%r, detected_language=%r, prob=%.2f",
                self.beam_size,
                self.language,
                getattr(info, 'language', '?'),
                getattr(info, 'language_probability', 0.0),
            )
            return transcript
        except Exception as exc:
            logger.error("Faster-Whisper transcription failed: %s", exc, exc_info=True)
            raise SpeechRecognitionException(f"Transcription process failed: {exc}") from exc
