"""Faster-Whisper engine wrapper (formerly Whisper.cpp).

Replaces Whisper.cpp subprocess execution with FasterWhisperEngine for
backward compatibility with existing imports.
"""

from typing import Optional
from app.infrastructure.speech_engine.faster_whisper_engine import FasterWhisperEngine


class WhisperCppEngine(FasterWhisperEngine):
    """Backward compatibility wrapper around FasterWhisperEngine."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        model_path: Optional[str] = None,
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        super().__init__(model_name=model_name, device=device, compute_type=compute_type)
        self.binary_path = binary_path or ""
        self.model_path = model_path or ""
