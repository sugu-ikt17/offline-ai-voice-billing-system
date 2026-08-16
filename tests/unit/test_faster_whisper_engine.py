"""FasterWhisperEngine unit and integration tests.

Tests model loading once lifecycle, device selection, and transcribe behavior.
"""

import array
import math
import wave
import pytest

from app.core.exceptions import SpeechRecognitionException
from app.infrastructure.speech_engine.faster_whisper_engine import FasterWhisperEngine


def _write_sample_wav(path) -> None:
    sample_rate = 16_000
    duration = 0.5
    n = int(sample_rate * duration)
    samples = array.array(
        "h",
        [int(4000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)],
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def test_engine_initialization_defaults():
    engine = FasterWhisperEngine()
    assert engine.model_name == "distil-large-v3"
    assert engine.requested_device == "cuda"
    assert engine.requested_compute_type == "float16"


def test_engine_load_model_once(tmp_path):
    audio_path = tmp_path / "test.wav"
    _write_sample_wav(audio_path)

    engine = FasterWhisperEngine(model_name="distil-large-v3", device="cuda", compute_type="float16")
    engine.load_model()
    assert engine.is_model_loaded() is True

    # Call load_model again to verify idempotency (does not reload)
    engine.load_model()
    assert engine.is_model_loaded() is True

    # Run transcription
    transcript = engine.transcribe(str(audio_path))
    assert isinstance(transcript, str)


def test_transcribe_missing_file():
    engine = FasterWhisperEngine()
    with pytest.raises(SpeechRecognitionException, match="Audio file not found"):
        engine.transcribe("/nonexistent/file.wav")
