"""SpeechService tests.

Uses a fake engine matching FasterWhisperEngine's interface so these
tests don't depend on heavy AI models during unit test execution.
"""

import array
import math
import wave
import pytest

from app.application.services.speech_service import SpeechService
from app.core.exceptions import SpeechRecognitionException
from app.infrastructure.speech_engine.faster_whisper_engine import FasterWhisperEngine


def _write_minimal_wav(path) -> None:
    """Write a 0.1-second 16 kHz mono 16-bit sine-wave WAV file."""
    sample_rate = 16_000
    duration = 0.1
    n = int(sample_rate * duration)
    samples = array.array(
        "h",
        [int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)],
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class FakeEngine:
    """Stand-in for FasterWhisperEngine — same shape, no heavy model loading."""

    def __init__(self, model_name: str = "distil-large-v3", transcript: str = "two idli", fail_load: bool = False) -> None:
        self.model_name = model_name
        self._transcript = transcript
        self.transcribe_called_with: str | None = None
        self._loaded = False
        self._fail_load = fail_load

    def is_available(self) -> bool:
        return True

    def is_model_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        if self._fail_load:
            raise SpeechRecognitionException("Failed to load model")
        self._loaded = True

    def transcribe(self, audio_file_path: str) -> str:
        self.transcribe_called_with = audio_file_path
        return self._transcript


@pytest.fixture()
def fake_paths(tmp_path):
    audio = tmp_path / "order.wav"
    _write_minimal_wav(audio)
    return audio


def test_is_model_loaded_false_before_load():
    engine = FakeEngine()
    service = SpeechService(engine)

    assert service.is_model_loaded() is False


def test_load_model_succeeds():
    engine = FakeEngine()
    service = SpeechService(engine)

    service.load_model()

    assert service.is_model_loaded() is True


def test_load_model_raises_when_engine_fails():
    engine = FakeEngine(fail_load=True)
    service = SpeechService(engine)

    with pytest.raises(SpeechRecognitionException, match="Failed to load model"):
        service.load_model()

    assert service.is_model_loaded() is False


def test_transcribe_returns_engine_output(fake_paths):
    audio = fake_paths
    engine = FakeEngine(transcript="two masala dosa")
    service = SpeechService(engine)

    result = service.transcribe(str(audio))

    assert result == "2 masala dosa"
    assert engine.transcribe_called_with == str(audio)


def test_transcribe_auto_loads_model_if_not_loaded_yet(fake_paths):
    audio = fake_paths
    engine = FakeEngine()
    service = SpeechService(engine)

    assert service.is_model_loaded() is False
    service.transcribe(str(audio))
    assert service.is_model_loaded() is True


def test_transcribe_raises_when_audio_file_missing():
    engine = FakeEngine()
    service = SpeechService(engine)

    with pytest.raises(SpeechRecognitionException, match="Audio file not found"):
        service.transcribe("/nonexistent/path/order.wav")


def test_transcribe_wraps_unexpected_engine_errors(fake_paths):
    audio = fake_paths

    class BrokenEngine(FakeEngine):
        def transcribe(self, audio_file_path: str) -> str:
            raise RuntimeError("engine exploded")

    engine = BrokenEngine()
    service = SpeechService(engine)

    with pytest.raises(SpeechRecognitionException, match="Speech recognition failed"):
        service.transcribe(str(audio))
