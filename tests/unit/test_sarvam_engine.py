"""SarvamSpeechEngine unit tests.

Uses mocks (unittest.mock) to ensure no real Sarvam AI network API calls are made during tests.
"""

import array
import math
import wave
from unittest.mock import MagicMock, patch

import pytest

from app.application.services.speech_service import SpeechService
from app.core.exceptions import SpeechRecognitionException
from app.infrastructure.speech_engine.sarvam_engine import SarvamSpeechEngine


def _write_minimal_wav(path) -> None:
    """Write a 0.1-second 16 kHz mono 16-bit PCM WAV file for test input."""
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


@pytest.fixture()
def fake_audio(tmp_path):
    audio_path = tmp_path / "test_sarvam.wav"
    _write_minimal_wav(audio_path)
    return audio_path


def test_api_key_read_from_configuration(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.sarvam_api_key", "test-config-key-12345")
    engine = SarvamSpeechEngine()
    assert engine.api_key == "test-config-key-12345"
    assert engine.is_available() is True


def test_engine_initializes_correctly():
    engine = SarvamSpeechEngine(
        api_key="secret-key-xyz",
        model="saaras:v3",
        language_code="ta-IN",
        mode="transcribe",
    )
    assert engine.model == "saaras:v3"
    assert engine.language_code == "ta-IN"
    assert engine.mode == "transcribe"
    assert engine.is_model_loaded() is False


@patch("sarvamai.SarvamAI")
def test_transcribe_sends_audio_and_returns_transcript(mock_sarvam_cls, fake_audio):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.transcript = "2 tea 2 dosa 4 samosa"
    mock_client.speech_to_text.transcribe.return_value = mock_response
    mock_sarvam_cls.return_value = mock_client

    engine = SarvamSpeechEngine(api_key="test-api-key")
    result = engine.transcribe(str(fake_audio))

    assert result == "2 tea 2 dosa 4 samosa"
    assert engine.is_model_loaded() is True
    mock_client.speech_to_text.transcribe.assert_called_once()
    kwargs = mock_client.speech_to_text.transcribe.call_args.kwargs
    assert kwargs["model"] == "saaras:v3"
    assert kwargs["mode"] == "transcribe"
    assert kwargs["language_code"] == "ta-IN"


def test_missing_api_key_raises_speech_exception():
    engine = SarvamSpeechEngine(api_key="")
    assert engine.is_available() is False
    with pytest.raises(SpeechRecognitionException, match="SARVAM_API_KEY.*missing"):
        engine.load_model()


@patch("sarvamai.SarvamAI")
def test_api_failure_raises_speech_exception_and_does_not_print_key(mock_sarvam_cls, fake_audio, capsys):
    mock_client = MagicMock()
    secret_key = "super-secret-key-999"
    mock_client.speech_to_text.transcribe.side_effect = Exception(f"HTTP 401 Unauthorized for key {secret_key}")
    mock_sarvam_cls.return_value = mock_client

    engine = SarvamSpeechEngine(api_key=secret_key)
    with pytest.raises(SpeechRecognitionException) as exc_info:
        engine.transcribe(str(fake_audio))

    # Verify exception message redacts secret key
    assert secret_key not in str(exc_info.value)

    # Verify stdout/stderr captured by capsys does not print secret key
    captured = capsys.readouterr()
    assert secret_key not in captured.out
    assert secret_key not in captured.err


def test_speech_service_dependency_injection_with_sarvam_engine(fake_audio):
    mock_engine = MagicMock()
    mock_engine.is_available.return_value = True
    mock_engine.is_model_loaded.return_value = True
    mock_engine.transcribe.return_value = "5 coffee 4 dosa"

    service = SpeechService(engine=mock_engine)
    result = service.transcribe(str(fake_audio))

    assert result == "5 coffee 4 dosa"
    mock_engine.transcribe.assert_called_once()
