"""Voice transcription API tests.

Overrides get_speech_service with a fake so these tests don't depend on
a real whisper.cpp binary being installed on the test machine.
"""

import pytest

from app.core.exceptions import SpeechRecognitionException
from app.presentation.api.v1.voice_routes import get_speech_service

TRANSCRIBE_URL = "/api/v1/voice/transcribe"


class FakeSpeechService:
    def __init__(self, text: str = "2 dosa 1 tea") -> None:
        self._text = text

    def transcribe(self, audio_file_path: str) -> str:
        return self._text


class FailingSpeechService:
    def transcribe(self, audio_file_path: str) -> str:
        raise SpeechRecognitionException("engine exploded")


@pytest.fixture()
def override_speech_service(client):
    """Yields a setter so each test can plug in whichever fake it needs."""
    from app.main import app

    def _set(service):
        app.dependency_overrides[get_speech_service] = lambda: service

    yield _set
    app.dependency_overrides.pop(get_speech_service, None)


def test_transcribe_returns_recognized_text(client, override_speech_service):
    override_speech_service(FakeSpeechService("2 dosa 1 tea"))

    response = client.post(
        TRANSCRIBE_URL,
        files={"audio_file": ("order.wav", b"fake wav bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"recognized_text": "2 dosa 1 tea"}


def test_transcribe_rejects_unsupported_format(client, override_speech_service):
    override_speech_service(FakeSpeechService())

    response = client.post(
        TRANSCRIBE_URL,
        files={"audio_file": ("order.mp3", b"fake mp3 bytes", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_transcribe_rejects_empty_file(client, override_speech_service):
    override_speech_service(FakeSpeechService())

    response = client.post(
        TRANSCRIBE_URL,
        files={"audio_file": ("order.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_handles_recognition_failure(client, override_speech_service):
    override_speech_service(FailingSpeechService())

    response = client.post(
        TRANSCRIBE_URL,
        files={"audio_file": ("order.wav", b"fake wav bytes", "audio/wav")},
    )

    assert response.status_code == 502
    assert "Speech recognition failed" in response.json()["detail"]
