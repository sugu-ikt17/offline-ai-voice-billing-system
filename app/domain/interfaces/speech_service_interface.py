"""Contract for any speech-to-text engine used by the system."""

from abc import ABC, abstractmethod


class SpeechServiceInterface(ABC):
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe an audio file into raw text."""
        raise NotImplementedError
