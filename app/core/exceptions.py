"""Application-wide custom exceptions.

These are framework-agnostic (no FastAPI imports here); the presentation
layer is responsible for translating them into HTTP responses.
"""


class AppException(Exception):
    """Base class for all application-specific exceptions."""


class NotFoundException(AppException):
    """Raised when a requested resource does not exist."""


class DuplicateException(AppException):
    """Raised when attempting to create a resource that already exists."""


class ValidationException(AppException):
    """Raised when input data fails a business rule."""


class SpeechRecognitionException(AppException):
    """Raised when audio transcription fails."""
