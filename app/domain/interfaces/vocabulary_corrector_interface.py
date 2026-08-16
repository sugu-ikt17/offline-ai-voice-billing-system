"""Vocabulary Corrector interface.

Defines the contract that any vocabulary correction strategy must fulfil.
The application layer depends only on this interface — concrete
implementations (fuzzy-DB, ML, dictionary lookup, …) are swappable without
touching any caller.
"""

from abc import ABC, abstractmethod


class VocabularyCorrectorInterface(ABC):
    """Contract for transcript vocabulary correction services."""

    @abstractmethod
    def correct(self, text: str, vocabulary: list[str]) -> str:
        """Correct unrecognised words in *text* against a *vocabulary*.

        Args:
            text:       Normalised transcript string (output of SpeechNormalizer).
            vocabulary: List of canonical strings to match against
                        (e.g. lower-cased menu item names).

        Returns:
            Corrected transcript string.  Must never modify numeric tokens
            or recognised quantity words.
        """
        raise NotImplementedError
