"""Menu Vocabulary Corrector — wrapper delegating to MenuContextEngine for full backward compatibility.

Pipeline position:

    Audio
    → Whisper / Faster-Whisper (raw transcript)
    → SpeechNormalizer.normalize()
    → MenuVocabularyCorrector.correct() / MenuContextEngine.correct()  ← THIS MODULE
    → OrderParserService.parse()
    → MenuMatcherService.match()
    → BillGeneratorService.generate()
"""

from typing import Final
from app.application.services.menu_context_engine import MenuContextEngine, _PROTECTED_WORDS  # noqa: F401
from app.core.config import settings
from app.domain.interfaces.vocabulary_corrector_interface import VocabularyCorrectorInterface


class MenuVocabularyCorrector(VocabularyCorrectorInterface):
    """Corrects spoken menu words delegating to MenuContextEngine.

    Maintains 100% backward compatibility with all imports and interfaces.
    """

    def __init__(self, threshold: float | None = None) -> None:
        raw_thresh = (
            threshold
            if threshold is not None
            else getattr(settings, "vocab_corrector_threshold", 0.65)
        )
        self._threshold = raw_thresh
        self._engine = MenuContextEngine(confidence_threshold=raw_thresh)

    @property
    def threshold(self) -> float:
        """The current similarity threshold (read-only)."""
        return self._threshold

    def correct(self, text: str, vocabulary: list[str]) -> str:
        """Correct transcript text using MenuContextEngine."""
        return self._engine.correct(text, vocabulary)

    @staticmethod
    def _extract_vocab_words(vocabulary: list[str]) -> list[str]:
        """Extract sorted, unique lowercased words from multi-word menu names."""
        words: set[str] = set()
        for name in vocabulary:
            for word in name.lower().split():
                clean = word.strip()
                if clean:
                    words.add(clean)
        return sorted(words)
