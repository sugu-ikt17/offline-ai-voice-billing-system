"""Menu Context AI Engine — Dynamic context-aware speech understanding and correction.

Pipeline position:

    Audio
    → Whisper / Faster-Whisper          (raw transcript)
    → SpeechNormalizer.normalize()
    → MenuContextEngine.correct()       ← THIS MODULE
    → OrderParserService.parse()
    → MenuMatcherService.match()
    → BillGeneratorService.generate()

Design principles
-----------------
* **Dynamic Menu Context** — menu items are loaded dynamically from SQLite database
  (or provided list) rather than relying solely on a fixed alias dictionary.
* **In-Memory Caching** — caches active menu items, extracted words, and alias mappings
  in memory to prevent redundant SQLite queries on repeated requests. Automatically
  refreshes/invalidates on TTL or menu changes.
* **Quantity-Bound Adjacency Correction** — menu item corrections (exact, alias,
  rapidfuzz, phonetic, word distance) are strictly restricted to tokens or multi-word
  phrases directly adjacent to a valid quantity token (NUMBER + ITEM or ITEM + NUMBER).
  Unrelated conversational words are left unchanged.
* **Multi-Strategy Cascade** — evaluates corrections in strict priority order:
    1. Exact Match        (Confidence: 99% / 1.0)
    2. Alias Match        (Confidence: 96% / 0.96)
    3. RapidFuzz          (Confidence: RapidFuzz ratio score 0–100%)
    4. Phonetic Match     (Confidence: Metaphone / Soundex / Jaro-Winkler score)
    5. Word Distance      (Confidence: Damerau-Levenshtein normalized edit distance)
* **Confidence & Low-Confidence Threshold** — computes confidence scores for all
  evaluated candidate tokens.
"""

from dataclasses import dataclass, field
import time
from typing import Final, Optional
import jellyfish
from rapidfuzz import fuzz, process

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.interfaces.vocabulary_corrector_interface import VocabularyCorrectorInterface

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Protected token sets & Quantity tokens
# ---------------------------------------------------------------------------


def _build_protected_words() -> frozenset[str]:
    """Return a frozenset of all words that must never be corrected."""
    from app.application.services.order_parser_service import QUANTITY_WORDS  # noqa: PLC0415
    from app.application.services.speech_normalizer import (  # noqa: PLC0415
        ENGLISH_NUMBER_HOMOPHONES,
        NUMBER_TO_DIGIT_MAP,
        TAMIL_NUMBER_WORDS,
    )

    protected: set[str] = set()
    protected.update(QUANTITY_WORDS.keys())
    protected.update(ENGLISH_NUMBER_HOMOPHONES.keys())
    protected.update(TAMIL_NUMBER_WORDS.keys())
    protected.update(NUMBER_TO_DIGIT_MAP.keys())
    protected.update({
        "and", "with", "plus", "the", "a", "an", "of", "it", "i", "have",
        "has", "had", "all", "me", "my", "your", "we", "us", "please",
        "this", "that", "is", "are", "was", "were", "be", "for", "to",
        "in", "on", "at", "so", "or", "if", "think", "soul", "get", "give", "want", "need", "like"
    })
    return frozenset(protected)


def _build_quantity_words() -> frozenset[str]:
    """Return a frozenset of all recognized quantity/number words."""
    from app.application.services.order_parser_service import QUANTITY_WORDS  # noqa: PLC0415
    from app.application.services.speech_normalizer import (  # noqa: PLC0415
        ENGLISH_NUMBER_HOMOPHONES,
        NUMBER_TO_DIGIT_MAP,
        TAMIL_NUMBER_WORDS,
    )

    q_words: set[str] = set()
    q_words.update(QUANTITY_WORDS.keys())
    q_words.update(ENGLISH_NUMBER_HOMOPHONES.keys())
    q_words.update(TAMIL_NUMBER_WORDS.keys())
    q_words.update(NUMBER_TO_DIGIT_MAP.keys())
    return frozenset(q_words)


_PROTECTED_WORDS: Final[frozenset[str]] = _build_protected_words()
_QUANTITY_WORDS: Final[frozenset[str]] = _build_quantity_words()


def _is_quantity_token(token: str) -> bool:
    """Return True if token represents a numeric quantity or number word."""
    token_lower = token.lower()
    return token_lower.isdigit() or token_lower in _QUANTITY_WORDS


# ---------------------------------------------------------------------------
# Data Models for Correction Results
# ---------------------------------------------------------------------------


@dataclass
class TokenMatch:
    """Match result for a single token in transcript."""

    original_token: str
    corrected_token: str
    confidence: float  # 0.0 to 1.0 (or 0% to 100%)
    strategy: str  # exact_match, alias_match, rapidfuzz_similarity, phonetic_similarity, word_distance, none, protected, consumed
    suggestions: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class CorrectionResult:
    """Full correction result for a transcript."""

    original_transcript: str
    normalized_transcript: str
    corrected_transcript: str
    token_matches: list[TokenMatch]
    overall_confidence: float  # 0.0 to 1.0 (or 0% to 100%)


# ---------------------------------------------------------------------------
# Menu Context Engine Service
# ---------------------------------------------------------------------------


class MenuContextEngine(VocabularyCorrectorInterface):
    """Context-aware speech understanding & quantity-bound menu correction engine."""

    def __init__(
        self,
        confidence_threshold: float | None = None,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        """Initialize MenuContextEngine.

        Args:
            confidence_threshold: Threshold (0.0-1.0 or 0-100) below which corrections
                                 are treated as Low Confidence / Unknown. Defaults to settings.vocab_corrector_threshold.
            cache_ttl_seconds: In-memory cache time-to-live in seconds (default 300s).
        """
        raw_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "vocab_corrector_threshold", 0.65)
        )
        # Normalize threshold to 0.0 - 1.0 float scale
        self.confidence_threshold = (
            raw_threshold / 100.0 if raw_threshold > 1.0 else raw_threshold
        )
        self.cache_ttl_seconds = cache_ttl_seconds

        # In-memory cache structure
        self._cached_vocabulary: list[str] = []
        self._cached_menu_words: list[str] = []
        self._cached_alias_map: dict[str, str] = {}
        self._cache_timestamp: float = 0.0

    def refresh_cache(self, vocabulary: list[str]) -> None:
        """Explicitly refresh the in-memory menu cache with new menu items."""
        from app.application.services.speech_normalizer import load_menu_aliases  # noqa: PLC0415

        aliases = load_menu_aliases()

        words: set[str] = set()
        for item_name in vocabulary:
            for word in item_name.lower().split():
                clean = word.strip()
                if clean:
                    words.add(clean)

        self._cached_vocabulary = list(vocabulary)
        self._cached_menu_words = sorted(words)
        self._cached_alias_map = aliases
        self._cache_timestamp = time.time()
        logger.debug(
            "MenuContextEngine cache refreshed with %d menu items and %d unique words",
            len(vocabulary),
            len(words),
        )

    def invalidate_cache(self) -> None:
        """Invalidate the in-memory menu cache."""
        self._cached_vocabulary = []
        self._cached_menu_words = []
        self._cached_alias_map = {}
        self._cache_timestamp = 0.0

    def _is_cache_valid(self) -> bool:
        """Return True if in-memory cache is populated and not expired."""
        if not self._cached_vocabulary or not self._cached_menu_words:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl_seconds

    def correct(self, text: str, vocabulary: list[str] | None = None) -> str:
        """Correct transcript text using dynamic menu context and returns string.

        Args:
            text: Transcript string, already processed by SpeechNormalizer.
            vocabulary: Optional list of menu item names. If provided or cached,
                        used as dynamic context.

        Returns:
            Corrected transcript string.
        """
        result = self.correct_with_details(text, vocabulary)
        return result.corrected_transcript

    def correct_with_details(
        self, text: str, vocabulary: list[str] | None = None
    ) -> CorrectionResult:
        """Perform quantity-bound correction cascade with per-token confidence and logging.

        Args:
            text: Transcript string.
            vocabulary: Optional list of active menu item names.

        Returns:
            CorrectionResult dataclass with per-token strategy, confidence, and suggestions.
        """
        if not text or not text.strip():
            return CorrectionResult(
                original_transcript=text,
                normalized_transcript=text,
                corrected_transcript=text,
                token_matches=[],
                overall_confidence=1.0,
            )

        # Update cache if vocabulary supplied or cache expired
        if vocabulary is not None:
            if vocabulary != self._cached_vocabulary or not self._is_cache_valid():
                self.refresh_cache(vocabulary)

        vocab_words = self._cached_menu_words
        alias_map = self._cached_alias_map
        active_vocab = self._cached_vocabulary

        tokens = text.split()
        if not tokens:
            return CorrectionResult(
                original_transcript=text,
                normalized_transcript=text,
                corrected_transcript=text,
                token_matches=[],
                overall_confidence=1.0,
            )

        # Determine max words in any single active menu item
        max_vocab_len = 1
        if active_vocab:
            max_vocab_len = max(len(item.split()) for item in active_vocab)
        max_vocab_len = max(1, max_vocab_len)

        # Identify indices of quantity tokens
        quantity_indices = [i for i, t in enumerate(tokens) if _is_quantity_token(t)]

        # Track assigned matches for each token index (None if unassigned)
        token_matches: list[Optional[TokenMatch]] = [None] * len(tokens)

        # Mark quantity tokens as protected
        for q_idx in quantity_indices:
            token_matches[q_idx] = TokenMatch(
                original_token=tokens[q_idx],
                corrected_token=tokens[q_idx],
                confidence=1.0,
                strategy="protected",
            )

        # For each quantity token, evaluate directly adjacent candidate menu tokens/phrases
        for q_idx in quantity_indices:
            # ---------------------------------------------------------------
            # 1. Number-before-item (quantity_index + 1)
            # ---------------------------------------------------------------
            start_idx = q_idx + 1
            if (
                start_idx < len(tokens)
                and token_matches[start_idx] is None
                and not _is_quantity_token(tokens[start_idx])
            ):
                max_k = min(max_vocab_len, len(tokens) - start_idx)
                last_single_match = None
                for k in range(max_k, 0, -1):
                    end_idx = start_idx + k
                    if any(
                        _is_quantity_token(tokens[j]) or token_matches[j] is not None
                        for j in range(start_idx, end_idx)
                    ):
                        continue

                    candidate_str = " ".join(tokens[start_idx:end_idx])
                    match = self._evaluate_candidate(
                        candidate_str, vocab_words, alias_map, active_vocab
                    )
                    if k == 1:
                        last_single_match = match

                    if match.confidence >= self.confidence_threshold and match.strategy != "none":
                        token_matches[start_idx] = match
                        for j in range(start_idx + 1, end_idx):
                            token_matches[j] = TokenMatch(
                                original_token=tokens[j],
                                corrected_token="",
                                confidence=match.confidence,
                                strategy="consumed",
                            )
                        break
                else:
                    if last_single_match is not None and token_matches[start_idx] is None:
                        token_matches[start_idx] = last_single_match

            # ---------------------------------------------------------------
            # 2. Item-before-number (quantity_index - 1)
            # ---------------------------------------------------------------
            end_idx = q_idx
            if (
                end_idx - 1 >= 0
                and token_matches[end_idx - 1] is None
                and not _is_quantity_token(tokens[end_idx - 1])
            ):
                max_k = min(max_vocab_len, end_idx)
                last_single_match = None
                for k in range(max_k, 0, -1):
                    start_idx = end_idx - k
                    if any(
                        _is_quantity_token(tokens[j]) or token_matches[j] is not None
                        for j in range(start_idx, end_idx)
                    ):
                        continue

                    candidate_str = " ".join(tokens[start_idx:end_idx])
                    match = self._evaluate_candidate(
                        candidate_str, vocab_words, alias_map, active_vocab
                    )
                    if k == 1:
                        last_single_match = match

                    if match.confidence >= self.confidence_threshold and match.strategy != "none":
                        token_matches[start_idx] = match
                        for j in range(start_idx + 1, end_idx):
                            token_matches[j] = TokenMatch(
                                original_token=tokens[j],
                                corrected_token="",
                                confidence=match.confidence,
                                strategy="consumed",
                            )
                        break
                else:
                    if last_single_match is not None and token_matches[end_idx - 1] is None:
                        token_matches[end_idx - 1] = last_single_match

        # Fill remaining unassigned tokens as unchanged original tokens
        final_matches: list[TokenMatch] = []
        corrected_tokens: list[str] = []

        for i, match in enumerate(token_matches):
            if match is None:
                match = TokenMatch(
                    original_token=tokens[i],
                    corrected_token=tokens[i],
                    confidence=1.0,
                    strategy="none",
                )
                final_matches.append(match)
                corrected_tokens.append(tokens[i])
            elif match.strategy == "consumed":
                pass
            else:
                final_matches.append(match)
                corrected_tokens.append(match.corrected_token)

        corrected_text = " ".join(corrected_tokens)

        # Compute overall confidence across evaluated candidate tokens
        eval_confidences = [
            m.confidence
            for m in final_matches
            if m.strategy not in ("protected", "none", "consumed")
        ]
        overall_conf = (
            sum(eval_confidences) / len(eval_confidences)
            if eval_confidences
            else 1.0
        )

        result = CorrectionResult(
            original_transcript=text,
            normalized_transcript=text,
            corrected_transcript=corrected_text,
            token_matches=final_matches,
            overall_confidence=overall_conf,
        )

        # Log detailed correction report
        applied_corrections = [
            f"{m.original_token} -> {m.corrected_token} (strategy: {m.strategy}, confidence: {m.confidence * 100:.1f}%)"
            for m in final_matches
            if m.original_token.lower() != m.corrected_token.lower()
            and m.strategy not in ("protected", "none", "consumed")
        ]

        if applied_corrections:
            logger.info(
                "Menu Context Engine Correction:\nOriginal Transcript: %r\nNormalized Transcript: %r\nCorrections Applied: %s\nOverall Confidence: %.1f%%",
                text,
                text,
                "; ".join(applied_corrections),
                overall_conf * 100.0,
            )
        else:
            logger.debug(
                "Menu Context Engine Evaluated Transcript: %r (Overall Confidence: %.1f%%)",
                text,
                overall_conf * 100.0,
            )

        return result

    def _evaluate_candidate(
        self,
        candidate_str: str,
        vocab_words: list[str],
        alias_map: dict[str, str],
        active_vocab: list[str],
    ) -> TokenMatch:
        """Evaluate a candidate string (single token or multi-word phrase) against correction strategies."""
        cand_lower = candidate_str.lower().strip()

        if not cand_lower:
            return TokenMatch(
                original_token=candidate_str,
                corrected_token=candidate_str,
                confidence=1.0,
                strategy="none",
            )

        # Do not correct protected tokens (digits, quantities, connectors)
        if cand_lower.isdigit() or cand_lower in _PROTECTED_WORDS:
            return TokenMatch(
                original_token=candidate_str,
                corrected_token=candidate_str,
                confidence=1.0,
                strategy="protected",
            )

        if not vocab_words and not active_vocab:
            return TokenMatch(
                original_token=candidate_str,
                corrected_token=candidate_str,
                confidence=1.0,
                strategy="none",
            )

        # -------------------------------------------------------------------
        # Strategy 1: Exact Match
        # -------------------------------------------------------------------
        for menu_name in active_vocab:
            if cand_lower == menu_name.lower():
                return TokenMatch(
                    original_token=candidate_str,
                    corrected_token=menu_name.lower(),
                    confidence=0.99,
                    strategy="exact_match",
                    suggestions=[(menu_name.lower(), 0.99)],
                )

        if cand_lower in vocab_words:
            return TokenMatch(
                original_token=candidate_str,
                corrected_token=cand_lower,
                confidence=0.99,
                strategy="exact_match",
                suggestions=[(cand_lower, 0.99)],
            )

        # -------------------------------------------------------------------
        # Strategy 2: Alias Match
        # -------------------------------------------------------------------
        if cand_lower in alias_map:
            canonical = alias_map[cand_lower]
            canon_words = canonical.lower().split()
            if any(w in vocab_words for w in canon_words) or any(
                canonical.lower() == v.lower() for v in active_vocab
            ):
                if 0.96 >= self.confidence_threshold:
                    return TokenMatch(
                        original_token=candidate_str,
                        corrected_token=canonical.lower(),
                        confidence=0.96,
                        strategy="alias_match",
                        suggestions=[(canonical.lower(), 0.96)],
                    )

        # -------------------------------------------------------------------
        # Strategy 3: RapidFuzz Similarity
        # -------------------------------------------------------------------
        choices = set()
        for v in active_vocab:
            choices.add(v.lower())
        for w in vocab_words:
            choices.add(w)

        rf_match = process.extractOne(
            cand_lower,
            list(choices),
            scorer=fuzz.ratio,
        )
        if rf_match:
            rf_word, rf_score, _ = rf_match
            rf_conf = rf_score / 100.0
            if rf_conf >= self.confidence_threshold:
                return TokenMatch(
                    original_token=candidate_str,
                    corrected_token=rf_word,
                    confidence=rf_conf,
                    strategy="rapidfuzz_similarity",
                    suggestions=[(rf_word, rf_conf)],
                )

        # -------------------------------------------------------------------
        # Strategy 4: Phonetic Similarity
        # -------------------------------------------------------------------
        phon_match, phon_conf = self._check_phonetic_similarity(
            cand_lower, vocab_words, active_vocab
        )
        if phon_match and phon_conf >= self.confidence_threshold:
            return TokenMatch(
                original_token=candidate_str,
                corrected_token=phon_match,
                confidence=phon_conf,
                strategy="phonetic_similarity",
                suggestions=[(phon_match, phon_conf)],
            )

        # -------------------------------------------------------------------
        # Strategy 5: Word Distance (Damerau-Levenshtein)
        # -------------------------------------------------------------------
        best_dist_word = None
        best_dist_conf = 0.0
        for choice in choices:
            dist = jellyfish.damerau_levenshtein_distance(cand_lower, choice)
            max_len = max(len(cand_lower), len(choice))
            conf = 1.0 - (dist / max_len)
            if conf > best_dist_conf:
                best_dist_conf = conf
                best_dist_word = choice

        if best_dist_word and best_dist_conf >= self.confidence_threshold:
            return TokenMatch(
                original_token=candidate_str,
                corrected_token=best_dist_word,
                confidence=best_dist_conf,
                strategy="word_distance",
                suggestions=[(best_dist_word, best_dist_conf)],
            )

        # -------------------------------------------------------------------
        # Low Confidence / Below Threshold
        # -------------------------------------------------------------------
        highest_cand = best_dist_word or (rf_match[0] if rf_match else cand_lower)
        highest_score = max(
            best_dist_conf,
            (rf_match[1] / 100.0) if rf_match else 0.0,
            phon_conf,
        )

        suggestions = [(highest_cand, highest_score)] if highest_cand else []

        logger.info(
            "Low confidence match for %r: best match %r with confidence %.1f%% < threshold %.1f%% — keeping original token",
            candidate_str,
            highest_cand,
            highest_score * 100.0,
            self.confidence_threshold * 100.0,
        )

        return TokenMatch(
            original_token=candidate_str,
            corrected_token=candidate_str,
            confidence=highest_score,
            strategy="none",
            suggestions=suggestions,
        )

    @staticmethod
    def _check_phonetic_similarity(
        token: str, vocab_words: list[str], active_vocab: list[str]
    ) -> tuple[str | None, float]:
        """Compute phonetic similarity using jellyfish metaphone, soundex, and custom rules."""
        DOMAIN_PHONETIC_MAP = {
            "curry": "puri",
            "kury": "puri",
            "copy": "coffee",
            "kapi": "coffee",
            "kaapi": "coffee",
            "coffer": "coffee",
            "cofi": "coffee",
            "tee": "tea",
            "poorii": "puri",
            "dosay": "dosa",
            "itly": "idli",
            "vadai": "vada",
        }

        if token in DOMAIN_PHONETIC_MAP:
            target = DOMAIN_PHONETIC_MAP[token]
            if target in vocab_words or any(
                target == v.lower() for v in active_vocab
            ):
                conf = 0.88 if token in ("curry", "kury") else 0.96
                return target, conf

        t_meta = jellyfish.metaphone(token)
        t_soundex = jellyfish.soundex(token)

        best_word = None
        best_score = 0.0

        for v_word in vocab_words:
            v_meta = jellyfish.metaphone(v_word)
            v_soundex = jellyfish.soundex(v_word)
            jw = jellyfish.jaro_winkler_similarity(token, v_word)

            score = 0.0
            if t_meta and t_meta == v_meta:
                score = min(0.95, 0.85 + (0.10 * jw))
            elif t_soundex and t_soundex == v_soundex:
                score = min(0.92, 0.75 + (0.15 * jw))
            elif t_meta and v_meta:
                meta_dist = jellyfish.damerau_levenshtein_distance(t_meta, v_meta)
                if meta_dist <= 1 and (len(t_meta) > 2 or jw >= 0.80):
                    score = min(0.88, 0.70 + (0.20 * jw))

            if score > best_score:
                best_score = score
                best_word = v_word

        return best_word, best_score

