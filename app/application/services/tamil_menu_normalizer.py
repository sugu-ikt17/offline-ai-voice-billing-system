"""Tamil Menu Normalizer — Transliterates Tamil-script menu words and numbers into English/Tanglish menu vocabulary.

Pipeline Position:
    Audio
    → Sarvam STT                       (raw transcript, e.g. "2 டீ 2 தோசை 4 சமோசா")
    → TamilMenuNormalizer.normalize()  (Tamil script menu & number normalization)  ← THIS MODULE
    → SpeechNormalizer.normalize()     (number words, alias map, reordering)
    → MenuContextEngine.correct()      (dynamic vocabulary correction)
    → OrderParserService.parse()       (token parsing)
    → MenuMatcherService.match()       (menu DB resolution)
    → BillGeneratorService.generate()  (billing)
"""

import re
from typing import Final, Optional
from rapidfuzz import fuzz, process

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tamil Script Number Mapping to Digits
# ---------------------------------------------------------------------------

TAMIL_SCRIPT_NUMBERS: Final[dict[str, str]] = {
    # 1
    "ஒரு": "1",
    "ஒன்று": "1",
    "ஒன்னு": "1",
    "ஒன்னா": "1",
    # 2
    "இரண்டு": "2",
    "ரெண்டு": "2",
    "ரெண்டூ": "2",
    "ரெண்டுஹ்": "2",
    # 3
    "மூன்று": "3",
    "மூனு": "3",
    "மூனூ": "3",
    # 4
    "நான்கு": "4",
    "நாலு": "4",
    "நாலூ": "4",
    # 5
    "ஐந்து": "5",
    "அஞ்சு": "5",
    # 6
    "ஆறு": "6",
    # 7
    "ஏழு": "7",
    # 8
    "எட்டு": "8",
    # 9
    "ஒன்பது": "9",
    "ஒம்போது": "9",
    "ஒம்பொது": "9",
    # 10
    "பத்து": "10",
}

# ---------------------------------------------------------------------------
# Tamil Script Common Food Term / Component Map
# ---------------------------------------------------------------------------

TAMIL_FOOD_COMPONENTS: Final[dict[str, str]] = {
    # Basic menu items
    "டீ": "tea",
    "டீயும்": "tea",
    "தேநீர்": "tea",
    "காபி": "coffee",
    "காஃபி": "coffee",
    "கோபி": "coffee",
    "தோசை": "dosa",
    "தோஸை": "dosa",
    "டோசை": "dosa",
    "தோசா": "dosa",
    "டோஸா": "dosa",
    "இட்லி": "idli",
    "இட்லீ": "idli",
    "பூரி": "poori",
    "புரி": "poori",
    "வடை": "vada",
    "வட": "vada",
    "சமோசா": "samosa",
    "சமுசா": "samosa",
    "பொங்கல்": "pongal",
    "சப்பாத்தி": "chapati",
    "சபதி": "chapati",
    "பரோட்டா": "parotta",
    "புரோட்டா": "parotta",
    "பரோடா": "parotta",
    # Modifiers & Descriptors
    "மசாலா": "masala",
    "பிளைன்": "plain",
    "ஆனியன்": "onion",
    "வெங்காயம்": "onion",
    "வெங்காய": "onion",
    "நெய்": "ghee",
    "கீ": "ghee",
    "ரோஸ்ட்": "roast",
    "ரவா": "rava",
    "பொடி": "podi",
    "சாம்பார்": "sambar",
    "பன்னீர்": "paneer",
    "பட்டர்": "butter",
    "மஷ்ரூம்": "mushroom",
    "சில்லி": "chilli",
    "ரைஸ்": "rice",
    "சாதம்": "rice",
    "கர்டு": "curd",
    "தயிர்": "curd",
    "மீல்ஸ்": "meals",
    "மினி": "mini",
    "ஸ்பெஷல்": "special",
}

# Regex for replacing Tamil script numbers
_SORTED_TAMIL_NUM_KEYS = sorted(TAMIL_SCRIPT_NUMBERS.keys(), key=len, reverse=True)
_TAMIL_NUM_REGEX = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in _SORTED_TAMIL_NUM_KEYS) + r")(?!\w)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tamil Unicode Transliteration (Deterministic Rule Engine)
# ---------------------------------------------------------------------------

_TAMIL_VOWELS = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u",
    "ஊ": "oo", "எ": "e", "ஏ": "ae", "ஐ": "ai", "ஒ": "o",
    "ஓ": "oa", "ஔ": "au",
}

_TAMIL_CONSONANTS = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "t",
    "ண": "n", "த": "th", "ந": "n", "ப": "p", "ம": "m",
    "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh",
    "ள": "l", "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh",
    "ஸ": "s", "ஹ": "h",
}

_TAMIL_VOWEL_SIGNS = {
    "ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo",
    "ெ": "e", "ே": "ae", "ை": "ai", "ொ": "o", "ோ": "oa",
    "ௌ": "au",
}

_TAMIL_VIRAMA = "்"


def transliterate_tamil_text(text: str) -> str:
    """Convert Tamil script text to phonetic Latin/Tanglish characters."""
    if not text or all(ord(c) < 128 for c in text):
        return text

    words = text.split()
    transliterated_words = []

    for word in words:
        if all(ord(c) < 128 for c in word):
            transliterated_words.append(word)
            continue

        result = []
        i = 0
        n = len(word)
        while i < n:
            char = word[i]
            if char in _TAMIL_VOWELS:
                result.append(_TAMIL_VOWELS[char])
                i += 1
            elif char in _TAMIL_CONSONANTS:
                base_cons = _TAMIL_CONSONANTS[char]
                if i + 1 < n and word[i + 1] == _TAMIL_VIRAMA:
                    result.append(base_cons)
                    i += 2
                elif i + 1 < n and word[i + 1] in _TAMIL_VOWEL_SIGNS:
                    result.append(base_cons + _TAMIL_VOWEL_SIGNS[word[i + 1]])
                    i += 2
                else:
                    result.append(base_cons + "a")
                    i += 1
            elif char in _TAMIL_VOWEL_SIGNS:
                result.append(_TAMIL_VOWEL_SIGNS[char])
                i += 1
            elif char == "ஃ":
                result.append("f")
                i += 1
            else:
                result.append(char)
                i += 1
        transliterated_words.append("".join(result))

    return " ".join(transliterated_words)


# ---------------------------------------------------------------------------
# Tamil Menu Normalizer Service
# ---------------------------------------------------------------------------

class TamilMenuNormalizer:
    """Normalizes Tamil-script transcripts into English/Tanglish menu vocabulary."""

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.confidence_threshold = confidence_threshold

    def normalize(self, text: str, active_vocabulary: list[str] | None = None) -> str:
        """Normalize raw Tamil script transcript into English menu tokens.

        Args:
            text: Raw transcript string (e.g. "2 டீ 2 தோசை 4 சமோசா").
            active_vocabulary: Optional list of active menu item names from repository.

        Returns:
            Normalized transcript with Tamil script menu names mapped to English/Tanglish
            and Tamil script numbers converted to digits.
        """
        if not text or not text.strip():
            return text

        # Step 1 — Clean punctuation boundaries and replace Tamil Unicode script numbers with digits
        cleaned_text = re.sub(r"[,.\?!;:]+", " ", text)
        working = _TAMIL_NUM_REGEX.sub(
            lambda m: TAMIL_SCRIPT_NUMBERS[m.group(0)], cleaned_text
        )

        tokens = working.split()
        if not tokens:
            return working

        active_vocab = active_vocabulary or []
        vocab_lowers = [item.strip().lower() for item in active_vocab if item.strip()]

        # Helper to check if token is quantity digit or English quantity word
        def _is_qty(token: str) -> bool:
            t = token.lower()
            return t.isdigit() or t in {
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "oru", "onnu", "rendu", "moonu", "naalu", "anju", "aaru", "ezhu", "ettu", "ombodhu", "pathu",
            }

        quantity_indices = [i for i, t in enumerate(tokens) if _is_qty(t)]

        # If no quantity tokens found, evaluate words as potential standalone items if active menu matches
        if not quantity_indices:
            return self._normalize_tokens_without_quantity(tokens, vocab_lowers)

        # Max word length of multi-word menu items
        max_vocab_len = max([len(v.split()) for v in vocab_lowers], default=2)
        max_vocab_len = max(2, max_vocab_len)

        replaced_flags = [False] * len(tokens)
        result_tokens = list(tokens)

        # Step 2 — For each quantity token, evaluate directly adjacent Tamil menu phrases
        for q_idx in quantity_indices:
            # 1. Number-before-item (q_idx + 1)
            start_idx = q_idx + 1
            if start_idx < len(tokens) and not replaced_flags[start_idx] and not _is_qty(tokens[start_idx]):
                max_k = min(max_vocab_len, len(tokens) - start_idx)
                for k in range(max_k, 0, -1):
                    end_idx = start_idx + k
                    if any(replaced_flags[j] or _is_qty(tokens[j]) for j in range(start_idx, end_idx)):
                        continue

                    candidate_str = " ".join(tokens[start_idx:end_idx])
                    replacement, score = self._resolve_tamil_candidate(candidate_str, vocab_lowers)
                    if score >= self.confidence_threshold and replacement:
                        result_tokens[start_idx] = replacement
                        replaced_flags[start_idx] = True
                        for j in range(start_idx + 1, end_idx):
                            result_tokens[j] = ""
                            replaced_flags[j] = True
                        break

            # 2. Item-before-number (q_idx - 1)
            end_idx = q_idx
            if end_idx - 1 >= 0 and not replaced_flags[end_idx - 1] and not _is_qty(tokens[end_idx - 1]):
                max_k = min(max_vocab_len, end_idx)
                for k in range(max_k, 0, -1):
                    start_idx = end_idx - k
                    if any(replaced_flags[j] or _is_qty(tokens[j]) for j in range(start_idx, end_idx)):
                        continue

                    candidate_str = " ".join(tokens[start_idx:end_idx])
                    replacement, score = self._resolve_tamil_candidate(candidate_str, vocab_lowers)
                    if score >= self.confidence_threshold and replacement:
                        result_tokens[start_idx] = replacement
                        replaced_flags[start_idx] = True
                        for j in range(start_idx + 1, end_idx):
                            result_tokens[j] = ""
                            replaced_flags[j] = True
                        break

        # Reassemble non-empty tokens
        normalized_output = " ".join([t for t in result_tokens if t])
        return normalized_output

    def _normalize_tokens_without_quantity(self, tokens: list[str], vocab_lowers: list[str]) -> str:
        """Fallback normalization when no explicit quantity is in the transcript."""
        result_tokens = list(tokens)
        for i, token in enumerate(tokens):
            replacement, score = self._resolve_tamil_candidate(token, vocab_lowers)
            if score >= self.confidence_threshold and replacement:
                result_tokens[i] = replacement
        return " ".join(result_tokens)

    def _resolve_tamil_candidate(self, candidate_str: str, vocab_lowers: list[str]) -> tuple[str | None, float]:
        """Resolve a Tamil script candidate phrase to an active menu item or canonical component."""
        clean_cand = candidate_str.strip()
        if not clean_cand:
            return None, 0.0

        # If candidate is pure ASCII/English, do not modify unless it's a known Tamil component
        is_tamil = any(ord(c) >= 0x0B80 and ord(c) <= 0x0BFF for c in clean_cand)
        if not is_tamil:
            # Check if it's already an active menu item or valid token
            if clean_cand.lower() in vocab_lowers:
                return clean_cand.lower(), 1.0
            return None, 0.0

        cand_words = clean_cand.split()

        # Strategy A: Direct Component Dictionary Mapping
        mapped_components = []
        all_words_mapped = True
        for word in cand_words:
            if word in TAMIL_FOOD_COMPONENTS:
                mapped_components.append(TAMIL_FOOD_COMPONENTS[word])
            else:
                all_words_mapped = False
                break

        if all_words_mapped and mapped_components:
            comp_phrase = " ".join(mapped_components)
            # If vocab_lowers provided, check exact match first, then multi-word overlap
            if vocab_lowers:
                for v_item in vocab_lowers:
                    if comp_phrase == v_item:
                        return v_item, 0.99
                sorted_vocab = sorted(vocab_lowers, key=len, reverse=True)
                for v_item in sorted_vocab:
                    if set(comp_phrase.split()) == set(v_item.split()):
                        return v_item, 0.98
                    if comp_phrase in v_item:
                        return v_item, 0.95
            return comp_phrase, 0.95

        # Strategy B: Transliteration + Phonetic & Fuzzy Matching against Active Vocabulary
        transliterated = transliterate_tamil_text(clean_cand).lower()

        if vocab_lowers:
            # Check exact transliteration match
            for v_item in vocab_lowers:
                if transliterated == v_item:
                    return v_item, 0.96

            # Fuzzy match transliterated string against active menu items
            rf_match = process.extractOne(transliterated, vocab_lowers, scorer=fuzz.ratio)
            if rf_match:
                best_item, score, _ = rf_match
                confidence = score / 100.0
                if confidence >= self.confidence_threshold:
                    return best_item, confidence

                # Try partial ratio or token set ratio for multi-word
                rf_token_match = process.extractOne(transliterated, vocab_lowers, scorer=fuzz.token_set_ratio)
                if rf_token_match:
                    best_item_ts, score_ts, _ = rf_token_match
                    confidence_ts = score_ts / 100.0
                    if confidence_ts >= 0.85:
                        return best_item_ts, confidence_ts

        # Strategy C: Partial component mapping fallback
        partial_components = []
        for word in cand_words:
            if word in TAMIL_FOOD_COMPONENTS:
                partial_components.append(TAMIL_FOOD_COMPONENTS[word])
            else:
                partial_components.append(transliterate_tamil_text(word).lower())

        fallback_phrase = " ".join(partial_components)

        if vocab_lowers:
            rf_match = process.extractOne(fallback_phrase, vocab_lowers, scorer=fuzz.ratio)
            if rf_match:
                best_item, score, _ = rf_match
                if (score / 100.0) >= self.confidence_threshold:
                    return best_item, score / 100.0

        return None, 0.0
