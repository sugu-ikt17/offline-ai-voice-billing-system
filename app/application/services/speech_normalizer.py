"""Speech Normalizer — post-processing layer between Whisper and the parser.

Pipeline position:

    Audio
    → Whisper.cpp          (raw transcript)
    → SpeechNormalizer.normalize()     ← THIS MODULE
    → MenuVocabularyCorrector.correct()
    → OrderParserService.parse()
    → MenuMatcherService.match()
    → BillGeneratorService.generate()

Responsibilities
----------------
1. **Number normalization into digits** — all English, Tanglish, and Tamil Unicode
   number words/homophones are mapped directly to numeric digits (e.g. "one", "oru",
   "ரெண்டு", "four", "naalu" → "1", "2", "4").
2. **Menu item vocabulary alias mapping** — spoken aliases (including Tamil Unicode
   and Tanglish transliterations) are mapped to canonical menu item names based on
   data/menu_aliases.json.
3. **Multiword parsing & quantity reordering** — reorders item-first quantity-last
   patterns (e.g. "tea 2", "coffee moonu", "பூரி நாலு") into standard quantity-first
   patterns ("2 tea", "3 coffee", "4 puri") before passing to the parser.
4. **Correction logging** — logs every normalization step with original text,
   corrected text, and reason.
"""

import json
import re
from pathlib import Path
from typing import Final

from app.core.config import BASE_DIR
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Number word mappings to Digits
# ---------------------------------------------------------------------------

NUMBER_TO_DIGIT_MAP: Final[dict[str, str]] = {
    # 1
    "one": "1",
    "won": "1",
    "wun": "1",
    "oneu": "1",
    "oru": "1",
    "onnu": "1",
    "onna": "1",
    "ஒரு": "1",
    "ஒன்று": "1",
    # 2
    "two": "2",
    "too": "2",
    "to": "2",
    "rendu": "2",
    "renduu": "2",
    "renduh": "2",
    "ரெண்டு": "2",
    "இரண்டு": "2",
    # 3
    "three": "3",
    "moonu": "3",
    "munu": "3",
    "munuu": "3",
    "மூன்று": "3",
    "மூனு": "3",
    # 4
    "four": "4",
    "for": "4",
    "naalu": "4",
    "naaluu": "4",
    "நாலு": "4",
    "நான்கு": "4",
    # 5
    "five": "5",
    "anju": "5",
    "ஐந்து": "5",
    "அஞ்சு": "5",
    # 6
    "six": "6",
    "aaru": "6",
    "ஆறு": "6",
    # 7
    "seven": "7",
    "ezhu": "7",
    "ஏழு": "7",
    # 8
    "eight": "8",
    "ettu": "8",
    "எட்டு": "8",
    # 9
    "nine": "9",
    "onbadhu": "9",
    "ombodhu": "9",
    "ஒன்பது": "9",
    "ஒம்போது": "9",
    # 10
    "ten": "10",
    "pathu": "10",
    "பத்து": "10",
}

# Retain backward-compatible exported dictionaries for external integrity checks
ENGLISH_NUMBER_HOMOPHONES: Final[dict[str, str]] = {
    "won": "one",
    "wun": "one",
    "too": "two",
    "to": "two",
    "for": "four",
}

TAMIL_NUMBER_WORDS: Final[dict[str, str]] = {
    "oru": "one",
    "onnu": "one",
    "rendu": "two",
    "renduu": "two",
    "moonu": "three",
    "naalu": "four",
    "anju": "five",
    "aaru": "six",
    "ezhu": "seven",
    "ettu": "eight",
    "ombodhu": "nine",
    "pathu": "ten",
}

MENU_PRONUNCIATIONS: Final[dict[str, str]] = {
    "coffer": "coffee",
    "cofi": "coffee",
    "coffey": "coffee",
    "cofee": "coffee",
    "caffe": "coffee",
    "kaapi": "coffee",
    "kapi": "coffee",
    "copy": "coffee",
    "காபி": "coffee",
    "tee": "tea",
    "te": "tea",
    "ti": "tea",
    "cha": "tea",
    "chay": "tea",
    "டீ": "tea",
    "doza": "dosa",
    "dosai": "dosa",
    "dosay": "dosa",
    "dhosa": "dosa",
    "thosai": "dosa",
    "தோசை": "dosa",
    "idly": "idli",
    "itly": "idli",
    "idlee": "idli",
    "iddly": "idli",
    "இட்லி": "idli",
    "vadai": "vada",
    "vaday": "vada",
    "wada": "vada",
    "vade": "vada",
    "வடை": "vada",
    "poori": "puri",
    "poorii": "puri",
    "பூரி": "puri",
    "sambhar": "sambar",
    "saambar": "sambar",
    "pongall": "pongal",
    "pongul": "pongal",
    "uppma": "upma",
    "uppuma": "upma",
    "parota": "parotta",
    "paratha": "parotta",
}


def load_menu_aliases() -> dict[str, str]:
    """Load data/menu_aliases.json and return a flat alias -> canonical mapping."""
    mapping: dict[str, str] = dict(MENU_PRONUNCIATIONS)
    json_path = BASE_DIR / "data" / "menu_aliases.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for canonical, aliases in data.items():
                    canonical_clean = canonical.lower().strip()
                    for alias in aliases:
                        mapping[alias.lower().strip()] = canonical_clean
        except Exception as exc:
            logger.warning("Could not load menu_aliases.json: %s", exc)
    return mapping


_MENU_ALIASES_MAP: Final[dict[str, str]] = load_menu_aliases()

# Build pre-compiled regex for number replacement (longest key first)
_SORTED_NUMBER_KEYS = sorted(NUMBER_TO_DIGIT_MAP.keys(), key=len, reverse=True)
_NUMBER_REGEX = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in _SORTED_NUMBER_KEYS) + r")(?!\w)",
    re.IGNORECASE,
)

# Build pre-compiled regex for alias replacement (longest key first)
_SORTED_ALIAS_KEYS = sorted(_MENU_ALIASES_MAP.keys(), key=len, reverse=True)
_ALIAS_REGEX = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in _SORTED_ALIAS_KEYS) + r")(?!\w)",
    re.IGNORECASE,
)

_CONNECTOR_WORDS: Final[set[str]] = {"and", "with", "plus"}


def _reorder_quantities_and_items(text: str) -> str:
    """Reorder item-first quantity-last expressions to quantity-first item-last.

    Examples:
        "tea 2" -> "2 tea"
        "coffee 3" -> "3 coffee"
        "puri 4" -> "4 puri"
        "masala dosa 2" -> "2 masala dosa"
        "tea 2 coffee 3" -> "2 tea 3 coffee"
    """
    tokens = text.split()
    if not tokens:
        return text

    # Group tokens into segments
    # Each segment is either:
    # - a quantity token (digit)
    # - a connector word
    # - a list of item words
    segments: list[dict[str, any]] = []

    current_item_words: list[str] = []

    def flush_items() -> None:
        nonlocal current_item_words
        if current_item_words:
            segments.append({"type": "items", "words": current_item_words})
            current_item_words = []

    for token in tokens:
        if token.isdigit():
            flush_items()
            segments.append({"type": "quantity", "value": token})
        elif token.lower() in _CONNECTOR_WORDS:
            flush_items()
            segments.append({"type": "connector", "value": token})
        else:
            current_item_words.append(token)
    flush_items()

    # Now inspect segments to flip [items, quantity] into [quantity, items]
    reordered_segments: list[str] = []
    i = 0
    n = len(segments)
    consumed_indices: set[int] = set()

    while i < n:
        if i in consumed_indices:
            i += 1
            continue

        seg = segments[i]
        # Check pattern: items followed by quantity (and NOT preceded by an unconsumed quantity)
        if seg["type"] == "items":
            has_preceding_unconsumed_qty = (
                i > 0
                and segments[i - 1]["type"] == "quantity"
                and (i - 1) not in consumed_indices
            )
            has_following_unconsumed_qty = (
                i + 1 < n
                and segments[i + 1]["type"] == "quantity"
                and (i + 1) not in consumed_indices
            )

            if has_following_unconsumed_qty and not has_preceding_unconsumed_qty:
                qty_seg = segments[i + 1]
                # Reorder: quantity first, then item words
                reordered_segments.append(qty_seg["value"])
                reordered_segments.extend(seg["words"])
                consumed_indices.add(i + 1)
                i += 1
                continue

            reordered_segments.extend(seg["words"])
            i += 1
        elif seg["type"] == "quantity":
            reordered_segments.append(seg["value"])
            i += 1
        elif seg["type"] == "connector":
            reordered_segments.append(seg["value"])
            i += 1
        else:
            i += 1

    return " ".join(reordered_segments)


def normalize(text: str) -> str:
    """Normalize raw spoken text to standard digit-quantity + canonical menu names.

    Phases performed:
    1. Number Normalization: Convert all spoken numbers/homophones to numeric digits.
    2. Menu Vocabulary Alias Mapping: Replace spoken aliases with canonical names.
    3. Multiword Parsing Reordering: Reorder "tea 2" -> "2 tea".
    4. Correction Logging: Log changes made.

    Args:
        text: Input transcript string.

    Returns:
        Normalized transcript string.
    """
    if not text or not text.strip():
        return text

    original_text = text
    working = text.lower()

    # Pass 1: Number normalization into digits
    working = _NUMBER_REGEX.sub(
        lambda m: NUMBER_TO_DIGIT_MAP[m.group(0).lower()], working
    )

    # Pass 2: Menu vocabulary alias mapping
    working = _ALIAS_REGEX.sub(
        lambda m: _MENU_ALIASES_MAP[m.group(0).lower()], working
    )

    # Pass 3: Multiword / Quantity position reordering
    working = _reorder_quantities_and_items(working)

    if working != original_text:
        logger.info(
            "Speech Normalization applied:\nOriginal: %r\nCorrected: %r\nReason: Vocabulary alias / Number normalization / Quantity reordering",
            original_text,
            working,
        )

    return working
