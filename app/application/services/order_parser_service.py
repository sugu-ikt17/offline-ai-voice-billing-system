"""Order Parser Service.

Converts raw recognized speech text into structured order data —
a list of {"item": ..., "quantity": ...} entries, unknown items, and parser confidence.

Strict Adjacency Architecture:
  1. Quantity and menu item MUST be directly adjacent in the transcript.
     - NUMBER + MENU ITEM ("2 dosa", "two dosa")
     - MENU ITEM + NUMBER ("dosa 2", "dosa two")
  2. Multi-word menu items in vocabulary are matched as complete phrases
     only when directly adjacent to a quantity or standalone.
  3. Conversational noise words between quantity and item break adjacency.
  4. Returns recognized items, unknown items, total segments, and confidence.
"""

import re
from typing import Final

from app.core.logging import get_logger
from app.domain.entities.order import ParseResult, ParsedOrderItem
from app.domain.interfaces.order_parser_interface import OrderParserInterface

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Number-word vocabularies & Connectors & Noise words
# ---------------------------------------------------------------------------

ENGLISH_WORDS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

TAMIL_WORDS: Final[dict[str, int]] = {
    "oru": 1,
    "onnu": 1,
    "rendu": 2,
    "moonu": 3,
    "naalu": 4,
    "anju": 5,
    "aaru": 6,
    "ezhu": 7,
    "ettu": 8,
    "ombodhu": 9,
    "pathu": 10,
}

QUANTITY_WORDS: Final[dict[str, int]] = {
    **ENGLISH_WORDS,
    **TAMIL_WORDS,
}

_CONNECTOR_WORDS: Final[set[str]] = {"and", "with", "plus"}

_IGNORE_WORDS: Final[set[str]] = {
    "thank", "thanks", "you", "your", "welcome",
    "i", "im", "i'm", "am", "is", "are", "was", "were",
    "please", "okay", "ok", "yes", "no", "yeah", "nope",
    "the", "a", "an", "this", "that", "these", "those",
    "give", "me", "want", "need", "like", "have", "take",
    "also", "too", "or", "for", "all", "of", "it", "soul",
    "just", "only", "more", "sir", "maam", "madam", "boss",
    "bhai", "anna", "akka", "ill", "i'll",
}

# Regex pattern for extracting numeric boundaries and word tokens safely
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b\d+\b|\b[^\W\d_]+\b")


class OrderParserService(OrderParserInterface):
    """Parses a raw speech transcript into structured order items and unknown items."""

    def parse(self, text: str) -> list[ParsedOrderItem]:
        """Convert a speech transcript string into structured order items.

        Maintains 100% backward compatibility with OrderParserInterface.
        """
        result = self.parse_with_details(text)
        return result.recognized_items

    def parse_with_details(
        self, text: str, vocabulary: list[str] | None = None
    ) -> ParseResult:
        """Tokenize transcript and parse orders using strict local adjacency.

        Valid patterns:
          1. NUMBER + MENU ITEM ("2 dosa", "two dosa")
          2. MENU ITEM + NUMBER ("dosa 2", "dosa two")
          3. Standalone MENU ITEM (quantity defaults to 1) if no disconnected quantity is nearby.

        Arbitrary words between quantity and menu item invalidate the pair.
        Conversational noise words ("thank you", "I am") are ignored and never turned into unknown items.

        Args:
            text: Transcript string.
            vocabulary: Optional list of active menu item names for filtering.

        Returns:
            ParseResult containing recognized_items, unknown_items, confidence,
            and total_segments.
        """
        if not text or not text.strip():
            return ParseResult(
                recognized_items=[], unknown_items=[], confidence=1.0, total_segments=0
            )

        raw_tokens: list[str] = _TOKEN_PATTERN.findall(text)
        if not raw_tokens:
            return ParseResult(
                recognized_items=[], unknown_items=[], confidence=1.0, total_segments=0
            )

        tokens_lower = [t.lower() for t in raw_tokens]
        n = len(tokens_lower)

        def _get_qty(token: str) -> int | None:
            if token.isdigit():
                return int(token)
            if token in QUANTITY_WORDS:
                return QUANTITY_WORDS[token]
            return None

        def _is_conn(token: str) -> bool:
            return token in _CONNECTOR_WORDS

        def _is_ignore(token: str) -> bool:
            return token in _IGNORE_WORDS

        # Build vocabulary phrase list
        vocab_phrases: list[tuple[tuple[str, ...], str]] = []
        if vocabulary:
            for v_item in vocabulary:
                v_clean = v_item.strip()
                if not v_clean:
                    continue
                v_toks = tuple(v_clean.lower().split())
                vocab_phrases.append((v_toks, v_clean.lower()))
            vocab_phrases.sort(key=lambda x: len(x[0]), reverse=True)

        recognized_items: list[ParsedOrderItem] = []
        unknown_items: list[str] = []
        consumed = [False] * n

        # Phase 1: Match vocabulary items (or item candidates) with ADJACENT quantities
        i = 0
        while i < n:
            if consumed[i]:
                i += 1
                continue

            matched_phrase: str | None = None
            phrase_len = 0

            if vocabulary:
                for p_tokens, p_name in vocab_phrases:
                    p_len = len(p_tokens)
                    if i + p_len <= n:
                        if tuple(tokens_lower[i : i + p_len]) == p_tokens:
                            if not any(consumed[i + k] for k in range(p_len)):
                                matched_phrase = p_name
                                phrase_len = p_len
                                break

            if matched_phrase is not None:
                # Check adjacent quantity before (i - 1)
                qty_before = None
                if i - 1 >= 0 and not consumed[i - 1]:
                    qty_before = _get_qty(tokens_lower[i - 1])

                # Check adjacent quantity after (i + phrase_len)
                qty_after = None
                if i + phrase_len < n and not consumed[i + phrase_len]:
                    qty_after = _get_qty(tokens_lower[i + phrase_len])

                if qty_before is not None:
                    recognized_items.append({"item": matched_phrase, "quantity": qty_before})
                    consumed[i - 1] = True
                    for k in range(phrase_len):
                        consumed[i + k] = True
                    i += phrase_len
                    continue

                if qty_after is not None:
                    recognized_items.append({"item": matched_phrase, "quantity": qty_after})
                    for k in range(phrase_len):
                        consumed[i + k] = True
                    consumed[i + phrase_len] = True
                    i += phrase_len + 1
                    continue

                # Check if there is an unconsumed quantity nearby separated by non-item words
                has_disconnected_qty_before = False
                for j in range(i - 1, -1, -1):
                    if consumed[j] or _is_conn(tokens_lower[j]):
                        break
                    if _get_qty(tokens_lower[j]) is not None:
                        has_disconnected_qty_before = True
                        break

                has_disconnected_qty_after = False
                for j in range(i + phrase_len, n):
                    if consumed[j] or _is_conn(tokens_lower[j]):
                        break
                    if _get_qty(tokens_lower[j]) is not None:
                        has_disconnected_qty_after = True
                        break

                if not has_disconnected_qty_before and not has_disconnected_qty_after:
                    recognized_items.append({"item": matched_phrase, "quantity": 1})
                    for k in range(phrase_len):
                        consumed[i + k] = True
                    i += phrase_len
                    continue
                else:
                    # Disconnected quantity phrase: e.g. "2 thank you dosa" -> ignore
                    i += phrase_len
                    continue

            i += 1

        # Phase 2: If vocabulary was NOT provided, match adjacent QTY + item or ITEM + QTY left-to-right
        if not vocabulary:
            i = 0
            while i < n:
                if consumed[i] or _is_conn(tokens_lower[i]):
                    i += 1
                    continue

                # Pattern A: ITEM + QTY (e.g. "tea 2")
                if (
                    _get_qty(tokens_lower[i]) is None
                    and not _is_ignore(tokens_lower[i])
                    and i + 1 < n
                    and not consumed[i + 1]
                    and _get_qty(tokens_lower[i + 1]) is not None
                ):
                    item_name = raw_tokens[i].lower()
                    qty_val = _get_qty(tokens_lower[i + 1])
                    recognized_items.append({"item": item_name, "quantity": qty_val})
                    consumed[i] = True
                    consumed[i + 1] = True
                    i += 2
                    continue

                # Pattern B: QTY + ITEM (e.g. "2 tea" or "2 masala dosa")
                if (
                    _get_qty(tokens_lower[i]) is not None
                    and i + 1 < n
                    and not consumed[i + 1]
                    and _get_qty(tokens_lower[i + 1]) is None
                    and not _is_conn(tokens_lower[i + 1])
                    and not _is_ignore(tokens_lower[i + 1])
                ):
                    qty_val = _get_qty(tokens_lower[i])
                    item_words = []
                    j = i + 1
                    while (
                        j < n
                        and not consumed[j]
                        and _get_qty(tokens_lower[j]) is None
                        and not _is_conn(tokens_lower[j])
                        and not _is_ignore(tokens_lower[j])
                    ):
                        item_words.append(raw_tokens[j])
                        j += 1

                    if item_words:
                        item_name = " ".join(item_words).lower()
                        recognized_items.append({"item": item_name, "quantity": qty_val})
                        consumed[i] = True
                        for k in range(i + 1, j):
                            consumed[k] = True
                        i = j
                        continue

                i += 1

            # Standalone item when vocabulary is None (e.g. "dosa", "dosa and tea")
            i = 0
            while i < n:
                if (
                    consumed[i]
                    or _get_qty(tokens_lower[i]) is not None
                    or _is_conn(tokens_lower[i])
                    or _is_ignore(tokens_lower[i])
                ):
                    i += 1
                    continue

                # Check if there's any unconsumed quantity token anywhere in the phrase
                has_unconsumed_qty = any(_get_qty(tokens_lower[k]) is not None and not consumed[k] for k in range(n))
                if has_unconsumed_qty:
                    i += 1
                    continue

                item_words = []
                j = i
                while (
                    j < n
                    and not consumed[j]
                    and _get_qty(tokens_lower[j]) is None
                    and not _is_conn(tokens_lower[j])
                    and not _is_ignore(tokens_lower[j])
                ):
                    item_words.append(raw_tokens[j])
                    j += 1
                if item_words:
                    item_name = " ".join(item_words).lower()
                    recognized_items.append({"item": item_name, "quantity": 1})
                    for k in range(i, j):
                        consumed[k] = True
                    i = j
                    continue
                i += 1

        # Phase 3: Identify Unknown Items (Quantity adjacent to non-menu, non-ignore word)
        if vocabulary:
            i = 0
            while i < n:
                if not consumed[i]:
                    qty_val = _get_qty(tokens_lower[i])
                    if qty_val is not None:
                        # Check next token
                        if i + 1 < n and not consumed[i + 1]:
                            target = tokens_lower[i + 1]
                            if not _is_conn(target) and not _is_ignore(target) and _get_qty(target) is None:
                                unknown_items.append(target)
                                consumed[i] = True
                                consumed[i + 1] = True
                                i += 2
                                continue
                        # Check prev token
                        if i - 1 >= 0 and not consumed[i - 1]:
                            target = tokens_lower[i - 1]
                            if not _is_conn(target) and not _is_ignore(target) and _get_qty(target) is None:
                                unknown_items.append(target)
                                consumed[i - 1] = True
                                consumed[i] = True
                                i += 1
                                continue
                i += 1

        total_segments = len(recognized_items) + len(unknown_items)
        confidence = (
            round(len(recognized_items) / total_segments, 2)
            if total_segments > 0
            else 1.0
        )

        logger.info(
            "Parsed order transcript: total_segments=%d recognized=%d unknown=%d confidence=%.2f",
            total_segments,
            len(recognized_items),
            len(unknown_items),
            confidence,
        )

        return ParseResult(
            recognized_items=recognized_items,
            unknown_items=unknown_items,
            confidence=confidence,
            total_segments=total_segments,
        )
