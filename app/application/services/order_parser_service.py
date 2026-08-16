"""Order Parser Service.

Converts raw recognized speech text into structured order data —
a list of {"item": ..., "quantity": ...} entries, unknown items, and parser confidence.

Segment-based Tokenization Architecture:
  1. Tokenizer splits full transcript into independent order segments by quantity
     and connector word boundaries.
  2. Each segment is parsed independently so that a failure or unknown item in one
     segment NEVER stops or drops the remaining valid items.
  3. Returns recognized items, collected unknown items, total segments, and parser confidence.
"""

import re
from typing import Final

from app.core.logging import get_logger
from app.domain.entities.order import ParseResult, ParsedOrderItem
from app.domain.interfaces.order_parser_interface import OrderParserInterface

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Number-word vocabularies & Connectors
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
        """Tokenize transcript into independent order segments, parse each segment,
        collect unknown items separately, and compute parser confidence.

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

        # -------------------------------------------------------------------
        # Step 1: Tokenize transcript into independent order segments
        # -------------------------------------------------------------------
        segments: list[list[str]] = []
        curr_segment: list[str] = []

        for token in raw_tokens:
            token_lower = token.lower()
            is_qty = token.isdigit() or token_lower in QUANTITY_WORDS
            is_conn = token_lower in _CONNECTOR_WORDS

            # Quantity or connector boundary marks start of a new segment
            if (is_qty or is_conn) and curr_segment:
                segments.append(curr_segment)
                curr_segment = []

            if not is_conn:
                curr_segment.append(token)

        if curr_segment:
            segments.append(curr_segment)

        # -------------------------------------------------------------------
        # Step 2: Parse each segment independently
        # -------------------------------------------------------------------
        recognized_items: list[ParsedOrderItem] = []
        unknown_items: list[str] = []

        vocab_words: set[str] = set()
        if vocabulary:
            for v_name in vocabulary:
                for v_word in v_name.lower().split():
                    vocab_words.add(v_word)

        for seg_idx, seg in enumerate(segments):
            try:
                quantity: int = 1
                item_words: list[str] = []

                for token in seg:
                    token_lower = token.lower()
                    if token.isdigit():
                        quantity = int(token)
                    elif token_lower in QUANTITY_WORDS:
                        quantity = QUANTITY_WORDS[token_lower]
                    elif token_lower in _CONNECTOR_WORDS:
                        continue
                    else:
                        item_words.append(token_lower)

                if not item_words:
                    continue

                item_name = " ".join(item_words).strip()

                # Flag unknown items
                if (
                    item_name.lower() == "unknown menu item"
                    or item_name.lower().startswith("unknown item")
                    or "unknown" in item_name.lower()
                    or (vocab_words and not any(w in vocab_words for w in item_name.split()))
                ):
                    unknown_items.append(item_name)
                    logger.warning(
                        "Segment %d parsed as unknown item: %r (quantity: %d)",
                        seg_idx + 1,
                        item_name,
                        quantity,
                    )
                else:
                    recognized_items.append({"item": item_name, "quantity": quantity})

            except Exception as exc:
                seg_str = " ".join(seg)
                unknown_items.append(seg_str)
                logger.error(
                    "Error parsing segment %d (%r): %s — continuing remaining segments",
                    seg_idx + 1,
                    seg_str,
                    exc,
                )

        # -------------------------------------------------------------------
        # Step 3: Compute parser confidence and return result
        # -------------------------------------------------------------------
        total_segments = len(segments)
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
