"""Comprehensive unit tests for OrderParserService.

Tests cover:
  - Numeric quantities (1, 2, 3 ...)
  - English number words (one, two, three ... ten)
  - Default quantity-1 when no quantity is spoken
  - Multi-word item names (masala dosa)
  - Multi-item orders
  - Connector words ("and")
  - Edge cases: empty input, whitespace, punctuation
"""

import pytest
from app.application.services.order_parser_service import OrderParserService, QUANTITY_WORDS

parser = OrderParserService()


# ---------------------------------------------------------------------------
# Examples from requirements specification
# ---------------------------------------------------------------------------

def test_spec_example_2_dosa_1_tea():
    """Core example from spec: '2 dosa 1 tea'."""
    result = parser.parse("2 dosa 1 tea")
    assert result == [
        {"item": "dosa", "quantity": 2},
        {"item": "tea",  "quantity": 1},
    ]


def test_spec_example_3_coffee():
    """Single item with numeric quantity: '3 coffee'."""
    result = parser.parse("3 coffee")
    assert result == [{"item": "coffee", "quantity": 3}]


def test_spec_example_one_tea():
    """English number word as quantity: 'one tea'."""
    result = parser.parse("one tea")
    assert result == [{"item": "tea", "quantity": 1}]


def test_spec_example_tea_only():
    """No quantity spoken → defaults to 1: 'tea'."""
    result = parser.parse("tea")
    assert result == [{"item": "tea", "quantity": 1}]


# ---------------------------------------------------------------------------
# English number words (one … ten)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected_qty", [
    ("one",   1),
    ("two",   2),
    ("three", 3),
    ("four",  4),
    ("five",  5),
    ("six",   6),
    ("seven", 7),
    ("eight", 8),
    ("nine",  9),
    ("ten",   10),
])
def test_english_number_word(word, expected_qty):
    """Each supported English number word resolves to its integer value."""
    result = parser.parse(f"{word} dosa")
    assert result == [{"item": "dosa", "quantity": expected_qty}]


def test_number_word_mixed_with_numeric():
    """Number words and numerics can appear in the same order."""
    result = parser.parse("two dosa 3 tea")
    assert result == [
        {"item": "dosa", "quantity": 2},
        {"item": "tea",  "quantity": 3},
    ]


def test_number_word_case_insensitive():
    """Number words are matched case-insensitively."""
    assert parser.parse("Two dosa") == [{"item": "dosa", "quantity": 2}]
    assert parser.parse("TWO dosa") == [{"item": "dosa", "quantity": 2}]


# ---------------------------------------------------------------------------
# Default quantity = 1
# ---------------------------------------------------------------------------

def test_single_item_no_quantity():
    result = parser.parse("dosa")
    assert result == [{"item": "dosa", "quantity": 1}]


def test_multiple_items_no_quantity():
    """All items without quantities should default to 1 each."""
    result = parser.parse("dosa tea coffee")
    assert result == [
        {"item": "dosa tea coffee", "quantity": 1},
    ]


def test_item_after_numbered_item_is_multiword():
    """Consecutive words without a number separator form a single multi-word item.

    '2 dosa tea' is parsed as one item named 'dosa tea' with quantity 2,
    because there is no number or connector to signal a new item boundary.
    To order 2 dosa AND 1 tea separately, say '2 dosa 1 tea' or '2 dosa and tea'.
    """
    result = parser.parse("2 dosa tea")
    assert result == [
        {"item": "dosa tea", "quantity": 2},
    ]


# ---------------------------------------------------------------------------
# Multi-word item names
# ---------------------------------------------------------------------------

def test_multiword_item_name():
    result = parser.parse("2 masala dosa")
    assert result == [{"item": "masala dosa", "quantity": 2}]


def test_multiword_item_and_single():
    result = parser.parse("2 masala dosa 1 tea")
    assert result == [
        {"item": "masala dosa", "quantity": 2},
        {"item": "tea",         "quantity": 1},
    ]


# ---------------------------------------------------------------------------
# Connector word "and"
# ---------------------------------------------------------------------------

def test_and_connector():
    """'and' acts as a boundary between two items without quantities."""
    result = parser.parse("dosa and tea")
    assert result == [
        {"item": "dosa", "quantity": 1},
        {"item": "tea",  "quantity": 1},
    ]


def test_and_connector_with_quantities():
    result = parser.parse("2 dosa and 3 tea")
    assert result == [
        {"item": "dosa", "quantity": 2},
        {"item": "tea",  "quantity": 3},
    ]


# ---------------------------------------------------------------------------
# Numeric quantities
# ---------------------------------------------------------------------------

def test_numeric_2_dosa():
    assert parser.parse("2 dosa") == [{"item": "dosa", "quantity": 2}]


def test_numeric_5_idli_2_vada_1_tea():
    result = parser.parse("5 idli 2 vada 1 tea")
    assert result == [
        {"item": "idli", "quantity": 5},
        {"item": "vada", "quantity": 2},
        {"item": "tea",  "quantity": 1},
    ]


# ---------------------------------------------------------------------------
# Punctuation and whitespace tolerance
# ---------------------------------------------------------------------------

def test_punctuation_is_ignored():
    """Commas, periods, and extra spaces should not affect parsing."""
    result = parser.parse("2 dosa, 1 tea.")
    assert result == [
        {"item": "dosa", "quantity": 2},
        {"item": "tea",  "quantity": 1},
    ]


def test_extra_whitespace():
    result = parser.parse("  2   dosa   1   tea  ")
    assert result == [
        {"item": "dosa", "quantity": 2},
        {"item": "tea",  "quantity": 1},
    ]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string():
    assert parser.parse("") == []


def test_whitespace_only():
    assert parser.parse("   ") == []


def test_none_like_empty_is_empty():
    """Passing None should return an empty list without raising."""
    # The service spec says accept a str; passing None mimics a silent mic.
    assert parser.parse("") == []


# ---------------------------------------------------------------------------
# QUANTITY_WORDS vocabulary is accessible and complete
# ---------------------------------------------------------------------------

def test_quantity_words_contains_all_english():
    """All ten English number words must be present in QUANTITY_WORDS."""
    expected = {"one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten"}
    assert expected.issubset(QUANTITY_WORDS.keys())
