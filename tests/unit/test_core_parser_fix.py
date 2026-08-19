"""Regression tests for Core Voice Billing Parser and Vocabulary Corrector Fix.

Tests:
  1. "2 tea 3 dosa" (PATTERN A: QTY ITEM)
  2. "tea 2 dosa 3" (PATTERN B: ITEM QTY)
  3. "2 tea 3 dosa 3 coffee"
  4. "3 coffee 2 tea 4 samosa"
  5. "4 dosa 5 samosa 3 tea"
  6. "2 tea 2 dosa 3 coffee"
  7. "3 coffee I'll have all the soul of it" (Noise/Conversational word resilience)
  8. Assertion that "Unknown Menu Item" is NEVER inserted into corrected transcript.
"""

import pytest
from app.application.services.order_parser_service import OrderParserService
from app.application.services.menu_vocabulary_corrector import MenuVocabularyCorrector

VOCABULARY = ["tea", "coffee", "dosa", "idli", "samosa", "puri"]


@pytest.fixture
def parser() -> OrderParserService:
    return OrderParserService()


@pytest.fixture
def corrector() -> MenuVocabularyCorrector:
    return MenuVocabularyCorrector()


def test_pattern_a_qty_item(parser):
    """Test 1: '2 tea 3 dosa' -> tea=2, dosa=3."""
    res = parser.parse_with_details("2 tea 3 dosa", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 2
    assert items[0] == {"item": "tea", "quantity": 2}
    assert items[1] == {"item": "dosa", "quantity": 3}


def test_pattern_b_item_qty(parser):
    """Test 2: 'tea 2 dosa 3' -> tea=2, dosa=3."""
    res = parser.parse_with_details("tea 2 dosa 3", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 2
    assert items[0] == {"item": "tea", "quantity": 2}
    assert items[1] == {"item": "dosa", "quantity": 3}


def test_3_items_tea_dosa_coffee(parser):
    """Test 3: '2 tea 3 dosa 3 coffee' -> tea=2, dosa=3, coffee=3."""
    res = parser.parse_with_details("2 tea 3 dosa 3 coffee", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 3
    assert items[0] == {"item": "tea", "quantity": 2}
    assert items[1] == {"item": "dosa", "quantity": 3}
    assert items[2] == {"item": "coffee", "quantity": 3}


def test_3_items_coffee_tea_samosa(parser):
    """Test 4: '3 coffee 2 tea 4 samosa' -> coffee=3, tea=2, samosa=4."""
    res = parser.parse_with_details("3 coffee 2 tea 4 samosa", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 3
    assert items[0] == {"item": "coffee", "quantity": 3}
    assert items[1] == {"item": "tea", "quantity": 2}
    assert items[2] == {"item": "samosa", "quantity": 4}


def test_3_items_dosa_samosa_tea(parser):
    """Test 5: '4 dosa 5 samosa 3 tea' -> dosa=4, samosa=5, tea=3."""
    res = parser.parse_with_details("4 dosa 5 samosa 3 tea", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 3
    assert items[0] == {"item": "dosa", "quantity": 4}
    assert items[1] == {"item": "samosa", "quantity": 5}
    assert items[2] == {"item": "tea", "quantity": 3}


def test_3_items_tea_dosa_coffee_variant(parser):
    """Test 6: '2 tea 2 dosa 3 coffee' -> tea=2, dosa=2, coffee=3."""
    res = parser.parse_with_details("2 tea 2 dosa 3 coffee", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 3
    assert items[0] == {"item": "tea", "quantity": 2}
    assert items[1] == {"item": "dosa", "quantity": 2}
    assert items[2] == {"item": "coffee", "quantity": 3}


def test_noise_words_resilience(parser):
    """Test 7: '3 coffee I'll have all the soul of it' -> coffee=3."""
    res = parser.parse_with_details("3 coffee I'll have all the soul of it", vocabulary=VOCABULARY)
    items = res.recognized_items
    assert len(items) == 1
    assert items[0] == {"item": "coffee", "quantity": 3}


def test_unknown_menu_item_literal_never_inserted(corrector):
    """Test 8: Verify 'Unknown Menu Item' literal is NEVER inserted into corrected transcript."""
    transcripts = [
        "2 tea 2 dosse 3 coffee i'll have all the soul of it",
        "3 coffee kasi",
        "random unrecognized tokens xyz abc",
    ]
    for tr in transcripts:
        corrected = corrector.correct(tr, VOCABULARY)
        assert "Unknown Menu Item" not in corrected
        assert "unknown menu item" not in corrected.lower()
