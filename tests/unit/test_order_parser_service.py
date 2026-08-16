"""Unit tests for upgraded segment-based OrderParserService.

Covers:
  - Independent segment tokenization
  - Parsing 2 items, 5 items, 10 items, 20 items, 50 items
  - Error isolation: valid items parsed even when unknown items are present
  - Detailed ParseResult (recognized_items, unknown_items, confidence, total_segments)
  - Backward compatibility with parse() returning list[ParsedOrderItem]
"""

import pytest

from app.application.services.order_parser_service import OrderParserService
from app.domain.entities.order import ParseResult


@pytest.fixture
def parser() -> OrderParserService:
    return OrderParserService()


# ---------------------------------------------------------------------------
# 1. Test 2, 5, 10, 20, 50 items as required in user prompt
# ---------------------------------------------------------------------------


def test_parse_2_items(parser):
    """Test parsing 2 items: '4 tea 3 coffee'."""
    text = "4 tea 3 coffee"
    res = parser.parse_with_details(text)

    assert isinstance(res, ParseResult)
    assert len(res.recognized_items) == 2
    assert res.recognized_items[0] == {"item": "tea", "quantity": 4}
    assert res.recognized_items[1] == {"item": "coffee", "quantity": 3}
    assert len(res.unknown_items) == 0
    assert res.confidence == 1.0
    assert res.total_segments == 2


def test_parse_5_items(parser):
    """Test parsing 5 items: '4 tea 3 coffee 9 dosa 2 idli 10 samosa'."""
    text = "4 tea 3 coffee 9 dosa 2 idli 10 samosa"
    res = parser.parse_with_details(text)

    assert len(res.recognized_items) == 5
    items = [item["item"] for item in res.recognized_items]
    quantities = [item["quantity"] for item in res.recognized_items]

    assert items == ["tea", "coffee", "dosa", "idli", "samosa"]
    assert quantities == [4, 3, 9, 2, 10]
    assert len(res.unknown_items) == 0
    assert res.confidence == 1.0
    assert res.total_segments == 5


def test_parse_10_items(parser):
    """Test parsing 10 items independently."""
    menu_sample = ["tea", "coffee", "dosa", "idli", "vada", "puri", "sambar", "upma", "pongal", "parotta"]
    parts = [f"{i+1} {menu_sample[i]}" for i in range(10)]
    text = " ".join(parts)

    res = parser.parse_with_details(text)
    assert len(res.recognized_items) == 10
    assert res.total_segments == 10
    assert res.confidence == 1.0


def test_parse_20_items(parser):
    """Test parsing 20 items independently."""
    menu_sample = ["tea", "coffee", "dosa", "idli", "vada", "puri", "sambar", "upma", "pongal", "parotta"]
    parts = [f"{i+1} {menu_sample[i % 10]}" for i in range(20)]
    text = " ".join(parts)

    res = parser.parse_with_details(text)
    assert len(res.recognized_items) == 20
    assert res.total_segments == 20
    assert res.confidence == 1.0


def test_parse_50_items(parser):
    """Test parsing 50 items independently."""
    menu_sample = ["tea", "coffee", "dosa", "idli", "vada", "puri", "sambar", "upma", "pongal", "parotta"]
    parts = [f"{i+1} {menu_sample[i % 10]}" for i in range(50)]
    text = " ".join(parts)

    res = parser.parse_with_details(text)
    assert len(res.recognized_items) == 50
    assert res.total_segments == 50
    assert res.confidence == 1.0


# ---------------------------------------------------------------------------
# 2. Test Error Isolation & Unknown Item Collection
# ---------------------------------------------------------------------------


def test_error_isolation_5_items_with_unknown(parser):
    """Verify valid items are billed even if some items are unknown.

    Input: '4 tea 3 cofeee 9 dosa 2 idlyy 10 samosa'
    'cofeee' and 'idlyy' should be collected as unknown, while tea, dosa, samosa are billed.
    """
    vocabulary = ["tea", "coffee", "dosa", "idli", "samosa"]
    text = "4 tea 3 cofeee 9 dosa 2 idlyy 10 samosa"

    res = parser.parse_with_details(text, vocabulary=vocabulary)

    recognized_names = [item["item"] for item in res.recognized_items]
    assert "tea" in recognized_names
    assert "dosa" in recognized_names
    assert "samosa" in recognized_names

    assert "cofeee" in res.unknown_items
    assert "idlyy" in res.unknown_items
    assert len(res.recognized_items) == 3
    assert len(res.unknown_items) == 2
    assert res.total_segments == 5
    assert res.confidence == 0.6  # 3 / 5 = 0.6


def test_error_isolation_50_items_with_unknowns(parser):
    """Verify 50 items with 5 unknown items: 45 valid items must still be parsed and billed."""
    menu_sample = ["tea", "coffee", "dosa", "idli", "vada", "puri", "sambar", "upma", "pongal", "parotta"]
    parts = []
    for i in range(1, 51):
        if i % 10 == 0:
            parts.append(f"{i} baditem")
        else:
            parts.append(f"{i} {menu_sample[(i - 1) % 10]}")

    text = " ".join(parts)
    res = parser.parse_with_details(text, vocabulary=menu_sample)

    assert len(res.recognized_items) == 45
    assert len(res.unknown_items) == 5
    assert res.total_segments == 50
    assert res.confidence == 0.9  # 45 / 50 = 0.90


# ---------------------------------------------------------------------------
# 3. Test Backward Compatibility
# ---------------------------------------------------------------------------


def test_parse_backward_compatibility(parser):
    """parse() must return list[ParsedOrderItem] for full backward compatibility."""
    items = parser.parse("2 dosa 1 tea")

    assert isinstance(items, list)
    assert items == [
        {"item": "dosa", "quantity": 2},
        {"item": "tea", "quantity": 1},
    ]
