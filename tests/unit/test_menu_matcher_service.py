"""Comprehensive unit tests for MenuMatcherService.

Covers:
  - Canonical spec example (Dosa + Tea from requirements)
  - Exact match (case-insensitive, whitespace-stripped)
  - Shared-word match  ("dosa" → "Masala Dosa")
  - Fuzzy match        ("tee" → "Tea")
  - Unmatched items returned in MatchResult.unmatched_items
  - Mixed order: some matched, some unmatched
  - Empty menu → all items unmatched
  - Empty parsed_items list → empty MatchResult
  - Quantity and subtotal (line_total) calculation
  - to_dict() wire format for Bill Generator
"""

import pytest

from app.application.services.menu_matcher_service import MenuMatcherService
from app.domain.entities.order import MatchResult
from app.infrastructure.database.models.menu_item_model import MenuItemModel
from app.infrastructure.database.repositories.menu_repository import MenuRepository


# ---------------------------------------------------------------------------
# Fixture: in-memory test DB seeded with spec menu
# ---------------------------------------------------------------------------

@pytest.fixture()
def menu_repo(client):
    """Isolated in-memory DB seeded with the menu from the spec.

    Menu:
        ID | Name         | Price
        ---+---+-----------+-------
        1  | Dosa         | 40
        2  | Tea          | 15
        3  | Coffee       | 20
        4  | Masala Dosa  | 55
        5  | Filter Coffee| 25
    """
    from tests.conftest import TestSessionLocal

    db = TestSessionLocal()
    repo = MenuRepository(db)
    repo.create(MenuItemModel(name="Dosa",          price=40.0))
    repo.create(MenuItemModel(name="Tea",           price=15.0))
    repo.create(MenuItemModel(name="Coffee",        price=20.0))
    repo.create(MenuItemModel(name="Masala Dosa",   price=55.0))
    repo.create(MenuItemModel(name="Filter Coffee", price=25.0))
    yield repo
    db.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_matcher(repo) -> MenuMatcherService:
    return MenuMatcherService(repo)


# ---------------------------------------------------------------------------
# Spec example
# ---------------------------------------------------------------------------

def test_spec_example_2_dosa_1_tea(menu_repo):
    """Canonical requirement example: 2 dosa + 1 tea → matched with subtotals."""
    matcher = make_matcher(menu_repo)
    result = matcher.match([
        {"item": "dosa", "quantity": 2},
        {"item": "tea",  "quantity": 1},
    ])

    assert isinstance(result, MatchResult)
    assert result.unmatched_items == []
    assert len(result.matched_items) == 2

    dosa = next(i for i in result.matched_items if "Dosa" in i.name)
    tea  = next(i for i in result.matched_items if i.name == "Tea")

    assert dosa.quantity   == 2
    assert dosa.unit_price == 40.0
    assert dosa.line_total == 80.0   # 2 × 40

    assert tea.quantity   == 1
    assert tea.unit_price == 15.0
    assert tea.line_total == 15.0   # 1 × 15


# ---------------------------------------------------------------------------
# Tier 1 — exact match
# ---------------------------------------------------------------------------

def test_exact_match_case_insensitive(menu_repo):
    result = make_matcher(menu_repo).match([{"item": "TEA", "quantity": 3}])
    assert len(result.matched_items) == 1
    assert result.matched_items[0].name == "Tea"
    assert result.matched_items[0].quantity == 3


def test_exact_match_strips_whitespace(menu_repo):
    result = make_matcher(menu_repo).match([{"item": "  Tea  ", "quantity": 1}])
    assert len(result.matched_items) == 1
    assert result.matched_items[0].name == "Tea"


def test_exact_match_single_word_coffee(menu_repo):
    result = make_matcher(menu_repo).match([{"item": "coffee", "quantity": 2}])
    assert result.matched_items[0].name == "Coffee"
    assert result.matched_items[0].line_total == 40.0  # 2 × 20


# ---------------------------------------------------------------------------
# Tier 2 — shared-word match
# ---------------------------------------------------------------------------

def test_shared_word_match_bare_dosa(menu_repo):
    """'dosa' resolves to 'Dosa' (exact single-word match beats multi-word)."""
    result = make_matcher(menu_repo).match([{"item": "dosa", "quantity": 1}])
    assert len(result.matched_items) == 1
    # 'Dosa' is shorter than 'Masala Dosa' — exact match wins tier 1.
    assert "Dosa" in result.matched_items[0].name


def test_shared_word_match_masala(menu_repo):
    """'masala' resolves to 'Masala Dosa' via shared-word tier."""
    result = make_matcher(menu_repo).match([{"item": "masala", "quantity": 2}])
    assert len(result.matched_items) == 1
    assert result.matched_items[0].name == "Masala Dosa"
    assert result.matched_items[0].line_total == 110.0  # 2 × 55


def test_shared_word_match_filter(menu_repo):
    """'filter' resolves to 'Filter Coffee'."""
    result = make_matcher(menu_repo).match([{"item": "filter", "quantity": 1}])
    assert result.matched_items[0].name == "Filter Coffee"


# ---------------------------------------------------------------------------
# Tier 3 — fuzzy match
# ---------------------------------------------------------------------------

def test_fuzzy_match_minor_typo(menu_repo):
    """'tee' is close enough to 'Tea' to match via difflib."""
    result = make_matcher(menu_repo).match([{"item": "tee", "quantity": 1}])
    assert len(result.matched_items) == 1
    assert result.matched_items[0].name == "Tea"


# ---------------------------------------------------------------------------
# Unmatched items
# ---------------------------------------------------------------------------

def test_unmatched_item_returned_in_result(menu_repo):
    """Items that match no menu entry should appear in unmatched_items."""
    result = make_matcher(menu_repo).match([{"item": "pizza", "quantity": 1}])
    assert result.matched_items   == []
    assert result.unmatched_items == ["pizza"]


def test_mixed_order_some_matched_some_not(menu_repo):
    """Partial order: matched items in matched_items, unknown in unmatched_items."""
    result = make_matcher(menu_repo).match([
        {"item": "tea",   "quantity": 1},
        {"item": "pizza", "quantity": 2},
        {"item": "dosa",  "quantity": 3},
    ])

    matched_names = {i.name for i in result.matched_items}
    assert "Tea"  in matched_names
    assert "pizza" not in matched_names

    assert "pizza" in result.unmatched_items
    assert "tea"   not in result.unmatched_items
    assert "dosa"  not in result.unmatched_items


def test_all_unmatched(menu_repo):
    result = make_matcher(menu_repo).match([
        {"item": "burger", "quantity": 1},
        {"item": "pizza",  "quantity": 2},
    ])
    assert result.matched_items   == []
    assert set(result.unmatched_items) == {"burger", "pizza"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_parsed_list_returns_empty_result(menu_repo):
    result = make_matcher(menu_repo).match([])
    assert result.matched_items   == []
    assert result.unmatched_items == []


def test_empty_menu_all_items_unmatched(client):
    """When the menu is empty every parsed item lands in unmatched_items."""
    from tests.conftest import TestSessionLocal

    db   = TestSessionLocal()
    repo = MenuRepository(db)
    result = make_matcher(repo).match([
        {"item": "dosa", "quantity": 1},
        {"item": "tea",  "quantity": 2},
    ])
    assert result.matched_items   == []
    assert set(result.unmatched_items) == {"dosa", "tea"}
    db.close()


# ---------------------------------------------------------------------------
# to_dict() — Bill Generator wire format
# ---------------------------------------------------------------------------

def test_to_dict_matches_bill_generator_format(menu_repo):
    """OrderItem.to_dict() must produce the canonical Bill Generator shape."""
    result = make_matcher(menu_repo).match([{"item": "tea", "quantity": 2}])

    assert len(result.matched_items) == 1
    d = result.matched_items[0].to_dict()

    assert set(d.keys()) == {"menu_id", "name", "price", "quantity", "subtotal"}
    assert d["name"]     == "Tea"
    assert d["price"]    == 15.0
    assert d["quantity"] == 2
    assert d["subtotal"] == 30.0   # 2 × 15


# ---------------------------------------------------------------------------
# menu_item_id field
# ---------------------------------------------------------------------------

def test_matched_item_has_menu_item_id(menu_repo):
    """Matched items must carry the DB primary key for downstream use."""
    result = make_matcher(menu_repo).match([{"item": "coffee", "quantity": 1}])
    assert result.matched_items[0].menu_item_id is not None
    assert isinstance(result.matched_items[0].menu_item_id, int)
