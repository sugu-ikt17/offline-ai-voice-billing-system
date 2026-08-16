"""Comprehensive unit tests for BillGeneratorService.

Tests are grouped by concern:
  A. Spec example (canonical 2 dosa + 1 tea)
  B. Bill structure — all required fields present and correctly typed
  C. Aggregation arithmetic — item_count, total_quantity, subtotal, grand_total
  D. Bill number format
  E. Timestamp — presence and UTC timezone
  F. Discount logic — clamping, default-zero
  G. Tax logic — default-zero, custom value
  H. Unmatched items → warnings
  I. Edge cases — empty input, all unmatched, single item
  J. BillResult.to_dict() wire format
  K. BillLineItem.to_dict() wire format
  L. Error handling — None input
"""

from datetime import timezone
from unittest.mock import patch

import pytest

from app.application.services.bill_generator_service import BillGeneratorService
from app.domain.entities.bill import BillLineItem, BillResult
from app.domain.entities.order import MatchResult, OrderItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _order_item(name: str, quantity: int, price: float, menu_item_id: int = 1) -> OrderItem:
    return OrderItem(menu_item_id=menu_item_id, name=name, quantity=quantity, unit_price=price)


def _match(matched: list[OrderItem], unmatched: list[str] | None = None) -> MatchResult:
    return MatchResult(matched_items=matched, unmatched_items=unmatched or [])


def _make_generator(**kwargs) -> BillGeneratorService:
    return BillGeneratorService(**kwargs)


# Canonical spec fixture
@pytest.fixture()
def spec_match() -> MatchResult:
    """2 Dosa (₹40 each) + 1 Tea (₹15) as per spec requirements."""
    return _match([
        _order_item("Dosa", 2, 40.0, menu_item_id=1),
        _order_item("Tea",  1, 15.0, menu_item_id=2),
    ])


# ---------------------------------------------------------------------------
# A. Canonical spec example
# ---------------------------------------------------------------------------

def test_spec_example_grand_total(spec_match):
    """Grand total = 80 + 15 = 95 (no tax, no discount per spec)."""
    result = _make_generator().generate(spec_match)
    assert result.grand_total == 95.0


def test_spec_example_subtotal(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.subtotal == 95.0


def test_spec_example_item_count(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.item_count == 2


def test_spec_example_total_quantity(spec_match):
    """2 Dosa + 1 Tea = 3 total units."""
    result = _make_generator().generate(spec_match)
    assert result.total_quantity == 3


def test_spec_example_discount_zero(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.discount == 0.0


def test_spec_example_tax_zero(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.tax == 0.0


def test_spec_example_unmatched_empty(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.unmatched_items == []
    assert result.warnings        == []


# ---------------------------------------------------------------------------
# B. Bill structure — all required fields present
# ---------------------------------------------------------------------------

def test_result_is_bill_result_instance(spec_match):
    result = _make_generator().generate(spec_match)
    assert isinstance(result, BillResult)


def test_bill_number_present(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.bill_number
    assert isinstance(result.bill_number, str)


def test_date_time_present(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.date_time is not None


def test_items_are_bill_line_items(spec_match):
    result = _make_generator().generate(spec_match)
    for item in result.items:
        assert isinstance(item, BillLineItem)


def test_all_spec_fields_in_to_dict(spec_match):
    d = _make_generator().generate(spec_match).to_dict()
    bill = d["bill"]
    for key in ("bill_number", "date_time", "items",
                "item_count", "total_quantity", "subtotal",
                "discount", "tax", "grand_total"):
        assert key in bill, f"Missing key: {key}"
    assert "warnings"        in d
    assert "unmatched_items" in d


# ---------------------------------------------------------------------------
# C. Aggregation arithmetic
# ---------------------------------------------------------------------------

def test_single_item_line_subtotal():
    """Line subtotal = unit_price × quantity."""
    match = _match([_order_item("Coffee", 3, 20.0)])
    result = _make_generator().generate(match)
    assert result.items[0].subtotal   == 60.0
    assert result.subtotal            == 60.0
    assert result.grand_total         == 60.0
    assert result.item_count          == 1
    assert result.total_quantity      == 3


def test_multi_item_subtotal_accuracy():
    match = _match([
        _order_item("Idli",  5, 10.0),
        _order_item("Vada",  2, 15.0),
        _order_item("Tea",   1, 10.0),
    ])
    result = _make_generator().generate(match)
    # 50 + 30 + 10 = 90
    assert result.subtotal       == 90.0
    assert result.item_count     == 3
    assert result.total_quantity == 8      # 5+2+1


def test_currency_rounded_to_2dp():
    """Rounding must not let floating-point drift through."""
    match = _match([_order_item("Item", 3, 10.1)])  # 3 × 10.1 = 30.299...
    result = _make_generator().generate(match)
    assert result.subtotal    == round(30.3, 2)
    assert result.grand_total == round(30.3, 2)


# ---------------------------------------------------------------------------
# D. Bill number format
# ---------------------------------------------------------------------------

def test_bill_number_prefix(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.bill_number.startswith("BILL-")


def test_bill_number_suffix_is_8_hex_chars(spec_match):
    result = _make_generator().generate(spec_match)
    suffix = result.bill_number[len("BILL-"):]
    assert len(suffix) == 8
    assert all(c in "0123456789ABCDEF" for c in suffix)


def test_bill_numbers_are_unique():
    """Each generate() call must produce a different bill number."""
    gen   = _make_generator()
    match = _match([_order_item("Tea", 1, 10.0)])
    bill1 = gen.generate(match)
    bill2 = gen.generate(match)
    assert bill1.bill_number != bill2.bill_number


# ---------------------------------------------------------------------------
# E. Timestamp
# ---------------------------------------------------------------------------

def test_date_time_is_utc(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.date_time.tzinfo == timezone.utc


def test_date_time_is_iso_string_in_dict(spec_match):
    d = _make_generator().generate(spec_match).to_dict()
    dt_str = d["bill"]["date_time"]
    assert isinstance(dt_str, str)
    assert "T" in dt_str


def test_date_time_can_be_mocked():
    """BillGeneratorService._now() is patchable for deterministic tests."""
    from datetime import datetime

    fixed = datetime(2025, 7, 23, 6, 0, 0, tzinfo=timezone.utc)
    match = _match([_order_item("Tea", 1, 10.0)])
    with patch.object(BillGeneratorService, "_now", return_value=fixed):
        result = _make_generator().generate(match)
    assert result.date_time == fixed


# ---------------------------------------------------------------------------
# F. Discount
# ---------------------------------------------------------------------------

def test_discount_default_zero(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.discount == 0.0


def test_custom_discount_reduces_grand_total():
    match  = _match([_order_item("Dosa", 2, 40.0)])   # subtotal = 80
    result = _make_generator(discount=10.0).generate(match)
    assert result.discount    == 10.0
    assert result.grand_total == 70.0   # 80 − 10


def test_discount_clamped_to_subtotal():
    """Discount cannot exceed subtotal — grand_total must be ≥ 0."""
    match  = _match([_order_item("Tea", 1, 10.0)])   # subtotal = 10
    result = _make_generator(discount=999.0).generate(match)
    assert result.discount    == 10.0   # clamped to subtotal
    assert result.grand_total == 0.0


# ---------------------------------------------------------------------------
# G. Tax
# ---------------------------------------------------------------------------

def test_tax_default_zero(spec_match):
    result = _make_generator().generate(spec_match)
    assert result.tax == 0.0


def test_custom_tax_increases_grand_total():
    match  = _match([_order_item("Coffee", 1, 20.0)])  # subtotal = 20
    result = _make_generator(tax=5.0).generate(match)
    assert result.tax         == 5.0
    assert result.grand_total == 25.0


def test_discount_and_tax_together():
    match  = _match([_order_item("Dosa", 2, 40.0)])  # subtotal = 80
    result = _make_generator(discount=10.0, tax=5.0).generate(match)
    # grand_total = 80 − 10 + 5 = 75
    assert result.discount    == 10.0
    assert result.tax         == 5.0
    assert result.grand_total == 75.0


# ---------------------------------------------------------------------------
# H. Unmatched items → warnings
# ---------------------------------------------------------------------------

def test_unmatched_items_in_result():
    match  = _match([_order_item("Tea", 1, 10.0)], unmatched=["pizza"])
    result = _make_generator().generate(match)
    assert "pizza" in result.unmatched_items


def test_one_warning_per_unmatched_item():
    match  = _match([], unmatched=["pizza", "burger"])
    result = _make_generator().generate(match)
    assert len(result.warnings) == 2


def test_warning_mentions_item_name():
    match  = _match([], unmatched=["pizza"])
    result = _make_generator().generate(match)
    assert "pizza" in result.warnings[0]


def test_warnings_in_to_dict(spec_match):
    """to_dict() must expose both warnings and unmatched_items at top level."""
    match = _match(
        [_order_item("Tea", 1, 10.0)],
        unmatched=["burger"],
    )
    d = _make_generator().generate(match).to_dict()
    assert "burger" in d["unmatched_items"]
    assert any("burger" in w for w in d["warnings"])


# ---------------------------------------------------------------------------
# I. Edge cases
# ---------------------------------------------------------------------------

def test_empty_match_result_returns_zero_bill():
    match  = _match([])
    result = _make_generator().generate(match)
    assert result.items          == []
    assert result.item_count     == 0
    assert result.total_quantity == 0
    assert result.subtotal       == 0.0
    assert result.grand_total    == 0.0


def test_all_items_unmatched():
    match  = _match([], unmatched=["pizza", "pasta"])
    result = _make_generator().generate(match)
    assert result.items          == []
    assert result.subtotal       == 0.0
    assert len(result.warnings)  == 2


def test_single_item_bill():
    match  = _match([_order_item("Coffee", 1, 20.0)])
    result = _make_generator().generate(match)
    assert result.item_count     == 1
    assert result.total_quantity == 1
    assert result.grand_total    == 20.0


def test_none_match_result_raises_value_error():
    gen = _make_generator()
    with pytest.raises(ValueError):
        gen.generate(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# J. BillResult.to_dict() wire format
# ---------------------------------------------------------------------------

def test_to_dict_bill_items_match_wire_format():
    match = _match([_order_item("Dosa", 2, 40.0)])
    d     = _make_generator().generate(match).to_dict()
    item  = d["bill"]["items"][0]
    assert item["name"]       == "Dosa"
    assert item["quantity"]   == 2
    assert item["unit_price"] == 40.0
    assert item["subtotal"]   == 80.0


def test_to_dict_top_level_keys():
    d = _make_generator().generate(_match([])).to_dict()
    assert set(d.keys()) == {"bill", "warnings", "unmatched_items"}


# ---------------------------------------------------------------------------
# K. BillLineItem.to_dict()
# ---------------------------------------------------------------------------

def test_bill_line_item_to_dict():
    li = BillLineItem(name="Tea", quantity=1, unit_price=15.0, subtotal=15.0)
    d  = li.to_dict()
    assert d == {"name": "Tea", "quantity": 1, "unit_price": 15.0, "subtotal": 15.0}
