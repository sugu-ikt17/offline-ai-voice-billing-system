"""Unit tests for MenuContextEngine and 5-strategy correction cascade.

Covers:
  - Exact match (coffee, tea, puri, dosa, idli, vada)
  - Quantity-bound Alias match (2 coffer -> 2 coffee, 3 tee -> 3 tea, 4 poorii -> 4 puri)
  - Quantity-bound RapidFuzz & Phonetic match (2 cofi -> 2 coffee, 4 curry -> 4 puri)
  - Word distance fallback
  - Dynamic SQLite / menu vocabulary context
  - In-memory caching and automatic refresh
  - Confidence scoring and low-confidence thresholding
  - Token protection (digits, quantities, connectors)
  - Required context-aware & quantity-bound regression tests
"""

import pytest

from app.application.services.menu_context_engine import (
    MenuContextEngine,
    CorrectionResult,
    TokenMatch,
)
from app.application.services.menu_vocabulary_corrector import MenuVocabularyCorrector


@pytest.fixture
def default_menu() -> list[str]:
    """Default active menu items from database."""
    return ["Tea", "Coffee", "Puri", "Dosa", "Idli", "Vada"]


@pytest.fixture
def engine() -> MenuContextEngine:
    """MenuContextEngine instance with 65% confidence threshold."""
    return MenuContextEngine(confidence_threshold=0.65)


# ---------------------------------------------------------------------------
# 1. Quantity-bound word test cases
# ---------------------------------------------------------------------------


def test_correction_coffee(engine, default_menu):
    """Test exact match '2 coffee'."""
    res = engine.correct("2 coffee", default_menu)
    assert res == "2 coffee"


def test_correction_coffer(engine, default_menu):
    """Test alias/phonetic match '2 coffer' -> '2 coffee'."""
    res = engine.correct("2 coffer", default_menu)
    assert res == "2 coffee"


def test_correction_copy(engine, default_menu):
    """Test phonetic match '2 copy' -> '2 coffee'."""
    res = engine.correct("2 copy", default_menu)
    assert res == "2 coffee"


def test_correction_tea(engine, default_menu):
    """Test exact match '2 tea'."""
    res = engine.correct("2 tea", default_menu)
    assert res == "2 tea"


def test_correction_tee(engine, default_menu):
    """Test alias match '2 tee' -> '2 tea'."""
    res = engine.correct("2 tee", default_menu)
    assert res == "2 tea"


def test_correction_puri(engine, default_menu):
    """Test exact match '4 puri'."""
    res = engine.correct("4 puri", default_menu)
    assert res == "4 puri"


def test_correction_curry_with_puri_in_menu(engine, default_menu):
    """Test phonetic/contextual correction '4 curry' -> '4 puri' when Puri is in menu."""
    res = engine.correct("4 curry", default_menu)
    assert res == "4 puri"


def test_correction_curry_without_puri_in_menu(engine):
    """Test 'curry' when Puri is NOT in menu -> low confidence / keeps original."""
    menu_without_puri = ["Tea", "Coffee", "Dosa"]
    res = engine.correct("4 curry", menu_without_puri)
    assert res == "4 curry"


def test_correction_poorii(engine, default_menu):
    """Test alias match '4 poorii' -> '4 puri'."""
    res = engine.correct("4 poorii", default_menu)
    assert res == "4 puri"


def test_correction_dosay(engine, default_menu):
    """Test alias match '2 dosay' -> '2 dosa'."""
    res = engine.correct("2 dosay", default_menu)
    assert res == "2 dosa"


def test_correction_itly(engine, default_menu):
    """Test alias match '2 itly' -> '2 idli'."""
    res = engine.correct("2 itly", default_menu)
    assert res == "2 idli"


def test_correction_vadai(engine, default_menu):
    """Test alias match '1 vadai' -> '1 vada'."""
    res = engine.correct("1 vadai", default_menu)
    assert res == "1 vada"


# ---------------------------------------------------------------------------
# 2. Required Regression Tests
# ---------------------------------------------------------------------------


def test_regression_2_tea_thank_you(engine, default_menu):
    """2 tea thank you -> tea remains tea, thank you is not converted."""
    res = engine.correct("2 tea thank you", default_menu)
    assert res == "2 tea thank you"


def test_regression_thank_you_2_tea(engine, default_menu):
    """thank you 2 tea -> tea remains tea, thank you untouched."""
    res = engine.correct("thank you 2 tea", default_menu)
    assert res == "thank you 2 tea"


def test_regression_2_thank_you_dosa(engine, default_menu):
    """2 thank you dosa -> do NOT convert thank/you into menu item or merge with 2."""
    res = engine.correct("2 thank you dosa", default_menu)
    assert res == "2 thank you dosa"


def test_regression_dosa_thank_you_2(engine, default_menu):
    """dosa thank you 2 -> do NOT convert thank/you into menu item or bridge dosa to 2."""
    res = engine.correct("dosa thank you 2", default_menu)
    assert res == "dosa thank you 2"


def test_regression_2_cofi(engine, default_menu):
    """2 cofi -> cofi can correct to coffee if coffee is active."""
    res = engine.correct("2 cofi", default_menu)
    assert res == "2 coffee"


def test_regression_cofi_thank_you(engine, default_menu):
    """cofi thank you -> do NOT correct cofi to coffee because no adjacent quantity."""
    res = engine.correct("cofi thank you", default_menu)
    assert res == "cofi thank you"


def test_regression_thank_cofi_2(engine, default_menu):
    """thank cofi 2 -> only the directly quantity-adjacent candidate (cofi) is eligible."""
    res = engine.correct("thank cofi 2", default_menu)
    assert res == "thank coffee 2"


def test_regression_2_coffee_4_dosa(engine, default_menu):
    """2 coffee 4 dosa -> both valid."""
    res = engine.correct("2 coffee 4 dosa", default_menu)
    assert res == "2 coffee 4 dosa"


def test_regression_2_coffee_thank_you_4_dosa(engine, default_menu):
    """2 coffee thank you 4 dosa -> both valid."""
    res = engine.correct("2 coffee thank you 4 dosa", default_menu)
    assert res == "2 coffee thank you 4 dosa"


def test_regression_unrelated_words_not_converted(engine):
    """2 tea 2 idly nada-samoza i've got 2 -> unrelated words must not be converted to menu items."""
    menu = ["Tea", "Idli", "Samosa", "Dosa"]
    res = engine.correct("2 tea 2 idly nada-samoza i've got 2", menu)
    assert res == "2 tea 2 idli nada-samoza i've got 2"


def test_regression_multi_word_menu_item(engine):
    """Multi-word menu item support ('masala dosa')."""
    menu = ["Tea", "Coffee", "Masala Dosa"]
    assert engine.correct("2 masala dosa", menu) == "2 masala dosa"
    assert engine.correct("masala dosa 2", menu) == "masala dosa 2"
    assert engine.correct("2 masla dosa", menu) == "2 masala dosa"
    assert engine.correct("2 thank you masala dosa", menu) == "2 thank you masala dosa"


# ---------------------------------------------------------------------------
# 3. Strategy evaluation & confidence reporting
# ---------------------------------------------------------------------------


def test_multi_item_phrase_correction(engine, default_menu):
    """Test full phrase: '2 coffer 3 tee 4 poorii 1 vadai'."""
    raw = "2 coffer 3 tee 4 poorii 1 vadai"
    expected = "2 coffee 3 tea 4 puri 1 vada"
    res = engine.correct(raw, default_menu)
    assert res == expected


def test_quantities_and_digits_never_modified(engine, default_menu):
    """Digits and quantity words must never be modified by the engine."""
    raw = "1 2 3 4 5 one two three four five"
    res = engine.correct(raw, default_menu)
    assert res == raw


def test_correct_with_details_returns_dataclass(engine, default_menu):
    """Verify detailed CorrectionResult structure with token strategies."""
    res = engine.correct_with_details("2 coffer", default_menu)

    assert isinstance(res, CorrectionResult)
    assert res.original_transcript == "2 coffer"
    assert res.corrected_transcript == "2 coffee"
    assert len(res.token_matches) == 2

    # Token 1: '2' (protected)
    m1 = res.token_matches[0]
    assert m1.original_token == "2"
    assert m1.strategy == "protected"
    assert m1.confidence == 1.0

    # Token 2: 'coffer' (alias_match)
    m2 = res.token_matches[1]
    assert m2.original_token == "coffer"
    assert m2.corrected_token == "coffee"
    assert m2.strategy == "alias_match"
    assert m2.confidence >= 0.90


def test_low_confidence_threshold_handling():
    """Tokens matching below confidence threshold yield suggestions."""
    strict_engine = MenuContextEngine(confidence_threshold=0.99)
    menu = ["Tea", "Coffee"]

    # 'cofi' matches 'coffee' at ~96% confidence (< 99% threshold)
    res = strict_engine.correct_with_details("1 cofi", menu)
    token_match = res.token_matches[1]

    assert token_match.corrected_token == "cofi"
    assert len(token_match.suggestions) > 0


# ---------------------------------------------------------------------------
# 4. In-Memory Caching & Refresh Tests
# ---------------------------------------------------------------------------


def test_in_memory_caching_and_refresh(engine):
    """Verify in-memory caching and automatic refresh when menu changes."""
    menu_v1 = ["Tea", "Coffee"]
    res1 = engine.correct("2 coffer", menu_v1)
    assert res1 == "2 coffee"
    assert engine._is_cache_valid() is True

    # Update menu to v2 with 'Puri'
    menu_v2 = ["Puri", "Dosa"]
    res2 = engine.correct("4 poorii", menu_v2)
    assert res2 == "4 puri"
    assert engine._cached_vocabulary == menu_v2


def test_cache_invalidation(engine, default_menu):
    """Verify cache invalidation."""
    engine.correct("2 tea", default_menu)
    assert engine._is_cache_valid() is True

    engine.invalidate_cache()
    assert engine._is_cache_valid() is False


# ---------------------------------------------------------------------------
# 5. MenuVocabularyCorrector wrapper test
# ---------------------------------------------------------------------------


def test_menu_vocabulary_corrector_wrapper(default_menu):
    """Verify backward compatibility of MenuVocabularyCorrector class."""
    corrector = MenuVocabularyCorrector(threshold=0.65)
    assert corrector.correct("2 coffer", default_menu) == "2 coffee"
    assert corrector.correct("4 vadai", default_menu) == "4 vada"
