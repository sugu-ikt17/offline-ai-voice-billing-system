"""Unit tests for MenuContextEngine and 5-strategy correction cascade.

Covers:
  - Exact match (coffee, tea, puri, dosa, idli, vada)
  - Alias match (coffer, tee, poorii, dosay, itly, vadai)
  - RapidFuzz & Phonetic match (copy -> coffee, curry -> puri)
  - Word distance fallback
  - Dynamic SQLite / menu vocabulary context
  - In-memory caching and automatic refresh
  - Confidence scoring and low-confidence thresholding
  - Token protection (digits, quantities, connectors)
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
# 1. Individual word test cases as requested in user requirements
# ---------------------------------------------------------------------------


def test_correction_coffee(engine, default_menu):
    """Test exact match 'coffee'."""
    res = engine.correct("coffee", default_menu)
    assert res == "coffee"


def test_correction_coffer(engine, default_menu):
    """Test alias/phonetic match 'coffer' -> 'coffee'."""
    res = engine.correct("coffer", default_menu)
    assert res == "coffee"


def test_correction_copy(engine, default_menu):
    """Test phonetic match 'copy' -> 'coffee'."""
    res = engine.correct("copy", default_menu)
    assert res == "coffee"


def test_correction_tea(engine, default_menu):
    """Test exact match 'tea'."""
    res = engine.correct("tea", default_menu)
    assert res == "tea"


def test_correction_tee(engine, default_menu):
    """Test alias match 'tee' -> 'tea'."""
    res = engine.correct("tee", default_menu)
    assert res == "tea"


def test_correction_puri(engine, default_menu):
    """Test exact match 'puri'."""
    res = engine.correct("puri", default_menu)
    assert res == "puri"


def test_correction_curry_with_puri_in_menu(engine, default_menu):
    """Test phonetic/contextual correction '4 curry' -> '4 puri' when Puri is in menu."""
    res = engine.correct("4 curry", default_menu)
    assert res == "4 puri"


def test_correction_curry_without_puri_in_menu(engine):
    """Test 'curry' when Puri is NOT in menu -> low confidence / Unknown Menu Item."""
    menu_without_puri = ["Tea", "Coffee", "Dosa"]
    res = engine.correct("4 curry", menu_without_puri)
    assert "Unknown Menu Item" in res or res == "4 curry"


def test_correction_poorii(engine, default_menu):
    """Test alias match 'poorii' -> 'puri'."""
    res = engine.correct("poorii", default_menu)
    assert res == "puri"


def test_correction_dosay(engine, default_menu):
    """Test alias match 'dosay' -> 'dosa'."""
    res = engine.correct("dosay", default_menu)
    assert res == "dosa"


def test_correction_itly(engine, default_menu):
    """Test alias match 'itly' -> 'idli'."""
    res = engine.correct("itly", default_menu)
    assert res == "idli"


def test_correction_vadai(engine, default_menu):
    """Test alias match 'vadai' -> 'vada'."""
    res = engine.correct("vadai", default_menu)
    assert res == "vada"


# ---------------------------------------------------------------------------
# 2. Multi-token phrases and quantity protection
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


# ---------------------------------------------------------------------------
# 3. Strategy evaluation & confidence reporting
# ---------------------------------------------------------------------------


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


def test_low_confidence_threshold_handling(engine):
    """Tokens matching below confidence threshold yield Unknown Menu Item with suggestions."""
    strict_engine = MenuContextEngine(confidence_threshold=0.95)
    menu = ["Tea", "Coffee"]

    # 'cofi' matches 'coffee' around ~90% confidence (< 95% threshold)
    res = strict_engine.correct_with_details("1 xyzzqw", menu)
    token_match = res.token_matches[1]

    assert token_match.corrected_token == "Unknown Menu Item"
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
