"""Unit tests for TamilMenuNormalizer and Tamil script end-to-end voice order integration.

Tests all required test cases from prompt:
1. "2 டீ 2 தோசை 4 சமோசா" -> Tea x 2, Dosa x 2, Samosa x 4
2. "2 tea 2 dosa 4 samosa" -> Tea x 2, Dosa x 2, Samosa x 4
3. "2 டீ 2 dosa 4 சமோசா" -> Tea x 2, Dosa x 2, Samosa x 4
4. "ரெண்டு டீ ரெண்டு தோசை நாலு சமோசா" -> Tea x 2, Dosa x 2, Samosa x 4
5. "2 டீ நன்றி 2 தோசை 4 சமோசா" -> ONLY Tea x 2, Dosa x 2, Samosa x 4
6. "நன்றி 2 டீ" -> Tea x 2
7. "2 நன்றி டீ" -> "நன்றி" is NOT a menu item
8. "2 மசாலா தோசை" -> Masala Dosa x 2
9. "மசாலா தோசை 2" -> Masala Dosa x 2
10. Random Tamil speech without valid quantity-menu relationship -> No menu items matched
"""

import pytest
from unittest.mock import MagicMock

from app.application.services.tamil_menu_normalizer import TamilMenuNormalizer, transliterate_tamil_text
from app.application.services.speech_normalizer import normalize as speech_normalize
from app.application.services.order_parser_service import OrderParserService
from app.application.services.menu_matcher_service import MenuMatcherService
from app.infrastructure.database.models.menu_item_model import MenuItemModel


@pytest.fixture
def active_vocabulary():
    return [
        "Tea",
        "Coffee",
        "Dosa",
        "Idly",
        "Poori",
        "Vada",
        "Samosa",
        "Pongal",
        "Chapati",
        "Parotta",
        "Masala Dosa",
        "Plain Dosa",
        "Onion Dosa",
    ]


@pytest.fixture
def mock_menu_repository(active_vocabulary):
    repo = MagicMock()
    mock_items = [
        MenuItemModel(id=1, name="Tea", price=15.0),
        MenuItemModel(id=2, name="Coffee", price=20.0),
        MenuItemModel(id=3, name="Dosa", price=40.0),
        MenuItemModel(id=4, name="Idly", price=30.0),
        MenuItemModel(id=5, name="Poori", price=35.0),
        MenuItemModel(id=6, name="Vada", price=15.0),
        MenuItemModel(id=7, name="Samosa", price=15.0),
        MenuItemModel(id=8, name="Pongal", price=45.0),
        MenuItemModel(id=9, name="Chapati", price=30.0),
        MenuItemModel(id=10, name="Parotta", price=35.0),
        MenuItemModel(id=11, name="Masala Dosa", price=60.0),
        MenuItemModel(id=12, name="Plain Dosa", price=45.0),
        MenuItemModel(id=13, name="Onion Dosa", price=55.0),
    ]
    repo.get_all.return_value = mock_items
    return repo


def run_pipeline(transcript: str, active_vocab: list[str], menu_repo: MagicMock):
    """Run full post-processing pipeline: TamilNormalizer -> SpeechNormalizer -> OrderParser -> MenuMatcher."""
    normalizer = TamilMenuNormalizer()
    tamil_normalized = normalizer.normalize(transcript, active_vocab)
    speech_normalized = speech_normalize(tamil_normalized)
    parser = OrderParserService()
    parsed_items = parser.parse(speech_normalized)
    matcher = MenuMatcherService(menu_repo)
    match_result = matcher.match(parsed_items)
    return match_result.matched_items, match_result.unmatched_items, tamil_normalized, speech_normalized


# ---------------------------------------------------------------------------
# Direct Tamil Transliteration Unit Tests
# ---------------------------------------------------------------------------

def test_transliterate_tamil_text():
    assert transliterate_tamil_text("தோசை") == "thoasai"
    assert transliterate_tamil_text("இட்லி") == "itli"
    assert transliterate_tamil_text("சமோசா") == "samoasaa"
    assert transliterate_tamil_text("2 tea") == "2 tea"


# ---------------------------------------------------------------------------
# Test Case 1: Pure Tamil Input
# ---------------------------------------------------------------------------

def test_case_1_pure_tamil(active_vocabulary, mock_menu_repository):
    input_text = "2 டீ 2 தோசை 4 சமோசா"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 3
    assert matched[0].name == "Tea" and matched[0].quantity == 2
    assert matched[1].name == "Dosa" and matched[1].quantity == 2
    assert matched[2].name == "Samosa" and matched[2].quantity == 4


# ---------------------------------------------------------------------------
# Test Case 2: Pure English Input
# ---------------------------------------------------------------------------

def test_case_2_pure_english(active_vocabulary, mock_menu_repository):
    input_text = "2 tea 2 dosa 4 samosa"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 3
    assert matched[0].name == "Tea" and matched[0].quantity == 2
    assert matched[1].name == "Dosa" and matched[1].quantity == 2
    assert matched[2].name == "Samosa" and matched[2].quantity == 4


# ---------------------------------------------------------------------------
# Test Case 3: Mixed Tamil + English Input
# ---------------------------------------------------------------------------

def test_case_3_mixed_tamil_english(active_vocabulary, mock_menu_repository):
    input_text = "2 டீ 2 dosa 4 சமோசா"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 3
    assert matched[0].name == "Tea" and matched[0].quantity == 2
    assert matched[1].name == "Dosa" and matched[1].quantity == 2
    assert matched[2].name == "Samosa" and matched[2].quantity == 4


# ---------------------------------------------------------------------------
# Test Case 4: Spoken Tamil Number Words
# ---------------------------------------------------------------------------

def test_case_4_spoken_tamil_numbers(active_vocabulary, mock_menu_repository):
    input_text = "ரெண்டு டீ ரெண்டு தோசை நாலு சமோசா"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 3
    assert matched[0].name == "Tea" and matched[0].quantity == 2
    assert matched[1].name == "Dosa" and matched[1].quantity == 2
    assert matched[2].name == "Samosa" and matched[2].quantity == 4


# ---------------------------------------------------------------------------
# Test Case 5: Conversational Noise in Middle ("நன்றி")
# ---------------------------------------------------------------------------

def test_case_5_conversational_noise_middle(active_vocabulary, mock_menu_repository):
    input_text = "2 டீ நன்றி 2 தோசை 4 சமோசா"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 3
    matched_names = {item.name: item.quantity for item in matched}
    assert matched_names == {"Tea": 2, "Dosa": 2, "Samosa": 4}
    assert not any(item.name == "நன்றி" for item in matched)


# ---------------------------------------------------------------------------
# Test Case 6: Conversational Noise Before Item ("நன்றி 2 டீ")
# ---------------------------------------------------------------------------

def test_case_6_conversational_noise_before(active_vocabulary, mock_menu_repository):
    input_text = "நன்றி 2 டீ"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 1
    assert matched[0].name == "Tea" and matched[0].quantity == 2


# ---------------------------------------------------------------------------
# Test Case 7: Conversational Noise Between Quantity and Item ("2 நன்றி டீ")
# ---------------------------------------------------------------------------

def test_case_7_conversational_noise_between(active_vocabulary, mock_menu_repository):
    input_text = "2 நன்றி டீ"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    # "நன்றி" is not a menu item and breaks quantity-bound adjacency for "டீ"
    assert not any(item.name == "நன்றி" for item in matched)


# ---------------------------------------------------------------------------
# Test Case 8 & 9: Multi-word Tamil Menu Item ("2 மசாலா தோசை", "மசாலா தோசை 2")
# ---------------------------------------------------------------------------

def test_case_8_multiword_tamil_item_prefix(active_vocabulary, mock_menu_repository):
    input_text = "2 மசாலா தோசை"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 1
    assert matched[0].name == "Masala Dosa" and matched[0].quantity == 2


def test_case_9_multiword_tamil_item_postfix(active_vocabulary, mock_menu_repository):
    input_text = "மசாலா தோசை 2"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 1
    assert matched[0].name == "Masala Dosa" and matched[0].quantity == 2


# ---------------------------------------------------------------------------
# Test Case 10: Random Tamil Speech Without Quantity/Menu
# ---------------------------------------------------------------------------

def test_case_10_random_tamil_speech(active_vocabulary, mock_menu_repository):
    input_text = "வணக்கம் எப்படி இருக்கீங்க"
    matched, unmatched, t_norm, s_norm = run_pipeline(input_text, active_vocabulary, mock_menu_repository)

    assert len(matched) == 0
