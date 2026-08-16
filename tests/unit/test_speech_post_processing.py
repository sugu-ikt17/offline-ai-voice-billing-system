"""Unit tests for upgraded speech post-processing, number normalization,
menu vocabulary aliases, RapidFuzz fuzzy matching, and multiword reordering.

Tests mandatory spec cases:
- "2 tea"
- "two tea"
- "rendu tea"
- "ரெண்டு டீ"
- "4 puri"
- "four puri"
- "naalu puri"
- "பூரி நாலு"
- "tea 2"
- "coffee one"
- "dosa moonu"
- "curry" -> "puri"
"""

import pytest
from app.application.services.speech_normalizer import normalize
from app.application.services.menu_vocabulary_corrector import MenuVocabularyCorrector
from app.application.services.order_parser_service import OrderParserService

STANDARD_MENU = [
    "Coffee",
    "Tea",
    "Dosa",
    "Idli",
    "Vada",
    "Puri",
    "Sambar",
    "Pongal",
    "Upma",
    "Parotta",
]

corrector = MenuVocabularyCorrector(threshold=0.72)
parser = OrderParserService()


def process_transcript(text: str, vocabulary: list[str] = STANDARD_MENU) -> str:
    """Run full speech post-processing: normalize -> correct."""
    norm = normalize(text)
    return corrector.correct(norm, vocabulary)


class TestMandatorySpecCases:
    """Tests all test cases specified in the user request prompt."""

    def test_2_tea(self):
        assert process_transcript("2 tea") == "2 tea"

    def test_two_tea(self):
        assert process_transcript("two tea") == "2 tea"

    def test_rendu_tea(self):
        assert process_transcript("rendu tea") == "2 tea"

    def test_tamil_rendu_tea(self):
        assert process_transcript("ரெண்டு டீ") == "2 tea"

    def test_4_puri(self):
        assert process_transcript("4 puri") == "4 puri"

    def test_four_puri(self):
        assert process_transcript("four puri") == "4 puri"

    def test_naalu_puri(self):
        assert process_transcript("naalu puri") == "4 puri"

    def test_poori_naalu(self):
        assert process_transcript("பூரி நாலு") == "4 puri"

    def test_tea_2(self):
        assert process_transcript("tea 2") == "2 tea"

    def test_coffee_one(self):
        assert process_transcript("coffee one") == "1 coffee"

    def test_dosa_moonu(self):
        assert process_transcript("dosa moonu") == "3 dosa"

    def test_curry_alias_to_puri(self):
        """'curry' is an alias for 'puri'."""
        assert process_transcript("4 curry") == "4 puri"


class TestBillingParserIntegration:
    """Verify post-processed transcripts parse cleanly in OrderParserService."""

    def test_parse_rendu_tea(self):
        proc = process_transcript("ரெண்டு டீ")
        parsed = parser.parse(proc)
        assert parsed == [{"item": "tea", "quantity": 2}]

    def test_parse_poori_naalu(self):
        proc = process_transcript("பூரி நாலு")
        parsed = parser.parse(proc)
        assert parsed == [{"item": "puri", "quantity": 4}]

    def test_parse_dosa_moonu(self):
        proc = process_transcript("dosa moonu")
        parsed = parser.parse(proc)
        assert parsed == [{"item": "dosa", "quantity": 3}]

    def test_parse_coffee_one(self):
        proc = process_transcript("coffee one")
        parsed = parser.parse(proc)
        assert parsed == [{"item": "coffee", "quantity": 1}]
