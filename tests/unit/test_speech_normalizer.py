"""Unit tests for the Speech Normalizer (speech_normalizer.py).

Covers:
  - English number homophones (won→1, too→2, to→2, for→4)
  - Tamil number words (oru→1, rendu→2, moonu→3, …, pathu→10)
  - Menu item pronunciation fixes (coffer→coffee, dosai→dosa, idly→idli, …)
  - Mixed / Tanglish phrases (Tamil quantity + English item, and vice versa)
  - Reordering quantity-last to quantity-first (tea 2 → 2 tea)
  - Edge cases: empty string, whitespace-only, no matches, word-boundary safety
  - Extensibility: direct access to the exported dicts for coverage checks
"""

import pytest
from app.application.services.speech_normalizer import (
    normalize,
    ENGLISH_NUMBER_HOMOPHONES,
    TAMIL_NUMBER_WORDS,
    MENU_PRONUNCIATIONS,
)


# ===========================================================================
# 1. English number homophones
# ===========================================================================

class TestEnglishNumberHomophones:

    def test_won_to_one(self):
        assert normalize("won dosa") == "1 dosa"

    def test_wun_to_one(self):
        assert normalize("wun tea") == "1 tea"

    def test_too_to_two(self):
        assert normalize("too coffee") == "2 coffee"

    def test_to_to_two(self):
        assert normalize("to idli") == "2 idli"

    def test_for_to_four(self):
        assert normalize("for vada") == "4 vada"

    def test_won_for_cofi(self):
        """Multiple English homophones in one sentence."""
        assert normalize("won cofi for dosa") == "1 coffee 4 dosa"

    def test_too_tea_for_idly(self):
        assert normalize("too tee for idly") == "2 tea 4 idli"


    def test_case_insensitive_won(self):
        assert normalize("WON dosa") == "1 dosa"

    def test_case_insensitive_for(self):
        assert normalize("FOR coffee") == "4 coffee"

    def test_to_does_not_corrupt_tornado(self):
        """'to' as a word boundary: should NOT replace 'to' inside 'tornado'."""
        assert normalize("tornado") == "tornado"

    def test_for_does_not_corrupt_fortune(self):
        """'for' should NOT replace the 'for' inside 'fortune'."""
        assert normalize("fortune cookie") == "fortune cookie"

    def test_too_does_not_corrupt_toothbrush(self):
        """'too' should NOT match inside 'toothbrush'."""
        assert normalize("toothbrush") == "toothbrush"


# ===========================================================================
# 2. Tamil number words
# ===========================================================================

class TestTamilNumberWords:

    def test_oru_to_one(self):
        assert normalize("oru dosa") == "1 dosa"

    def test_onnu_to_one(self):
        assert normalize("onnu coffee") == "1 coffee"

    def test_rendu_to_two(self):
        assert normalize("rendu idli") == "2 idli"

    def test_renduu_to_two(self):
        assert normalize("renduu vada") == "2 vada"

    def test_moonu_to_three(self):
        assert normalize("moonu tea") == "3 tea"

    def test_naalu_to_four(self):
        assert normalize("naalu dosa") == "4 dosa"

    def test_anju_to_five(self):
        assert normalize("anju coffee") == "5 coffee"

    def test_aaru_to_six(self):
        assert normalize("aaru idli") == "6 idli"

    def test_ezhu_to_seven(self):
        assert normalize("ezhu vada") == "7 vada"

    def test_ettu_to_eight(self):
        assert normalize("ettu tea") == "8 tea"

    def test_ombodhu_to_nine(self):
        assert normalize("ombodhu dosa") == "9 dosa"

    def test_pathu_to_ten(self):
        assert normalize("pathu coffee") == "10 coffee"

    def test_tamil_number_case_insensitive(self):
        assert normalize("RENDU dosa") == "2 dosa"

    def test_multiple_tamil_numbers(self):
        assert normalize("oru coffee rendu dosa") == "1 coffee 2 dosa"

    def test_tanglish_oru_coffee_rendu_dosai(self):
        """Tanglish: Tamil qty + English item name (with menu correction)."""
        assert normalize("oru coffer rendu dosai") == "1 coffee 2 dosa"


# ===========================================================================
# 3. Menu item pronunciation normalizations
# ===========================================================================

class TestMenuPronunciations:

    def test_coffer_to_coffee(self):
        assert normalize("2 coffer") == "2 coffee"

    def test_cofi_to_coffee(self):
        assert normalize("2 cofi") == "2 coffee"

    def test_coffey_to_coffee(self):
        assert normalize("1 coffey") == "1 coffee"

    def test_cofee_to_coffee(self):
        assert normalize("3 cofee") == "3 coffee"

    def test_caffe_to_coffee(self):
        assert normalize("1 caffe") == "1 coffee"

    def test_tee_to_tea(self):
        assert normalize("1 tee") == "1 tea"

    def test_te_to_tea(self):
        assert normalize("2 te") == "2 tea"

    def test_cha_to_tea(self):
        assert normalize("1 cha") == "1 tea"

    def test_doza_to_dosa(self):
        assert normalize("2 doza") == "2 dosa"

    def test_dosai_to_dosa(self):
        assert normalize("1 dosai") == "1 dosa"

    def test_dosay_to_dosa(self):
        assert normalize("3 dosay") == "3 dosa"

    def test_dhosa_to_dosa(self):
        assert normalize("2 dhosa") == "2 dosa"

    def test_thosai_to_dosa(self):
        assert normalize("1 thosai") == "1 dosa"

    def test_idly_to_idli(self):
        assert normalize("2 idly") == "2 idli"

    def test_itly_to_idli(self):
        assert normalize("3 itly") == "3 idli"

    def test_idlee_to_idli(self):
        assert normalize("1 idlee") == "1 idli"

    def test_iddly_to_idli(self):
        assert normalize("4 iddly") == "4 idli"


    def test_vadai_to_vada(self):
        assert normalize("1 vadai") == "1 vada"

    def test_vaday_to_vada(self):
        assert normalize("2 vaday") == "2 vada"

    def test_wada_to_vada(self):
        assert normalize("3 wada") == "3 vada"

    def test_vade_to_vada(self):
        assert normalize("1 vade") == "1 vada"

    def test_sambhar_to_sambar(self):
        assert normalize("1 sambhar") == "1 sambar"

    def test_saambar_to_sambar(self):
        assert normalize("2 saambar") == "2 sambar"

    def test_pongall_to_pongal(self):
        assert normalize("1 pongall") == "1 pongal"

    def test_pongul_to_pongal(self):
        assert normalize("2 pongul") == "2 pongal"

    def test_uppma_to_upma(self):
        assert normalize("1 uppma") == "1 upma"

    def test_uppuma_to_upma(self):
        assert normalize("3 uppuma") == "3 upma"

    def test_chay_to_chai(self):
        assert normalize("2 chay") == "2 tea"

    def test_parota_to_parotta(self):
        assert normalize("2 parota") == "2 parotta"

    def test_paratha_to_parotta(self):
        assert normalize("1 paratha") == "1 parotta"


# ===========================================================================
# 4. Mixed / Tanglish full-phrase tests
# ===========================================================================

class TestMixedPhrases:

    def test_full_tanglish_order(self):
        """Realistic Tanglish: Tamil numbers with Tamil-Anglicised menu words."""
        result = normalize("rendu dosai onnu coffer")
        assert result == "2 dosa 1 coffee"

    def test_english_homophones_with_menu_fix(self):
        """English homophone quantity + mis-pronounced item."""
        assert normalize("won cofi too dosai") == "1 coffee 2 dosa"

    def test_for_idly_and_too_tee(self):
        assert normalize("for idly and too tee") == "4 idli and 2 tea"

    def test_oru_coffer_and_oru_tee(self):
        assert normalize("oru coffer and oru tee") == "1 coffee and 1 tea"

    def test_naalu_idly_rendu_vadai(self):
        assert normalize("naalu idly rendu vadai") == "4 idli 2 vada"


    def test_pathu_doza(self):
        assert normalize("pathu doza") == "10 dosa"

    def test_ombodhu_cofi_anju_dosai(self):
        assert normalize("ombodhu cofi anju dosai") == "9 coffee 5 dosa"


# ===========================================================================
# 5. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_string_returns_empty(self):
        assert normalize("") == ""

    def test_whitespace_only_returns_unchanged(self):
        result = normalize("   ")
        assert result == "   "

    def test_no_matches_lowercases_output(self):
        """Text with no substitutions is still lowercased."""
        assert normalize("2 dosa 1 tea") == "2 dosa 1 tea"

    def test_already_correct_coffee(self):
        assert normalize("1 coffee") == "1 coffee"

    def test_already_correct_idli(self):
        assert normalize("2 idli") == "2 idli"

    def test_already_correct_vada(self):
        assert normalize("3 vada") == "3 vada"

    def test_already_correct_dosa(self):
        assert normalize("1 dosa") == "1 dosa"

    def test_already_correct_tea(self):
        assert normalize("2 tea") == "2 tea"

    def test_mixed_case_input_lowercased(self):
        """Arbitrary capitalisation is lowercased before normalization."""
        assert normalize("TWO Dosai") == "2 dosa"

    def test_numbers_not_affected(self):
        """Digit tokens should pass through unchanged."""
        assert normalize("2 dosa 1 tea") == "2 dosa 1 tea"

    def test_word_boundary_oru_inside_longer_word(self):
        """'oru' should NOT be replaced inside 'forum'."""
        assert normalize("forum discussion") == "forum discussion"


# ===========================================================================
# 6. Extensibility / dictionary integrity checks
# ===========================================================================

class TestDictionaryIntegrity:

    def test_english_homophones_dict_has_required_entries(self):
        required = {"won", "wun", "too", "to", "for"}
        assert required.issubset(ENGLISH_NUMBER_HOMOPHONES.keys())

    def test_tamil_numbers_dict_covers_one_to_ten(self):
        """All ten Tamil number words required by the spec must be present."""
        required = {
            "oru", "onnu",          # 1
            "rendu", "renduu",      # 2
            "moonu",                # 3
            "naalu",                # 4
            "anju",                 # 5
            "aaru",                 # 6
            "ezhu",                 # 7
            "ettu",                 # 8
            "ombodhu",              # 9
            "pathu",                # 10
        }
        assert required.issubset(TAMIL_NUMBER_WORDS.keys())

    def test_menu_dict_has_required_entries(self):
        """Spec-mandated menu corrections must be present in the dict."""
        required = {
            "coffer", "cofi",   # coffee
            "tee",              # tea
            "doza", "dosai",    # dosa
            "idly", "itly",     # idli
            "vadai",            # vada
        }
        assert required.issubset(MENU_PRONUNCIATIONS.keys())

    def test_tamil_values_are_valid_english_number_words(self):
        from app.application.services.order_parser_service import QUANTITY_WORDS
        for tamil_word, english_word in TAMIL_NUMBER_WORDS.items():
            assert english_word in QUANTITY_WORDS, (
                f"'{tamil_word}' maps to '{english_word}' which is not in QUANTITY_WORDS"
            )

    def test_english_homophone_values_are_valid_number_words(self):
        from app.application.services.order_parser_service import QUANTITY_WORDS
        for spoken, canonical in ENGLISH_NUMBER_HOMOPHONES.items():
            assert canonical in QUANTITY_WORDS, (
                f"Homophone '{spoken}' maps to '{canonical}' which is not in QUANTITY_WORDS"
            )
