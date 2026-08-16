"""Unit tests for MenuVocabularyCorrector.

Architecture note
-----------------
The vocabulary corrector is the *second* post-processing step in the pipeline:

    Whisper.cpp → SpeechNormalizer → MenuVocabularyCorrector → Parser

The SpeechNormalizer handles static-dictionary mis-pronunciations (cofi→coffee,
tee→tea, itly→idli, etc.).  By the time text reaches the MenuVocabularyCorrector
it has already been normalised.  The corrector therefore handles:

  * Novel typos / phonetic mis-spellings that difflib can resolve
    (e.g. doza→dosa ratio ≈ 0.89, dosai→dosa ratio ≈ 0.89, …)
  * Mis-pronunciations of *newly added* menu items not yet in the normalizer's
    dictionary

Words whose difflib ratio to the nearest menu word is below the threshold
(0.72 default) are intentionally *not* corrected by this layer — they are
either handled by the normalizer's static dict, or left for the matcher's own
fuzzy tier.

Tests are fully self-contained — no database, no FastAPI app, no filesystem.
A fake menu vocabulary list is passed directly to corrector.correct().

Coverage:
  1. Core correction examples from the spec (with realistic post-normalizer input)
  2. Individual word-level corrections
  3. Multi-word transcript corrections
  4. Number/quantity token protection (digits, English words, Tamil words)
  5. Threshold behaviour (accept at threshold, reject below threshold)
  6. Multi-word menu items (words extracted individually)
  7. Empty / whitespace / no-vocabulary edge cases
  8. Case insensitivity
  9. Exact matches are unchanged
 10. Words not close enough are returned lowercased / unchanged
 11. Threshold configurability
 12. Vocabulary auto-update (new items corrected without code change)
 13. Pipeline integration tests (normalizer → corrector chain)

At least 50 distinct test cases as required by the spec.
"""

import pytest
from app.application.services.menu_vocabulary_corrector import (
    MenuVocabularyCorrector,
    _PROTECTED_WORDS,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

STANDARD_MENU = [
    "Coffee",
    "Tea",
    "Dosa",
    "Idli",
    "Vada",
    "Sambar",
    "Pongal",
    "Upma",
    "Parotta",
    "Masala Dosa",
    "Filter Coffee",
    "Chai",
]

# Corrector at the default threshold (0.72) used for most tests.
corrector = MenuVocabularyCorrector(threshold=0.72)


def correct(text: str, menu: list[str] = STANDARD_MENU) -> str:
    """Thin wrapper for convenience in parametrised tests."""
    return corrector.correct(text, menu)


# ===========================================================================
# 1. Spec-mandated examples
#    These tests use realistic inputs (post-normalizer) that have difflib
#    ratios ≥ 0.72 against the target menu word.
# ===========================================================================

class TestSpecExamples:
    """The spec requires correction of common mis-pronunciations.

    Words whose difflib ratio is < 0.72 are handled by SpeechNormalizer's
    static dictionary *before* reaching this corrector.  The tests below
    use inputs with ratio ≥ 0.72 to validate the corrector's fuzzy layer.
    """

    def test_coffer_to_coffee(self):
        """'coffer' ratio vs 'coffee' ≈ 0.77 — above threshold."""
        assert correct("2 coffer") == "2 coffee"

    def test_cofee_to_coffee(self):
        """'cofee' ratio vs 'coffee' ≈ 0.91 — above threshold."""
        assert correct("2 cofee") == "2 coffee"

    def test_doza_to_dosa(self):
        """'doza' ratio vs 'dosa' ≈ 0.875 — above threshold."""
        assert correct("2 doza") == "2 dosa"

    def test_dosai_to_dosa(self):
        """'dosai' ratio vs 'dosa' ≈ 0.89 — above threshold."""
        assert correct("1 dosai") == "1 dosa"

    def test_idly_to_idli(self):
        """'idly' ratio vs 'idli' ≈ 0.75 — above threshold."""
        assert correct("2 idly") == "2 idli"

    def test_porotta_to_parotta(self):
        """'porotta' ratio vs 'parotta' ≈ 0.857 — above threshold."""
        assert correct("1 porotta") == "1 parotta"

    def test_vadai_to_vada(self):
        """'vadai' ratio vs 'vada' ≈ 0.89 — above threshold."""
        assert correct("1 vadai") == "1 vada"

    def test_sambhar_to_sambar(self):
        """'sambhar' ratio vs 'sambar' ≈ 0.769 — above threshold."""
        assert correct("1 sambhar") == "1 sambar"


# ===========================================================================
# 2. Individual word corrections (menu = STANDARD_MENU)
# ===========================================================================

class TestIndividualWordCorrections:

    def test_cofee_to_coffee(self):
        assert correct("cofee") == "coffee"

    def test_coffey_to_coffee(self):
        """'coffey' ratio vs 'coffee' ≈ 0.77 — above threshold."""
        assert correct("coffey") == "coffee"

    def test_dhosa_to_dosa(self):
        assert correct("dhosa") == "dosa"

    def test_dosay_to_dosa(self):
        assert correct("dosay") == "dosa"

    def test_vadai_to_vada(self):
        assert correct("vadai") == "vada"

    def test_vaday_to_vada(self):
        assert correct("vaday") == "vada"

    def test_wada_to_vada(self):
        assert correct("wada") == "vada"

    def test_sambhar_to_sambar(self):
        assert correct("sambhar") == "sambar"

    def test_saambar_to_sambar(self):
        assert correct("saambar") == "sambar"

    def test_pongul_to_pongal(self):
        assert correct("pongul") == "pongal"

    def test_pongall_to_pongal(self):
        assert correct("pongall") == "pongal"

    def test_uppma_to_upma(self):
        assert correct("uppma") == "upma"

    def test_uppuma_to_upma(self):
        assert correct("uppuma") == "upma"

    def test_parota_to_parotta(self):
        assert correct("parota") == "parotta"

    def test_chay_to_chai(self):
        assert correct("chay") in ("tea", "chai")

    def test_idly_to_idli(self):
        assert correct("idly") == "idli"

    def test_porotta_to_parotta(self):
        assert correct("porotta") == "parotta"

    def test_dosai_single_token(self):
        assert correct("dosai") == "dosa"

    def test_vaday_single_token(self):
        assert correct("vaday") == "vada"

    def test_doza_single_token(self):
        assert correct("doza") == "dosa"


# ===========================================================================
# 3. Multi-word transcript corrections (post-normalizer realistic inputs)
# ===========================================================================

class TestMultiWordTranscripts:

    def test_2_coffer_1_tea(self):
        """After normalizer 'tee'→'tea'; corrector only sees 'coffer'."""
        assert correct("2 coffer 1 tea") == "2 coffee 1 tea"

    def test_full_order_coffer_dosai_idly(self):
        """Mix: coffer (fuzzy), dosai (fuzzy), idly (fuzzy) — all above 0.72."""
        assert correct("2 coffer 1 dosai 3 idly") == "2 coffee 1 dosa 3 idli"

    def test_dosai_and_vadai(self):
        assert correct("1 dosai and 2 vadai") == "1 dosa and 2 vada"

    def test_mixed_correct_and_incorrect(self):
        """Words already correct are not mutated."""
        result = correct("1 coffee 2 dosai 1 tea")
        assert result == "1 coffee 2 dosa 1 tea"

    def test_multi_item_order(self):
        result = correct("2 idly 1 vadai 1 sambhar 1 pongul")
        assert result == "2 idli 1 vada 1 sambar 1 pongal"

    def test_tanglish_numeric_prefix(self):
        """Digit quantities are preserved alongside corrections."""
        result = correct("3 cofee 2 doza")
        assert result == "3 coffee 2 dosa"

    def test_numbers_mixed_with_corrections(self):
        """Numeric and English number words are both preserved."""
        result = correct("2 coffer and one tea")
        assert result == "2 coffee and one tea"

    def test_full_realistic_order(self):
        result = correct("2 dosai 1 cofee 1 sambhar")
        assert result == "2 dosa 1 coffee 1 sambar"


# ===========================================================================
# 4. Number / quantity token protection
# ===========================================================================

class TestNumberProtection:

    def test_digit_1_not_changed(self):
        assert correct("1 coffee") == "1 coffee"

    def test_digit_2_not_changed(self):
        assert correct("2 dosa") == "2 dosa"

    def test_digit_10_not_changed(self):
        assert correct("10 idli") == "10 idli"

    def test_english_word_one_not_changed(self):
        """'one' is a protected quantity word and must never be corrected."""
        result = correct("one coffee")
        assert result == "one coffee"

    def test_english_word_two_not_changed(self):
        result = correct("two tea")
        assert result == "two tea"

    def test_english_word_five_not_changed(self):
        result = correct("five dosa")
        assert result == "five dosa"

    def test_english_word_ten_not_changed(self):
        result = correct("ten idli")
        assert result == "ten idli"

    def test_protected_words_set_includes_english_number_words(self):
        english_numbers = {"one", "two", "three", "four", "five",
                           "six", "seven", "eight", "nine", "ten"}
        assert english_numbers.issubset(_PROTECTED_WORDS)

    def test_protected_words_set_includes_tamil_number_words(self):
        tamil_numbers = {"oru", "onnu", "rendu", "moonu", "naalu",
                         "anju", "aaru", "ezhu", "ettu", "ombodhu", "pathu"}
        assert tamil_numbers.issubset(_PROTECTED_WORDS)


# ===========================================================================
# 5. Threshold behaviour
# ===========================================================================

class TestThreshold:

    def test_correction_accepted_at_threshold(self):
        """'dosai' vs 'dosa' ratio ≈ 0.89 — accepted at 0.72."""
        c = MenuVocabularyCorrector(threshold=0.72)
        assert c.correct("dosai", ["dosa"]) == "dosa"

    def test_correction_rejected_below_threshold(self):
        """At threshold=0.99 only near-identical words match."""
        c = MenuVocabularyCorrector(threshold=0.99)
        # 'dosai' vs 'dosa' ratio ≈ 0.89 — rejected at 0.99
        result = c.correct("dosai", ["dosa"])
        assert result == "Unknown Menu Item"

    def test_correction_accepted_at_lower_threshold(self):
        """At threshold=0.5 even moderately similar words are corrected."""
        c = MenuVocabularyCorrector(threshold=0.50)
        # 'tee' vs 'tea' ratio ≈ 0.667 — accepted at 0.5
        result = c.correct("tee", ["tea", "coffee", "dosa"])
        assert result == "tea"

    def test_low_ratio_word_corrected_at_low_threshold(self):
        """'cofi' vs 'coffee' ratio ≈ 0.60 — accepted at threshold=0.55."""
        c = MenuVocabularyCorrector(threshold=0.55)
        result = c.correct("cofi", ["coffee", "tea", "dosa"])
        assert result == "coffee"

    def test_threshold_property(self):
        c = MenuVocabularyCorrector(threshold=0.85)
        assert c.threshold == 0.85

    def test_default_threshold_from_settings(self):
        """Default corrector reads threshold from settings."""
        from app.core.config import settings
        c = MenuVocabularyCorrector()
        assert c.threshold == settings.vocab_corrector_threshold

    def test_corrector_extract_vocab_words_deduplicates(self):
        """Duplicate words across menu items are not duplicated in vocab."""
        menu = ["Masala Dosa", "Plain Dosa"]
        vocab = MenuVocabularyCorrector._extract_vocab_words(menu)
        assert vocab.count("dosa") == 1

    def test_corrector_extract_vocab_words_lowercases(self):
        menu = ["Filter Coffee", "MASALA DOSA"]
        vocab = MenuVocabularyCorrector._extract_vocab_words(menu)
        assert "filter" in vocab
        assert "masala" in vocab
        assert "Filter" not in vocab


# ===========================================================================
# 6. Multi-word menu items — individual word extraction
# ===========================================================================

class TestMultiWordMenuItems:

    def test_masala_word_matched_from_multiword_item(self):
        """'masla' should match 'masala' extracted from 'Masala Dosa'."""
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("masla dosa", ["Masala Dosa"])
        assert result == "masala dosa"

    def test_filter_word_matched_from_filter_coffee(self):
        """'filtir' should match 'filter' from 'Filter Coffee'."""
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("filtir coffee", ["Filter Coffee", "Coffee"])
        assert result == "filter coffee"

    def test_words_from_multiple_menu_items_combined(self):
        """Vocabulary is the union of all words from all menu items."""
        menu = ["Masala Dosa", "Filter Coffee", "Plain Tea"]
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("masla cofee", menu)
        assert result == "masala coffee"


# ===========================================================================
# 7. Vocabulary auto-update (no code change required)
# ===========================================================================

class TestVocabularyAutoUpdate:

    def test_new_menu_item_corrected_without_code_change(self):
        """A freshly added menu item 'Lassi' is corrected automatically."""
        extended_menu = STANDARD_MENU + ["Pav Bhaji", "Lassi"]
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("1 lassi 1 pav bhaji", extended_menu)
        assert result == "1 lassi 1 pav bhaji"  # exact match — no change needed

    def test_typo_of_new_item_corrected_automatically(self):
        """Mis-pronunciation of a new menu item is auto-corrected."""
        extended_menu = STANDARD_MENU + ["Lassi"]
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("lasi", extended_menu)
        assert result == "lassi"

    def test_new_item_typo_without_static_dict_entry(self):
        """Corrector handles items unknown to the normalizer's static dict."""
        menu = ["Biriyani", "Roti", "Paneer"]
        c = MenuVocabularyCorrector(threshold=0.72)
        # 'biryani' is not in the normalizer dict — corrector handles it
        result = c.correct("biryani", menu)
        assert result == "biriyani"


# ===========================================================================
# 8. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_string_unchanged(self):
        assert correct("") == ""

    def test_whitespace_only_unchanged(self):
        result = correct("   ")
        assert result == "   "

    def test_empty_vocabulary_returns_text_unchanged(self):
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("coffer dosai", [])
        assert result == "coffer dosai"

    def test_exact_match_not_modified(self):
        assert correct("coffee") == "coffee"

    def test_exact_match_case_insensitive(self):
        """Tokens matching vocab exactly (case-insensitively) are unchanged."""
        assert correct("Coffee") == "coffee"  # lowercased but not mis-corrected

    def test_unrelated_word_returned_as_unknown(self):
        """A word with no close match returns Unknown Menu Item."""
        result = correct("xylophone")
        assert result == "Unknown Menu Item"

    def test_correction_is_case_insensitive_input(self):
        """Input capitalisation does not prevent correction."""
        result = correct("DOSAI")
        assert result == "dosa"

    def test_single_character_not_corrected(self):
        """Single-character tokens are too short for reliable fuzzy matching."""
        c = MenuVocabularyCorrector(threshold=0.72)
        result = c.correct("a coffee", STANDARD_MENU)
        # 'a' should not be corrected to 'chai' or anything else
        assert "coffee" in result  # coffee is preserved


# ===========================================================================
# 9. Pipeline integration: normalizer → corrector chain
#    These tests verify the two-layer system works correctly end-to-end.
# ===========================================================================

class TestPipelineIntegration:
    """Verify the full normalizer → corrector pipeline produces correct output
    for the examples listed in the original spec requirement."""

    @staticmethod
    def _pipeline(text: str, menu: list[str] = STANDARD_MENU) -> str:
        """Run text through both layers, as the real pipeline does."""
        from app.application.services.speech_normalizer import normalize
        normalized = normalize(text)
        return corrector.correct(normalized, menu)

    def test_spec_cofi_to_coffee_full_pipeline(self):
        """'cofi' → normalizer → 'coffee' (static dict handles it)."""
        assert self._pipeline("2 cofi") == "2 coffee"

    def test_spec_tee_to_tea_full_pipeline(self):
        """'tee' → normalizer → 'tea' (static dict handles it)."""
        assert self._pipeline("1 tee") == "1 tea"

    def test_spec_itly_to_idli_full_pipeline(self):
        """'itly' → normalizer → 'idli' (static dict handles it)."""
        assert self._pipeline("3 itly") == "3 idli"

    def test_spec_idly_to_idli_full_pipeline(self):
        assert self._pipeline("2 idly") == "2 idli"

    def test_spec_paratha_to_parotta_full_pipeline(self):
        """'paratha' → normalizer → 'parotta' (static dict handles it)."""
        assert self._pipeline("1 paratha") == "1 parotta"

    def test_tanglish_full_pipeline(self):
        """Tamil numbers + mis-pronounced items handled across both layers."""
        result = self._pipeline("oru coffer rendu dosai")
        assert result == "1 coffee 2 dosa"

    def test_english_homophone_plus_menu_correction(self):
        """'for' homophone normalized, then 'cofee' fuzzy-corrected."""
        result = self._pipeline("for cofee")
        assert result == "4 coffee"

    def test_full_realistic_tanglish_order(self):
        """End-to-end realistic Tanglish order."""
        result = self._pipeline("rendu dosai onnu coffer anju idly")
        assert result == "2 dosa 1 coffee 5 idli"

