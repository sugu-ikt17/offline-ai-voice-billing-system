#!/usr/bin/env python3
"""Benchmark Script for Voice Billing System Pipeline.

Tests text and order pipeline latency, parsing accuracy, and menu matching across:
  - "2 tea 3 dosa"
  - "3 coffee 2 tea 4 samosa"
  - "tea 2 dosa 3"
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.application.services.speech_normalizer import normalize
from app.application.services.menu_vocabulary_corrector import MenuVocabularyCorrector
from app.application.services.order_parser_service import OrderParserService
from app.application.services.menu_matcher_service import MenuMatcherService
from app.infrastructure.database.database import SessionLocal, Base, engine
from app.infrastructure.database.repositories.menu_repository import MenuRepository


def run_benchmark():
    print("==========================================================================")
    print("BENCHMARKING OFFLINE AI VOICE BILLING SYSTEM PIPELINE")
    print("==========================================================================")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        menu_repo = MenuRepository(db)
        menu_items = menu_repo.get_all()
        vocab_list = [m.name for m in menu_items] if menu_items else []

        corrector = MenuVocabularyCorrector()
        parser = OrderParserService()
        matcher = MenuMatcherService(menu_repo)

        test_inputs = [
            "2 tea 3 dosa",
            "3 coffee 2 tea 4 samosa",
            "tea 2 dosa 3",
        ]

        results = []

        for idx, text in enumerate(test_inputs, 1):
            print(f"\n--- Test Case {idx}: {text!r} ---")
            t_start = time.perf_counter()

            # 1. Normalization
            t0 = time.perf_counter()
            norm_text = normalize(text)
            t_norm = time.perf_counter() - t0

            # 2. Vocabulary Correction
            t0 = time.perf_counter()
            vocab_text = corrector.correct(norm_text, vocab_list)
            t_vocab = time.perf_counter() - t0

            # 3. Order Parsing
            t0 = time.perf_counter()
            parsed_result = parser.parse_with_details(vocab_text, vocabulary=vocab_list)
            parsed_items = parsed_result.recognized_items
            t_parse = time.perf_counter() - t0

            # 4. Menu Matching
            t0 = time.perf_counter()
            match_result = matcher.match(parsed_items)
            t_match = time.perf_counter() - t0

            t_total = time.perf_counter() - t_start

            matched_summary = [
                {"name": item.name, "qty": item.quantity, "subtotal": item.line_total}
                for item in match_result.matched_items
            ]

            print(f"RAW TRANSCRIPT : {text}")
            print(f"NORMALIZED     : {norm_text}")
            print(f"VOCAB CORRECTED: {vocab_text}")
            print(f"PARSED RESULT  : {parsed_items}")
            print(f"MATCHED RESULT : {matched_summary}")
            print(f"UNMATCHED      : {match_result.unmatched_items}")
            print(f"TIMINGS        : norm={t_norm*1000:.2f}ms vocab={t_vocab*1000:.2f}ms parse={t_parse*1000:.2f}ms match={t_match*1000:.2f}ms total={t_total*1000:.2f}ms")

            results.append({
                "input": text,
                "parsed": parsed_items,
                "matched": matched_summary,
                "total_time_ms": t_total * 1000.0
            })

        print("\n==========================================================================")
        print("SUMMARY RESULTS")
        print("==========================================================================")
        for r in results:
            print(f"Input: {r['input']:<25} | Parsed Items: {len(r['parsed'])} | Matched Items: {len(r['matched'])} | Latency: {r['total_time_ms']:.2f}ms")
        print("==========================================================================")

    finally:
        db.close()


if __name__ == "__main__":
    run_benchmark()
