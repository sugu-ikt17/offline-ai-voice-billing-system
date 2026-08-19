"""Script to test real Sarvam STT engine integration with SpeechService."""

import os
from pathlib import Path
from gtts import gTTS

from app.application.services.speech_service import SpeechService
from app.infrastructure.speech_engine.sarvam_engine import SarvamSpeechEngine
from app.core.config import settings
from app.infrastructure.database.database import SessionLocal
from app.infrastructure.database.repositories.menu_repository import MenuRepository
from app.application.services.order_parser_service import OrderParserService
from app.application.services.menu_matcher_service import MenuMatcherService

def main():
    print("Initializing DB session & repositories...")
    db = SessionLocal()
    menu_repo = MenuRepository(db)

    print("Initializing Sarvam Speech Engine & SpeechService...")
    sarvam_engine = SarvamSpeechEngine(
        api_key=settings.sarvam_api_key,
        model=settings.sarvam_model,
        language_code=settings.sarvam_language_code,
        mode=settings.sarvam_mode,
    )
    speech_service = SpeechService(engine=sarvam_engine, menu_repository=menu_repo)
    parser = OrderParserService()
    matcher = MenuMatcherService(menu_repo)

    temp_dir = Path("data/scratch_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)

    test_phrases = [
        "2 டீ 2 தோசை 4 சமோசா",
        "2 டீ நன்றி 2 தோசை 4 சமோசா"
    ]

    for idx, phrase in enumerate(test_phrases, start=1):
        print(f"\n========================================================")
        print(f"REAL SARVAM TEST #{idx}: Spoken input = {phrase!r}")
        print(f"========================================================")
        mp3_path = temp_dir / f"test_{idx}.mp3"
        tts = gTTS(text=phrase, lang="ta")
        tts.save(str(mp3_path))

        # Transcribe through SpeechService
        transcript = speech_service.transcribe(str(mp3_path))
        print(f"\n[FINAL RETURNED TRANSCRIPT FROM SPEECH SERVICE]: {transcript!r}")

        # Parse & Match
        parsed_items = parser.parse(transcript)
        match_result = matcher.match(parsed_items)

        print("\nPARSED ITEMS:")
        for item in parsed_items:
            print(f"  - Item: {item['item']!r}, Quantity: {item['quantity']}")

        print("\nFINAL MATCHED BILL ITEMS:")
        for item in match_result.matched_items:
            print(f"  - {item.name} × {item.quantity} (@ ₹{item.unit_price:.2f} = ₹{item.line_total:.2f})")

        print("\nUNMATCHED ITEMS:")
        print(f"  {match_result.unmatched_items}")

        # Cleanup
        mp3_path.unlink(missing_ok=True)

    db.close()

if __name__ == "__main__":
    main()
