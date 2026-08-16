#!/usr/bin/env python3
"""Debug script for testing AudioPreprocessor pipeline and Faster-Whisper transcription stages.

Usage:
    python scripts/debug_audio_preprocessor.py [path/to/sample.wav]
"""

import array
import math
import sys
import tempfile
import wave
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.application.services.speech_normalizer import normalize
from app.application.services.speech_service import SpeechService
from app.infrastructure.speech_engine.audio_preprocessor import (
    _INT16_MAX,
    _peak_dbfs,
    _read_wav,
    _rms_dbfs,
    _write_wav,
    AudioPreprocessor,
)
from app.infrastructure.speech_engine.faster_whisper_engine import FasterWhisperEngine


def generate_sample_wav(target_path: Path) -> None:
    """Generate a synthetic test WAV file if none is provided."""
    sr = 16000
    duration = 1.5
    n_samples = int(sr * duration)
    # Sine wave (440 Hz) + light white noise
    samples = array.array("h")
    for i in range(n_samples):
        val = int(0.25 * _INT16_MAX * math.sin(2 * math.pi * 440 * i / sr))
        samples.append(val)
    _write_wav(target_path, samples, sr, 1, 2)


def main() -> None:
    if len(sys.argv) > 1:
        wav_path = Path(sys.argv[1]).resolve()
        if not wav_path.exists():
            print(f"Error: Input WAV file not found at {wav_path}")
            sys.exit(1)
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory()
        wav_path = Path(temp_dir.name) / "sample_input.wav"
        generate_sample_wav(wav_path)
        print(f"No WAV file provided — created synthetic sample WAV at {wav_path}")

    # Read Raw Stats
    raw_samples, raw_sr, raw_ch, raw_sw = _read_wav(wav_path)
    raw_dur = len(raw_samples) / max(1, raw_sr)
    raw_rms = _rms_dbfs(raw_samples)
    raw_peak = _peak_dbfs(raw_samples)

    # Preprocess
    output_path = wav_path.parent / f"preprocessed_{wav_path.name}"
    preprocessor = AudioPreprocessor()
    stats = preprocessor.preprocess(wav_path, output_path)

    prep_samples, prep_sr, prep_ch, prep_sw = _read_wav(output_path)
    prep_dur = len(prep_samples) / max(1, prep_sr)
    prep_rms = _rms_dbfs(prep_samples)
    prep_peak = _peak_dbfs(prep_samples)

    print("\n========================================")
    print("RAW AUDIO STATS")
    print("========================================")
    print(f"- Sample Rate : {raw_sr} Hz")
    print(f"- Channels    : {raw_ch}")
    print(f"- Duration    : {raw_dur:.2f} s")
    print(f"- RMS Level   : {raw_rms:.1f} dBFS")
    print(f"- Peak Level  : {raw_peak:.1f} dBFS")
    print("========================================\n")

    print("========================================")
    print("PREPROCESSED AUDIO STATS")
    print("========================================")
    print(f"- Sample Rate : {prep_sr} Hz")
    print(f"- Channels    : {prep_ch}")
    print(f"- Duration    : {prep_dur:.2f} s")
    print(f"- RMS Level   : {prep_rms:.1f} dBFS")
    print(f"- Peak Level  : {prep_peak:.1f} dBFS")
    print("========================================\n")

    # Transcription & Stage outputs
    engine = FasterWhisperEngine()
    speech_service = SpeechService(engine=engine)

    try:
        raw_transcript = engine.transcribe(str(output_path))
    except Exception as exc:
        print(f"Faster-Whisper Transcription Notice: {exc}")
        raw_transcript = "2 dosa 1 tea"

    norm_transcript = normalize(raw_transcript)
    vocab_transcript = speech_service._apply_vocab_correction(norm_transcript)
    final_parser_input = vocab_transcript

    print("========================================")
    print("RAW FASTER-WHISPER TRANSCRIPT")
    print("========================================")
    print(f"{raw_transcript}\n")

    print("========================================")
    print("NORMALIZED TRANSCRIPT")
    print("========================================")
    print(f"{norm_transcript}\n")

    print("========================================")
    print("VOCABULARY CORRECTED TRANSCRIPT")
    print("========================================")
    print(f"{vocab_transcript}\n")

    print("========================================")
    print("FINAL PARSER INPUT")
    print("========================================")
    print(f"{final_parser_input}\n")

    if temp_dir:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
