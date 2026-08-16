"""Unit tests for audio_converter module.

Tests automatic conversion of 48000 Hz / stereo WAV files and non-WAV formats
to 16000 Hz mono 16-bit PCM WAV required by Whisper.cpp.
"""

import array
import math
import wave
from pathlib import Path

import pytest

from app.core.exceptions import SpeechRecognitionException
from app.infrastructure.speech_engine.audio_converter import (
    convert_to_wav,
    is_target_format,
    _convert_wav_python,
)
from app.infrastructure.speech_engine.audio_preprocessor import _read_wav


def _create_wav(
    path: Path,
    sample_rate: int = 48000,
    channels: int = 2,
    duration_s: float = 0.2,
) -> None:
    """Helper to generate a test WAV file with specified sample rate and channels."""
    n_samples = int(sample_rate * duration_s)
    mono_samples = [int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]

    if channels == 1:
        data = array.array("h", mono_samples).tobytes()
    else:
        interleaved = array.array("h")
        for s in mono_samples:
            interleaved.append(s)
            interleaved.append(s)
        data = interleaved.tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def test_is_target_format(tmp_path):
    target_wav = tmp_path / "target.wav"
    _create_wav(target_wav, sample_rate=16000, channels=1)

    non_target_wav = tmp_path / "browser_48k.wav"
    _create_wav(non_target_wav, sample_rate=48000, channels=2)

    assert is_target_format(target_wav) is True
    assert is_target_format(non_target_wav) is False


def test_convert_48k_stereo_to_16k_mono(tmp_path):
    src = tmp_path / "browser_48k_stereo.wav"
    dst = tmp_path / "converted_16k_mono.wav"
    _create_wav(src, sample_rate=48000, channels=2)

    convert_to_wav(src, dst)

    assert dst.exists()
    assert is_target_format(dst) is True

    samples, sr, ch, sw = _read_wav(dst)
    assert sr == 16000
    assert ch == 1
    assert sw == 2


def test_convert_already_compliant_wav(tmp_path):
    src = tmp_path / "already_16k.wav"
    dst = tmp_path / "output.wav"
    _create_wav(src, sample_rate=16000, channels=1)

    convert_to_wav(src, dst)

    assert dst.exists()
    assert is_target_format(dst) is True


def test_python_fallback_resampler(tmp_path):
    src = tmp_path / "browser_48k_python.wav"
    dst = tmp_path / "python_16k.wav"
    _create_wav(src, sample_rate=48000, channels=2)

    _convert_wav_python(src, dst)

    assert dst.exists()
    assert is_target_format(dst) is True


def test_convert_nonexistent_file_raises(tmp_path):
    src = tmp_path / "missing.wav"
    dst = tmp_path / "out.wav"

    with pytest.raises(SpeechRecognitionException, match="Audio file not found"):
        convert_to_wav(src, dst)


def test_convert_unsupported_format_raises(tmp_path):
    src = tmp_path / "test.xyz"
    src.write_text("dummy")
    dst = tmp_path / "out.wav"

    with pytest.raises(SpeechRecognitionException, match="Unsupported audio format"):
        convert_to_wav(src, dst)
