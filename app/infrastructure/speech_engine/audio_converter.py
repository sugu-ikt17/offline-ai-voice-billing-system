"""Audio format conversion utilities for Whisper.cpp compatibility.

Whisper.cpp requires 16-bit mono WAV audio at 16 kHz.  Browser MediaRecorder
outputs WebM/Opus by default, or WAV at 48000 Hz / stereo.

This module automatically converts any uploaded or recorded audio to:
  - 16000 Hz sample rate
  - Mono (1 channel)
  - 16-bit signed PCM WAV

Conversion strategies (tried in order):
  1. ffmpeg subprocess (most reliable, handles WebM, OGG, MP3, 48kHz WAV, etc.).
  2. Pure-Python stdlib WAV resampler (fallback for WAV files if ffmpeg is unavailable).

Logs original sample rate, converted sample rate (16000 Hz), and conversion time.
"""

import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

from app.core.exceptions import SpeechRecognitionException
from app.core.logging import get_logger

logger = get_logger(__name__)

_WAV_FORMAT = ".wav"
_CONVERTIBLE_FORMATS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac", ".aac"}

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes → 16-bit PCM


def _get_ffmpeg_binary() -> str | None:
    """Return path to an available ffmpeg binary, or None."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    venv_ffmpeg = Path(__file__).resolve().parents[3] / ".venv" / "bin" / "ffmpeg"
    if venv_ffmpeg.exists() and os.access(venv_ffmpeg, os.X_OK):
        return str(venv_ffmpeg)
    return None


def _get_wav_info(path: Path) -> tuple[int, int, int] | None:
    """Return (sample_rate, channels, sample_width) if path is a valid WAV, else None."""
    if path.suffix.lower() != _WAV_FORMAT:
        return None
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    except Exception:
        return None


def get_audio_duration(path: Path) -> float:
    """Return the duration of a WAV audio file in seconds (rounded to 2 decimal places)."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 2) if rate > 0 else 0.0
    except Exception:
        return 0.0


def is_target_format(path: Path) -> bool:
    """Return True if *path* is already a 16000 Hz mono 16-bit PCM WAV."""
    info = _get_wav_info(path)
    if not info:
        return False
    sr, ch, sw = info
    return sr == TARGET_SAMPLE_RATE and ch == TARGET_CHANNELS and sw == TARGET_SAMPLE_WIDTH


def convert_to_wav(input_path: Path, output_path: Path, profiler: Any = None) -> None:
    """Convert *input_path* to a 16000 Hz mono 16-bit WAV at *output_path*.

    Handles any input format (WAV at 48000 Hz / stereo, WebM, OGG, MP3, etc.).
    If the input is already 16000 Hz mono 16-bit WAV, skips conversion.

    Args:
        input_path:  Absolute path of the source audio file.
        output_path: Destination path for the converted WAV file.
        profiler:    Optional PipelineProfiler instance.

    Raises:
        SpeechRecognitionException: if conversion fails or input format is unsupported.
    """
    if not input_path.exists():
        raise SpeechRecognitionException(f"Audio file not found at: {input_path}")

    ext = input_path.suffix.lower()
    if ext not in _CONVERTIBLE_FORMATS:
        raise SpeechRecognitionException(
            f"Unsupported audio format '{ext}'. Accepted: wav, webm, ogg, mp3, m4a, flac."
        )

    start_time = time.perf_counter()
    t0_decode = time.perf_counter()
    wav_info = _get_wav_info(input_path)
    if profiler and hasattr(profiler, "record_stage"):
        profiler.record_stage("Decode", (time.perf_counter() - t0_decode) * 1000.0)

    orig_sample_rate = wav_info[0] if wav_info else None

    # Check if already 16000 Hz mono 16-bit WAV
    if ext == _WAV_FORMAT and is_target_format(input_path):
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        logger.debug("Audio is already 16000 Hz mono 16-bit WAV; skipping conversion: %s", input_path.name)
        if profiler and hasattr(profiler, "record_stage"):
            profiler.record_stage("Resample", 0.0)
        return

    t0_resample = time.perf_counter()
    # Temporary file if input == output to avoid in-place ffmpeg overwrite issues
    temp_output: Path | None = None
    target_dest = output_path
    if input_path.resolve() == output_path.resolve():
        temp_output = output_path.parent / f"tmp_conv_{output_path.name}"
        target_dest = temp_output

    try:
        ffmpeg_bin = _get_ffmpeg_binary()
        if ffmpeg_bin:
            logger.info("Converting %s → 16000 Hz mono WAV via ffmpeg (%s)", input_path.name, ffmpeg_bin)
            _convert_via_ffmpeg(ffmpeg_bin, input_path, target_dest)
        elif ext == _WAV_FORMAT:
            logger.info("Converting %s → 16000 Hz mono WAV via Python resampler", input_path.name)
            _convert_wav_python(input_path, target_dest)
        else:
            raise SpeechRecognitionException(
                f"Cannot convert non-WAV '{ext}' format: ffmpeg is not installed."
            )

        if temp_output and temp_output.exists():
            shutil.move(temp_output, output_path)

        if profiler and hasattr(profiler, "record_stage"):
            profiler.record_stage("Resample", (time.perf_counter() - t0_resample) * 1000.0)

        conversion_time = time.perf_counter() - start_time
        orig_rate_str = f"{orig_sample_rate}" if orig_sample_rate else "unknown"

        logger.info(
            "Audio conversion completed: original_sample_rate=%s Hz, converted_sample_rate=%d Hz, conversion_time=%.3fs (%s → %s)",
            orig_rate_str,
            TARGET_SAMPLE_RATE,
            conversion_time,
            input_path.name,
            output_path.name,
        )

    except Exception:
        if temp_output and temp_output.exists():
            temp_output.unlink(missing_ok=True)
        raise


def _convert_via_ffmpeg(ffmpeg_bin: str, input_path: Path, output_path: Path) -> None:
    """Run ffmpeg to produce a 16 kHz mono 16-bit PCM WAV."""
    cmd = [
        ffmpeg_bin,
        "-y",                       # overwrite output without asking
        "-i", str(input_path),      # input file
        "-ar", str(TARGET_SAMPLE_RATE),  # 16000 Hz
        "-ac", str(TARGET_CHANNELS),     # 1 (mono)
        "-sample_fmt", "s16",       # 16-bit PCM
        str(output_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace")
        logger.error("ffmpeg conversion error: %s", err)
        raise SpeechRecognitionException(f"Audio conversion via ffmpeg failed: {err}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SpeechRecognitionException("Audio conversion via ffmpeg timed out.") from exc


def _convert_wav_python(input_path: Path, output_path: Path) -> None:
    """Pure-Python stdlib fallback converter/resampler for WAV files."""
    import array
    import struct

    try:
        with wave.open(str(input_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
    except Exception as exc:
        raise SpeechRecognitionException(f"Failed to read WAV file: {exc}") from exc

    if sampwidth == 2:
        samples = array.array("h", raw_bytes)
    elif sampwidth == 1:
        samples = array.array("h", [(b - 128) * 256 for b in raw_bytes])
    elif sampwidth == 3:
        samples = array.array("h")
        for i in range(0, len(raw_bytes), 3):
            b0, b1, b2 = raw_bytes[i], raw_bytes[i + 1], raw_bytes[i + 2]
            val = struct.unpack("<i", bytes([b0, b1, b2, 0xFF if b2 & 0x80 else 0x00]))[0] >> 8
            samples.append(max(-32768, min(32767, val)))
    else:
        raise SpeechRecognitionException(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    # Downmix multi-channel to mono
    if n_channels > 1:
        mono = array.array("h")
        for i in range(0, len(samples), n_channels):
            frame = samples[i : i + n_channels]
            avg = int(sum(frame) / len(frame))
            mono.append(avg)
        samples = mono

    # Resample to 16000 Hz if sample rate differs
    if framerate != TARGET_SAMPLE_RATE:
        ratio = framerate / TARGET_SAMPLE_RATE
        new_len = max(1, int(len(samples) / ratio))
        resampled = array.array("h")
        for i in range(new_len):
            pos = i * ratio
            idx = int(pos)
            frac = pos - idx
            s0 = samples[min(idx, len(samples) - 1)]
            s1 = samples[min(idx + 1, len(samples) - 1)]
            interp = int((1.0 - frac) * s0 + frac * s1)
            resampled.append(max(-32768, min(32767, interp)))
        samples = resampled

    # Write target WAV
    try:
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(TARGET_CHANNELS)
            wf.setsampwidth(TARGET_SAMPLE_WIDTH)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(samples.tobytes())
    except Exception as exc:
        raise SpeechRecognitionException(f"Failed to write converted WAV file: {exc}") from exc
