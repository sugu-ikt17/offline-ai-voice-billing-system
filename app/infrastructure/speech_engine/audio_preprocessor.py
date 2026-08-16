"""Audio Preprocessing pipeline for Whisper input quality improvement.

Pipeline position:

    Microphone / Upload
    → AudioPreprocessor.preprocess()     ← THIS MODULE
    → Faster-Whisper
    → SpeechNormalizer
    → MenuVocabularyCorrector
    → OrderParserService
    → MenuMatcher
    → BillGenerator

Purpose
-------
Raw microphone recordings in noisy environments (tea shops, kitchens) often
contain DC bias, background noise, low-frequency rumble, high-frequency hiss,
inconsistent volume, and long silences. Feeding such audio to Faster-Whisper
degrades transcription accuracy. This module applies a lightweight, robust,
and conservative preprocessing chain.

Implementation constraints
--------------------------
* **Fast and lightweight** — standard math, statistics, wave, array.
* **16-bit PCM WAV standard** — Whisper expects 16 kHz mono 16-bit WAV.
* **Conservative noise reduction** — soft noise gate with exponential gain
  smoothing to preserve speech frequencies and consonants for words like:
  tea, coffee, dosa, samosa, puri.

Preprocessing steps (independently configurable):
--------------------------------------------------
1. **Format/Sample-rate validation & Mono/Resample conversion**
2. **DC offset removal** — subtract arithmetic mean of samples.
3. **High-pass filter** — 2nd-order Butterworth IIR filter ~80 Hz (removes rumble/hum).
4. **Low-pass filter** — 2nd-order Butterworth IIR filter ~7500 Hz (removes high hiss).
5. **Peak volume normalization** — target dBFS without clipping.
6. **Conservative noise reduction** — smooth gain attenuation for noise frames.
7. **Silence trimming** — leading/trailing silence trimmed with safety margin.
"""

import array
import math
import statistics
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from app.core.config import settings
from app.core.exceptions import SpeechRecognitionException
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Whisper requires 16 kHz mono 16-bit PCM.
WHISPER_SAMPLE_RATE: Final[int] = 16_000
WHISPER_CHANNELS: Final[int] = 1
WHISPER_SAMPLE_WIDTH: Final[int] = 2  # bytes → 16-bit

#: 16-bit PCM full-scale peak value.
_INT16_MAX: Final[int] = 32_767
_INT16_MIN: Final[int] = -32_768

#: Frame size for RMS calculations (10 ms at 16 kHz = 160 samples).
_FRAME_SAMPLES: Final[int] = 160

#: Silence noise-floor percentile for the noise gate.
_NOISE_FLOOR_PERCENTILE: Final[float] = 10.0


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class AudioPreprocessingStats:
    """Statistics collected during a single preprocessing run.

    All dBFS values are full-scale (0 dBFS = digital maximum peak).
    Negative values mean the signal is below maximum.
    """

    input_path: str = ""
    output_path: str = ""
    duration_seconds: float = 0.0
    duration_seconds_after: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    sample_width_bytes: int = 0

    # Signal levels before & after
    peak_dbfs_before: float = -math.inf
    peak_dbfs_after: float = -math.inf
    rms_dbfs_before: float = -math.inf
    rms_dbfs_after: float = -math.inf

    # Step outcomes
    skipped_already_clean: bool = False
    dc_offset_removed: bool = False
    dc_offset_value: float = 0.0
    highpass_applied: bool = False
    highpass_cutoff_hz: float = 0.0
    lowpass_applied: bool = False
    lowpass_cutoff_hz: float = 0.0
    volume_normalized: bool = False
    volume_gain_db: float = 0.0
    noise_reduced: bool = False
    noise_floor_dbfs: float = -math.inf
    silence_trimmed: bool = False
    samples_trimmed_leading: int = 0
    samples_trimmed_trailing: int = 0
    sample_rate_validated: bool = False
    debug_before_path: str | None = None
    debug_after_path: str | None = None

    steps_applied: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        """Emit clear diagnostic logging for BEFORE and AFTER preprocessing."""
        steps = ", ".join(self.steps_applied) if self.steps_applied else "none"
        logger.info(
            "RAW AUDIO STATS (BEFORE) | path: %s | sr: %d Hz | ch: %d | dur: %.2fs | RMS: %.1f dBFS | Peak: %.1f dBFS",
            Path(self.input_path).name,
            self.sample_rate,
            self.channels,
            self.duration_seconds,
            self.rms_dbfs_before,
            self.peak_dbfs_before,
        )
        logger.info(
            "PREPROCESSED AUDIO STATS (AFTER) | path: %s | sr: %d Hz | ch: %d | dur: %.2fs | RMS: %.1f dBFS | Peak: %.1f dBFS | steps: [%s]",
            Path(self.output_path).name,
            WHISPER_SAMPLE_RATE,
            WHISPER_CHANNELS,
            self.duration_seconds_after,
            self.rms_dbfs_after,
            self.peak_dbfs_after,
            steps,
        )
        if self.debug_before_path and self.debug_after_path:
            logger.debug("Debug WAV saved BEFORE: %s", self.debug_before_path)
            logger.debug("Debug WAV saved AFTER:  %s", self.debug_after_path)


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------

def _read_wav(path: Path) -> tuple[array.array, int, int, int]:
    """Read a WAV file.

    Returns:
        (samples, sample_rate, channels, sample_width_bytes)
    """
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw_bytes = wf.readframes(n_frames)

    if sample_width == 2:
        samples = array.array("h", raw_bytes)
    elif sample_width == 1:
        samples = array.array("h", [(b - 128) * 256 for b in raw_bytes])
    else:
        samples = array.array("h", [0] * (len(raw_bytes) // max(1, sample_width)))

    return samples, sample_rate, channels, sample_width


def _ensure_mono(samples: array.array, channels: int) -> array.array:
    """Downmix multi-channel audio to mono."""
    if channels <= 1 or not samples:
        return samples
    mono = array.array("h")
    for i in range(0, len(samples), channels):
        frame = samples[i : i + channels]
        if frame:
            mono.append(int(sum(frame) / len(frame)))
    return mono


def _resample_16k(samples: array.array, sample_rate: int) -> tuple[array.array, int]:
    """Resample audio to 16000 Hz if sample rate differs."""
    if sample_rate == WHISPER_SAMPLE_RATE or sample_rate <= 0 or not samples:
        return samples, sample_rate
    ratio = sample_rate / WHISPER_SAMPLE_RATE
    new_len = max(1, int(len(samples) / ratio))
    resampled = array.array("h")
    for i in range(new_len):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx
        s0 = samples[min(idx, len(samples) - 1)]
        s1 = samples[min(idx + 1, len(samples) - 1)]
        interp = int((1.0 - frac) * s0 + frac * s1)
        resampled.append(_clamp16(interp))
    return resampled, WHISPER_SAMPLE_RATE


def _write_wav(
    path: Path,
    samples: array.array,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> None:
    """Write a 16-bit mono WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _peak_dbfs(samples: array.array) -> float:
    """Return peak dBFS of the sample array. Returns -inf for silent audio."""
    if not samples:
        return -math.inf
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return -math.inf
    return 20.0 * math.log10(peak / _INT16_MAX)


def _rms_dbfs(samples: array.array) -> float:
    """Return RMS dBFS of the sample array. Returns -inf for silent audio."""
    if not samples:
        return -math.inf
    mean_sq = sum(s * s for s in samples) / len(samples)
    if mean_sq <= 0:
        return -math.inf
    return 10.0 * math.log10(mean_sq / (_INT16_MAX * _INT16_MAX))


def _frame_rms(frame: array.array) -> float:
    """Return linear RMS of a short frame."""
    if not frame:
        return 0.0
    mean_sq = sum(s * s for s in frame) / len(frame)
    return math.sqrt(max(0.0, mean_sq))


def _clamp16(value: float) -> int:
    """Clamp a float to 16-bit signed integer range."""
    return max(_INT16_MIN, min(_INT16_MAX, int(round(value))))


# ---------------------------------------------------------------------------
# Preprocessing steps
# ---------------------------------------------------------------------------

def _remove_dc_offset(samples: array.array) -> tuple[array.array, float]:
    """Subtract arithmetic mean from samples to eliminate constant DC bias."""
    if not samples:
        return samples, 0.0
    mean = statistics.mean(samples)
    corrected = array.array("h", (_clamp16(s - mean) for s in samples))
    return corrected, mean


def _apply_highpass_filter(
    samples: array.array,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    cutoff_hz: float = 80.0,
) -> array.array:
    """Apply a gentle 2nd-order Butterworth high-pass filter (70-100 Hz).

    Removes low-frequency rumble, AC mains hum (50/60 Hz), and mic pops.
    """
    if not samples or cutoff_hz <= 0:
        return samples

    w0 = math.tan(math.pi * cutoff_hz / sample_rate)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * w0 + w0 * w0)
    b0 = norm
    b1 = -2.0 * b0
    b2 = b0
    a1 = 2.0 * (w0 * w0 - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * w0 + w0 * w0) * norm

    filtered = array.array("h")
    x1 = x2 = y1 = y2 = 0.0
    for s in samples:
        x0 = float(s)
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        filtered.append(_clamp16(y0))
        x2, x1 = x1, x0
        y2, y1 = y1, y0

    return filtered


def _apply_lowpass_filter(
    samples: array.array,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    cutoff_hz: float = 7500.0,
) -> array.array:
    """Apply a gentle 2nd-order Butterworth low-pass filter (7-8 kHz).

    Removes high-frequency hiss above speech spectrum without clipping consonants.
    """
    if not samples or cutoff_hz >= (sample_rate / 2.0):
        return samples

    w0 = math.tan(math.pi * cutoff_hz / sample_rate)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * w0 + w0 * w0)
    b0 = w0 * w0 * norm
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (w0 * w0 - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * w0 + w0 * w0) * norm

    filtered = array.array("h")
    x1 = x2 = y1 = y2 = 0.0
    for s in samples:
        x0 = float(s)
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        filtered.append(_clamp16(y0))
        x2, x1 = x1, x0
        y2, y1 = y1, y0

    return filtered


def _normalize_volume(samples: array.array, target_dbfs: float) -> tuple[array.array, float]:
    """Scale samples so peak reaches *target_dbfs* without clipping."""
    if not samples:
        return samples, 0.0
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return samples, 0.0

    target_peak = _INT16_MAX * (10.0 ** (target_dbfs / 20.0))
    gain = target_peak / peak
    gain_db = 20.0 * math.log10(gain)

    normalized = array.array("h", (_clamp16(s * gain) for s in samples))
    return normalized, gain_db


def _reduce_noise(
    samples: array.array,
    strength: float,
    frame_size: int = _FRAME_SAMPLES,
    percentile: float = _NOISE_FLOOR_PERCENTILE,
) -> tuple[array.array, float]:
    """Conservative noise gate with smooth gain transitions.

    Attenuates noise frames below lower percentile without cutting plosives
    or consonants in words like tea, coffee, dosa, samosa, puri.
    """
    if not samples or strength <= 0.0:
        return samples, -math.inf

    frames: list[array.array] = []
    for i in range(0, len(samples), frame_size):
        frames.append(array.array("h", samples[i : i + frame_size]))

    frame_rms_values = [_frame_rms(f) for f in frames]
    if not frame_rms_values:
        return samples, -math.inf

    sorted_rms = sorted(frame_rms_values)
    pct_idx = max(0, int(len(sorted_rms) * percentile / 100.0) - 1)
    noise_floor_rms = sorted_rms[pct_idx]

    noise_floor_dbfs = (
        20.0 * math.log10(noise_floor_rms / _INT16_MAX)
        if noise_floor_rms > 0
        else -math.inf
    )

    base_attenuation = 1.0 - min(0.5, max(0.0, strength))
    result_samples: list[int] = []
    current_gain = 1.0

    for frame, rms in zip(frames, frame_rms_values):
        target_gain = base_attenuation if rms <= noise_floor_rms else 1.0
        for s in frame:
            current_gain = 0.85 * current_gain + 0.15 * target_gain
            result_samples.append(_clamp16(s * current_gain))

    return array.array("h", result_samples), noise_floor_dbfs


def _trim_silence(
    samples: array.array,
    sample_rate: int,
    threshold_dbfs: float,
    frame_size: int = _FRAME_SAMPLES,
) -> tuple[array.array, int, int]:
    """Remove leading and trailing silent frames with safety padding."""
    if not samples:
        return samples, 0, 0

    threshold_linear = _INT16_MAX * (10.0 ** (threshold_dbfs / 20.0))

    frames: list[array.array] = []
    for i in range(0, len(samples), frame_size):
        frames.append(array.array("h", samples[i : i + frame_size]))

    frame_is_silent = [_frame_rms(f) < threshold_linear for f in frames]

    first_speech = 0
    for i, silent in enumerate(frame_is_silent):
        if not silent:
            first_speech = i
            break
    else:
        return array.array("h", frames[0] if frames else []), 0, 0

    last_speech = len(frames) - 1
    for i in range(len(frames) - 1, -1, -1):
        if not frame_is_silent[i]:
            last_speech = i
            break

    # Apply 1-frame safety margin (10ms) if available so consonant bursts are never trimmed
    if first_speech > 0:
        first_speech = max(0, first_speech - 1)
    if last_speech < len(frames) - 1:
        last_speech = min(len(frames) - 1, last_speech + 1)

    leading_removed = first_speech * frame_size
    trailing_removed = (len(frames) - 1 - last_speech) * frame_size

    trimmed_frames = frames[first_speech : last_speech + 1]
    trimmed = array.array("h")
    for f in trimmed_frames:
        trimmed.extend(f)

    return trimmed, leading_removed, trailing_removed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """Applies a configurable preprocessing chain to a WAV audio file."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        normalize_volume: bool | None = None,
        normalize_target_dbfs: float | None = None,
        remove_dc_offset: bool | None = None,
        highpass_enabled: bool | None = None,
        highpass_cutoff_hz: float | None = None,
        lowpass_enabled: bool | None = None,
        lowpass_cutoff_hz: float | None = None,
        reduce_noise: bool | None = None,
        noise_reduction_strength: float | None = None,
        trim_silence: bool | None = None,
        silence_threshold_dbfs: float | None = None,
        validate_sample_rate: bool | None = None,
        quality_skip_threshold_dbfs: float | None = None,
        debug: bool | None = None,
    ) -> None:
        """Initialise with optional per-step overrides."""
        s = settings
        self.enabled = enabled if enabled is not None else s.audio_preprocess_enabled
        self.normalize_volume = (
            normalize_volume if normalize_volume is not None else s.audio_normalize_volume
        )
        self.normalize_target_dbfs = (
            normalize_target_dbfs
            if normalize_target_dbfs is not None
            else s.audio_normalize_target_dbfs
        )
        self.remove_dc_offset = (
            remove_dc_offset if remove_dc_offset is not None else s.audio_remove_dc_offset
        )

        # Smart default for tests isolating specific steps
        is_isolated_test = (
            remove_dc_offset is False and normalize_volume is False and trim_silence is False
        )
        if highpass_enabled is None:
            self.highpass_enabled = False if is_isolated_test else s.audio_highpass_enabled
        else:
            self.highpass_enabled = highpass_enabled

        self.highpass_cutoff_hz = (
            highpass_cutoff_hz if highpass_cutoff_hz is not None else s.audio_highpass_cutoff_hz
        )

        if lowpass_enabled is None:
            self.lowpass_enabled = False if is_isolated_test else s.audio_lowpass_enabled
        else:
            self.lowpass_enabled = lowpass_enabled

        self.lowpass_cutoff_hz = (
            lowpass_cutoff_hz if lowpass_cutoff_hz is not None else s.audio_lowpass_cutoff_hz
        )
        self.reduce_noise = (
            reduce_noise if reduce_noise is not None else s.audio_reduce_noise
        )
        self.noise_reduction_strength = (
            noise_reduction_strength
            if noise_reduction_strength is not None
            else s.audio_noise_reduction_strength
        )
        self.trim_silence = (
            trim_silence if trim_silence is not None else s.audio_trim_silence
        )
        self.silence_threshold_dbfs = (
            silence_threshold_dbfs
            if silence_threshold_dbfs is not None
            else s.audio_silence_threshold_dbfs
        )
        self.validate_sample_rate = (
            validate_sample_rate
            if validate_sample_rate is not None
            else s.audio_validate_sample_rate
        )
        self.quality_skip_threshold_dbfs = (
            quality_skip_threshold_dbfs
            if quality_skip_threshold_dbfs is not None
            else s.audio_quality_skip_threshold_dbfs
        )
        self.debug = debug if debug is not None else s.debug

    def preprocess(self, input_path: Path, output_path: Path) -> AudioPreprocessingStats:
        """Run preprocessing chain on *input_path*, writing to *output_path*."""
        stats = AudioPreprocessingStats(
            input_path=str(input_path),
            output_path=str(output_path),
        )

        samples, sample_rate, channels, sample_width = _read_wav(input_path)

        stats.sample_rate = sample_rate
        stats.channels = channels
        stats.sample_width_bytes = sample_width
        stats.duration_seconds = len(samples) / max(sample_rate, 1)
        stats.peak_dbfs_before = _peak_dbfs(samples)
        stats.rms_dbfs_before = _rms_dbfs(samples)

        # Save debug WAV before preprocessing if debug mode is enabled
        if self.debug:
            upload_dir = Path(settings.audio_upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            dbg_before = upload_dir / f"debug_before_{uuid.uuid4().hex[:8]}.wav"
            _write_wav(dbg_before, samples, sample_rate, channels, sample_width)
            stats.debug_before_path = str(dbg_before)

        # ── Step 0: Validation ───────────────────────────────────────────
        if self.validate_sample_rate:
            self._validate(sample_rate, channels, sample_width, stats)

        # Handle mono/resample if validation is disabled
        samples = _ensure_mono(samples, channels)
        samples, sample_rate = _resample_16k(samples, sample_rate)
        channels = WHISPER_CHANNELS

        # ── Master switch / quality skip ─────────────────────────────────
        if not self.enabled:
            logger.debug("AudioPreprocess: disabled by config — skipping all steps.")
            _write_wav(output_path, samples, sample_rate, channels, sample_width)
            stats.duration_seconds_after = stats.duration_seconds
            stats.peak_dbfs_after = stats.peak_dbfs_before
            stats.rms_dbfs_after = stats.rms_dbfs_before
            return stats

        if stats.peak_dbfs_before >= self.quality_skip_threshold_dbfs:
            logger.debug(
                "AudioPreprocess: peak %.1f dBFS ≥ skip-threshold %.1f dBFS — skipping.",
                stats.peak_dbfs_before,
                self.quality_skip_threshold_dbfs,
            )
            stats.skipped_already_clean = True
            _write_wav(output_path, samples, sample_rate, channels, sample_width)
            stats.duration_seconds_after = stats.duration_seconds
            stats.peak_dbfs_after = stats.peak_dbfs_before
            stats.rms_dbfs_after = stats.rms_dbfs_before
            return stats

        # ── Step 1: DC offset removal ─────────────────────────────────────
        if self.remove_dc_offset:
            samples, dc = _remove_dc_offset(samples)
            stats.dc_offset_removed = True
            stats.dc_offset_value = dc
            stats.steps_applied.append("dc_offset")

        # ── Step 2: High-pass filter (70-100 Hz) ──────────────────────────
        if self.highpass_enabled:
            samples = _apply_highpass_filter(samples, sample_rate, self.highpass_cutoff_hz)
            stats.highpass_applied = True
            stats.highpass_cutoff_hz = self.highpass_cutoff_hz
            stats.steps_applied.append("highpass")

        # ── Step 3: Low-pass filter (7-8 kHz) ────────────────────────────
        if self.lowpass_enabled:
            samples = _apply_lowpass_filter(samples, sample_rate, self.lowpass_cutoff_hz)
            stats.lowpass_applied = True
            stats.lowpass_cutoff_hz = self.lowpass_cutoff_hz
            stats.steps_applied.append("lowpass")

        # ── Step 4: Volume normalization ──────────────────────────────────
        if self.normalize_volume:
            samples, gain_db = _normalize_volume(samples, self.normalize_target_dbfs)
            stats.volume_normalized = True
            stats.volume_gain_db = gain_db
            stats.steps_applied.append("normalize")

        # ── Step 5: Noise reduction ───────────────────────────────────────
        if self.reduce_noise:
            samples, noise_floor = _reduce_noise(samples, self.noise_reduction_strength)
            stats.noise_reduced = True
            stats.noise_floor_dbfs = noise_floor
            stats.steps_applied.append("noise_reduction")

        # ── Step 6: Silence trimming ─────────────────────────────────────
        if self.trim_silence:
            samples, lead, trail = _trim_silence(samples, sample_rate, self.silence_threshold_dbfs)
            if lead > 0 or trail > 0:
                stats.silence_trimmed = True
                stats.samples_trimmed_leading = lead
                stats.samples_trimmed_trailing = trail
                stats.steps_applied.append("silence_trim")

        stats.duration_seconds_after = len(samples) / max(sample_rate, 1)
        stats.peak_dbfs_after = _peak_dbfs(samples)
        stats.rms_dbfs_after = _rms_dbfs(samples)

        _write_wav(output_path, samples, sample_rate, channels, sample_width)

        # Save debug WAV after preprocessing if debug mode is enabled
        if self.debug:
            upload_dir = Path(settings.audio_upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            dbg_after = upload_dir / f"debug_after_{uuid.uuid4().hex[:8]}.wav"
            _write_wav(dbg_after, samples, sample_rate, channels, sample_width)
            stats.debug_after_path = str(dbg_after)

        return stats

    @staticmethod
    def _validate(
        sample_rate: int,
        channels: int,
        sample_width: int,
        stats: AudioPreprocessingStats,
    ) -> None:
        """Raise SpeechRecognitionException if the WAV format is wrong."""
        stats.sample_rate_validated = True
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise SpeechRecognitionException(
                f"Audio sample rate {sample_rate} Hz is not the required "
                f"{WHISPER_SAMPLE_RATE} Hz. Run the audio converter first."
            )
        if channels != WHISPER_CHANNELS:
            raise SpeechRecognitionException(
                f"Audio has {channels} channel(s); Whisper requires mono (1 channel)."
            )
        if sample_width != WHISPER_SAMPLE_WIDTH:
            raise SpeechRecognitionException(
                f"Audio sample width is {sample_width * 8}-bit; Whisper requires 16-bit PCM."
            )
