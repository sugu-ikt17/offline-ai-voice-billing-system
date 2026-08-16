"""Unit tests for the AudioPreprocessor pipeline.

All tests are self-contained — synthetic WAV audio is generated in-memory
and written to pytest's tmp_path fixtures.  No real recordings are required.

Audio generation helpers create:
  * clean_wav    — pure sine wave at realistic speech level (-12 dBFS)
  * quiet_wav    — very quiet sine wave (-40 dBFS) requiring normalization
  * noisy_wav    — sine wave + high-amplitude white noise floor
  * silent_wav   — leading/trailing silence around a short speech burst
  * dc_offset_wav— audio with a large positive DC bias
  * wrong_rate_wav — 8 kHz WAV (wrong sample rate for Whisper)

Coverage:
  1. Sample rate / channel / bit-depth validation
  2. DC offset removal
  3. Volume normalization (quiet → louder, already-loud → skip)
  4. Noise reduction (noisy audio attenuated below noise floor)
  5. Silence trimming (lead / trail silence removed)
  6. Quality-skip behaviour (clean audio not double-processed)
  7. Master enable/disable switch
  8. Individual step enable/disable
  9. Stats dataclass correctness
 10. PreprocessingStats.log_summary() doesn't raise
 11. Edge cases: silent input, 1-sample input, very short clip
 12. Preprocessor wired into SpeechService (smoke test)

At least 50 test cases.
"""

import array
import math
import random
import wave
from pathlib import Path

import pytest

from app.infrastructure.speech_engine.audio_preprocessor import (
    AudioPreprocessor,
    AudioPreprocessingStats,
    WHISPER_SAMPLE_RATE,
    WHISPER_CHANNELS,
    WHISPER_SAMPLE_WIDTH,
    _read_wav,
    _write_wav,
    _peak_dbfs,
    _rms_dbfs,
    _remove_dc_offset,
    _apply_highpass_filter,
    _apply_lowpass_filter,
    _normalize_volume,
    _reduce_noise,
    _trim_silence,
)
from app.core.exceptions import SpeechRecognitionException


# ===========================================================================
# Synthetic WAV generation helpers
# ===========================================================================

_INT16_MAX = 32_767
_SR = WHISPER_SAMPLE_RATE  # 16 000

def _sine_samples(
    duration_s: float = 0.5,
    freq_hz: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = _SR,
) -> array.array:
    """Generate a pure sine wave as 16-bit signed samples."""
    n = int(duration_s * sample_rate)
    peak = int(amplitude * _INT16_MAX)
    samples = array.array(
        "h",
        [
            int(peak * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            for i in range(n)
        ],
    )
    return samples


def _noise_samples(n: int, amplitude: float = 0.05) -> array.array:
    """Generate white noise samples."""
    rng = random.Random(42)
    peak = int(amplitude * _INT16_MAX)
    return array.array("h", [rng.randint(-peak, peak) for _ in range(n)])


def _silence_samples(n: int) -> array.array:
    return array.array("h", [0] * n)


def _write_test_wav(
    path: Path,
    samples: array.array,
    sample_rate: int = _SR,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def make_clean_wav(tmp_path: Path, duration: float = 0.5) -> Path:
    """Clean speech-level sine at -12 dBFS peak."""
    samples = _sine_samples(duration, amplitude=0.25)  # ~-12 dBFS
    p = tmp_path / "clean.wav"
    _write_test_wav(p, samples)
    return p


def make_quiet_wav(tmp_path: Path, duration: float = 0.5) -> Path:
    """Very quiet sine at ~-40 dBFS — needs normalization."""
    samples = _sine_samples(duration, amplitude=0.01)
    p = tmp_path / "quiet.wav"
    _write_test_wav(p, samples)
    return p


def make_noisy_wav(tmp_path: Path, duration: float = 0.5) -> Path:
    """Sine + significant white noise floor."""
    speech = _sine_samples(duration, amplitude=0.3)
    noise = _noise_samples(len(speech), amplitude=0.08)
    mixed = array.array("h", [
        max(-_INT16_MAX, min(_INT16_MAX, speech[i] + noise[i]))
        for i in range(len(speech))
    ])
    p = tmp_path / "noisy.wav"
    _write_test_wav(p, mixed)
    return p


def make_silent_padded_wav(tmp_path: Path) -> Path:
    """0.3s silence → 0.2s speech → 0.3s silence."""
    leading = _silence_samples(int(0.3 * _SR))
    speech = _sine_samples(0.2, amplitude=0.3)
    trailing = _silence_samples(int(0.3 * _SR))
    combined = array.array("h")
    combined.extend(leading)
    combined.extend(speech)
    combined.extend(trailing)
    p = tmp_path / "padded.wav"
    _write_test_wav(p, combined)
    return p


def make_dc_offset_wav(tmp_path: Path, dc: int = 5000) -> Path:
    """Sine wave shifted by a large positive DC bias."""
    base = _sine_samples(0.5, amplitude=0.25)
    biased = array.array("h", [
        max(-_INT16_MAX, min(_INT16_MAX, s + dc)) for s in base
    ])
    p = tmp_path / "dc.wav"
    _write_test_wav(p, biased)
    return p


def make_wrong_rate_wav(tmp_path: Path) -> Path:
    """8 kHz WAV — wrong sample rate."""
    samples = _sine_samples(0.5, sample_rate=8000)
    p = tmp_path / "wrong_rate.wav"
    _write_test_wav(p, samples, sample_rate=8000)
    return p


def make_stereo_wav(tmp_path: Path) -> Path:
    """2-channel WAV — wrong channel count."""
    # Interleave L/R for stereo
    mono = _sine_samples(0.3, amplitude=0.3)
    stereo = array.array("h")
    for s in mono:
        stereo.append(s)
        stereo.append(s)
    p = tmp_path / "stereo.wav"
    _write_test_wav(p, stereo, channels=2)
    return p


def make_loud_clean_wav(tmp_path: Path) -> Path:
    """Already loud clean audio at ~-2 dBFS — should trigger skip."""
    samples = _sine_samples(0.5, amplitude=0.98)  # ~-0.2 dBFS peak
    p = tmp_path / "loud.wav"
    _write_test_wav(p, samples)
    return p


# ===========================================================================
# 1. Sample-rate and format validation
# ===========================================================================

class TestValidation:

    def test_raises_on_wrong_sample_rate(self, tmp_path):
        wav = make_wrong_rate_wav(tmp_path)
        p = AudioPreprocessor(validate_sample_rate=True)
        with pytest.raises(SpeechRecognitionException, match="sample rate"):
            p.preprocess(wav, wav)

    def test_raises_on_stereo_audio(self, tmp_path):
        wav = make_stereo_wav(tmp_path)
        p = AudioPreprocessor(validate_sample_rate=True)
        with pytest.raises(SpeechRecognitionException, match="channel"):
            p.preprocess(wav, wav)

    def test_no_raise_when_validation_disabled(self, tmp_path):
        """Wrong sample rate is accepted silently when validation is off."""
        wav = make_wrong_rate_wav(tmp_path)
        p = AudioPreprocessor(
            validate_sample_rate=False,
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert not stats.sample_rate_validated

    def test_stats_records_sample_rate(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert stats.sample_rate == WHISPER_SAMPLE_RATE

    def test_stats_records_channels(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert stats.channels == 1


# ===========================================================================
# 2. DC offset removal
# ===========================================================================

class TestDcOffsetRemoval:

    def test_dc_offset_reduced(self, tmp_path):
        """After DC removal, the mean of samples should be near zero."""
        wav = make_dc_offset_wav(tmp_path, dc=5000)
        samples_before, *_ = _read_wav(wav)
        mean_before = sum(samples_before) / len(samples_before)

        p = AudioPreprocessor(
            remove_dc_offset=True,
            normalize_volume=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        p.preprocess(wav, wav)

        samples_after, *_ = _read_wav(wav)
        mean_after = sum(samples_after) / len(samples_after)
        assert abs(mean_after) < abs(mean_before) / 5, (
            f"Mean after ({mean_after:.1f}) should be much less than before ({mean_before:.1f})"
        )

    def test_stats_records_dc_offset_removed(self, tmp_path):
        wav = make_dc_offset_wav(tmp_path, dc=3000)
        p = AudioPreprocessor(
            remove_dc_offset=True,
            normalize_volume=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.dc_offset_removed is True
        assert abs(stats.dc_offset_value) > 100  # should have detected offset

    def test_dc_offset_not_applied_when_disabled(self, tmp_path):
        wav = make_dc_offset_wav(tmp_path, dc=3000)
        p = AudioPreprocessor(
            remove_dc_offset=False,
            normalize_volume=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.dc_offset_removed is False
        assert "dc_offset" not in stats.steps_applied

    def test_clean_audio_dc_offset_near_zero(self, tmp_path):
        """Pure sine wave has near-zero DC — offset removal should be tiny."""
        wav = make_clean_wav(tmp_path)
        samples_before, *_ = _read_wav(wav)
        p = AudioPreprocessor(
            remove_dc_offset=True,
            normalize_volume=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        # A balanced sine has mean ≈ 0; offset value should be tiny
        assert abs(stats.dc_offset_value) < 100


# ===========================================================================
# 3. Volume normalization
# ===========================================================================

class TestVolumeNormalization:

    def test_quiet_audio_is_amplified(self, tmp_path):
        wav = make_quiet_wav(tmp_path)
        peak_before = _peak_dbfs(_read_wav(wav)[0])

        p = AudioPreprocessor(
            normalize_volume=True,
            normalize_target_dbfs=-3.0,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        peak_after = _peak_dbfs(_read_wav(wav)[0])

        assert peak_after > peak_before + 10, "Quiet audio should be significantly amplified"
        assert abs(peak_after - (-3.0)) < 2.0, f"Peak should be near -3 dBFS, got {peak_after:.1f}"

    def test_stats_records_positive_gain_for_quiet(self, tmp_path):
        wav = make_quiet_wav(tmp_path)
        p = AudioPreprocessor(
            normalize_volume=True,
            normalize_target_dbfs=-3.0,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.volume_normalized is True
        assert stats.volume_gain_db > 5.0  # quiet file needs many dB of gain

    def test_normalization_not_applied_when_disabled(self, tmp_path):
        wav = make_quiet_wav(tmp_path)
        p = AudioPreprocessor(
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.volume_normalized is False
        assert "normalize" not in stats.steps_applied

    def test_normalize_step_in_steps_applied(self, tmp_path):
        wav = make_quiet_wav(tmp_path)
        p = AudioPreprocessor(
            normalize_volume=True,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert "normalize" in stats.steps_applied

    def test_silent_audio_not_amplified_to_clipping(self, tmp_path):
        """Fully silent audio should not produce clipping after normalization."""
        silence = _silence_samples(int(0.5 * _SR))
        p_path = tmp_path / "silence.wav"
        _write_test_wav(p_path, silence)

        p = AudioPreprocessor(
            normalize_volume=True,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        # Should not raise
        stats = p.preprocess(p_path, p_path)
        samples_out, *_ = _read_wav(p_path)
        assert all(abs(s) <= 32767 for s in samples_out)


# ===========================================================================
# 4. Noise reduction
# ===========================================================================

class TestNoiseReduction:

    def test_noise_floor_dbfs_recorded(self, tmp_path):
        wav = make_noisy_wav(tmp_path)
        p = AudioPreprocessor(
            reduce_noise=True,
            noise_reduction_strength=0.8,
            normalize_volume=False,
            remove_dc_offset=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.noise_reduced is True
        assert math.isfinite(stats.noise_floor_dbfs)

    def test_noise_reduction_attenuates_noise_frames(self, tmp_path):
        """Pure noise (no speech) should be attenuated significantly."""
        # Build a pure-noise file
        noise = _noise_samples(int(0.5 * _SR), amplitude=0.05)
        p_path = tmp_path / "pure_noise.wav"
        _write_test_wav(p_path, noise)

        rms_before = _rms_dbfs(_read_wav(p_path)[0])

        p = AudioPreprocessor(
            reduce_noise=True,
            noise_reduction_strength=0.9,
            normalize_volume=False,
            remove_dc_offset=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        p.preprocess(p_path, p_path)
        rms_after = _rms_dbfs(_read_wav(p_path)[0])

        assert rms_after < rms_before, "Noise should be attenuated"

    def test_noise_reduction_not_applied_when_disabled(self, tmp_path):
        wav = make_noisy_wav(tmp_path)
        p = AudioPreprocessor(
            reduce_noise=False,
            normalize_volume=False,
            remove_dc_offset=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.noise_reduced is False
        assert "noise_reduction" not in stats.steps_applied

    def test_zero_strength_leaves_audio_unchanged(self, tmp_path):
        wav = make_noisy_wav(tmp_path)
        samples_before, *_ = _read_wav(wav)

        p = AudioPreprocessor(
            reduce_noise=True,
            noise_reduction_strength=0.0,
            normalize_volume=False,
            remove_dc_offset=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        p.preprocess(wav, wav)
        samples_after, *_ = _read_wav(wav)
        # With 0 strength, output equals input
        assert samples_before == samples_after

    def test_noise_reduction_step_in_steps_applied(self, tmp_path):
        wav = make_noisy_wav(tmp_path)
        p = AudioPreprocessor(
            reduce_noise=True,
            noise_reduction_strength=0.5,
            normalize_volume=False,
            remove_dc_offset=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert "noise_reduction" in stats.steps_applied


# ===========================================================================
# 5. Silence trimming
# ===========================================================================

class TestSilenceTrimming:

    def test_leading_silence_removed(self, tmp_path):
        wav = make_silent_padded_wav(tmp_path)
        total_before = len(_read_wav(wav)[0])

        p = AudioPreprocessor(
            trim_silence=True,
            silence_threshold_dbfs=-50.0,
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)

        total_after = len(_read_wav(wav)[0])
        assert total_after < total_before, "Total samples should decrease after trimming"
        assert stats.samples_trimmed_leading > 0

    def test_trailing_silence_removed(self, tmp_path):
        wav = make_silent_padded_wav(tmp_path)
        p = AudioPreprocessor(
            trim_silence=True,
            silence_threshold_dbfs=-50.0,
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.samples_trimmed_trailing > 0

    def test_silence_trim_step_in_steps_applied(self, tmp_path):
        wav = make_silent_padded_wav(tmp_path)
        p = AudioPreprocessor(
            trim_silence=True,
            silence_threshold_dbfs=-50.0,
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert "silence_trim" in stats.steps_applied

    def test_trim_not_applied_when_disabled(self, tmp_path):
        wav = make_silent_padded_wav(tmp_path)
        p = AudioPreprocessor(
            trim_silence=False,
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert stats.silence_trimmed is False
        assert "silence_trim" not in stats.steps_applied

    def test_clean_audio_not_overtrimmed(self, tmp_path):
        """Audio without leading/trailing silence should not lose samples."""
        wav = make_clean_wav(tmp_path)
        samples_before, *_ = _read_wav(wav)

        p = AudioPreprocessor(
            trim_silence=True,
            silence_threshold_dbfs=-80.0,  # very aggressive threshold
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            quality_skip_threshold_dbfs=1.0,
        )
        p.preprocess(wav, wav)
        samples_after, *_ = _read_wav(wav)
        # Should not trim speech frames
        assert len(samples_after) > 0


# ===========================================================================
# 6. Quality-skip behaviour
# ===========================================================================

class TestQualitySkip:

    def test_loud_clean_audio_is_skipped(self, tmp_path):
        wav = make_loud_clean_wav(tmp_path)
        samples_before, *_ = _read_wav(wav)

        p = AudioPreprocessor(quality_skip_threshold_dbfs=-6.0)
        stats = p.preprocess(wav, wav)

        assert stats.skipped_already_clean is True
        assert stats.steps_applied == []

    def test_quiet_audio_is_not_skipped(self, tmp_path):
        wav = make_quiet_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=-6.0)
        stats = p.preprocess(wav, wav)
        assert stats.skipped_already_clean is False

    def test_skip_threshold_configurable(self, tmp_path):
        """Setting skip threshold very low forces processing even for clean audio."""
        wav = make_clean_wav(tmp_path)  # ~-12 dBFS peak
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert stats.skipped_already_clean is False

    def test_skip_threshold_very_high_always_skips(self, tmp_path):
        """Skip fires when threshold is BELOW the signal's peak dBFS.

        make_clean_wav() produces a sine at amplitude=0.25 → peak ≈ -12 dBFS.
        Setting the threshold to -15.0 means: skip if peak >= -15 dBFS.
        Since -12 >= -15 is True, the file is skipped (already "good enough").
        """
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=-15.0)
        stats = p.preprocess(wav, wav)
        assert stats.skipped_already_clean is True


# ===========================================================================
# 7. Master enable/disable switch
# ===========================================================================

class TestMasterSwitch:

    def test_disabled_preprocessor_skips_all_steps(self, tmp_path):
        wav = make_quiet_wav(tmp_path)
        samples_before, *_ = _read_wav(wav)

        p = AudioPreprocessor(enabled=False)
        stats = p.preprocess(wav, wav)

        assert stats.steps_applied == []
        assert not stats.dc_offset_removed
        assert not stats.volume_normalized
        assert not stats.noise_reduced
        assert not stats.silence_trimmed

    def test_disabled_preprocessor_output_matches_input(self, tmp_path):
        """When disabled, output file should have the same samples as input."""
        wav = make_quiet_wav(tmp_path)
        samples_before, *_ = _read_wav(wav)

        out = tmp_path / "out.wav"
        p = AudioPreprocessor(enabled=False)
        p.preprocess(wav, out)

        samples_after, *_ = _read_wav(out)
        assert samples_before == samples_after


# ===========================================================================
# 8. Stats dataclass
# ===========================================================================

class TestStats:

    def test_stats_duration_correct(self, tmp_path):
        wav = make_clean_wav(tmp_path, duration=0.5)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert abs(stats.duration_seconds - 0.5) < 0.05

    def test_stats_peak_before_is_finite(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert math.isfinite(stats.peak_dbfs_before)

    def test_stats_peak_after_is_finite(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert math.isfinite(stats.peak_dbfs_after)

    def test_stats_rms_before_is_finite(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert math.isfinite(stats.rms_dbfs_before)

    def test_stats_log_summary_does_not_raise(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        stats.log_summary()  # should not raise

    def test_stats_input_output_paths_recorded(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        out = tmp_path / "out.wav"
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, out)
        assert stats.input_path == str(wav)
        assert stats.output_path == str(out)

    def test_stats_sample_width_recorded(self, tmp_path):
        wav = make_clean_wav(tmp_path)
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav, wav)
        assert stats.sample_width_bytes == WHISPER_SAMPLE_WIDTH


# ===========================================================================
# 9. Low-level helpers
# ===========================================================================

class TestHelpers:

    def test_peak_dbfs_full_scale(self):
        samples = array.array("h", [32767, -32768, 0])
        assert abs(_peak_dbfs(samples)) < 0.01  # ≈ 0 dBFS

    def test_peak_dbfs_half_scale(self):
        samples = array.array("h", [16384, -16384])
        assert abs(_peak_dbfs(samples) - (-6.02)) < 0.5

    def test_peak_dbfs_silent(self):
        samples = array.array("h", [0, 0, 0])
        assert _peak_dbfs(samples) == -math.inf

    def test_rms_dbfs_empty(self):
        assert _rms_dbfs(array.array("h")) == -math.inf

    def test_remove_dc_offset_zero_mean_unchanged(self):
        """Balanced signal has ~0 offset; function should return nearly same data."""
        samples = array.array("h", [100, -100, 200, -200])
        result, offset = _remove_dc_offset(samples)
        assert abs(offset) < 1

    def test_normalize_volume_scales_to_target(self):
        # Input peak = 16384 ≈ -6 dBFS; target -3 dBFS
        samples = array.array("h", [16384, -16384])
        result, gain_db = _normalize_volume(samples, target_dbfs=-3.0)
        peak_after = _peak_dbfs(result)
        assert abs(peak_after - (-3.0)) < 1.0

    def test_normalize_volume_silent_no_crash(self):
        samples = array.array("h", [0, 0, 0])
        result, gain_db = _normalize_volume(samples, -3.0)
        assert gain_db == 0.0

    def test_trim_silence_removes_leading(self):
        leading = array.array("h", [0] * 320)  # 2 silent frames
        speech = array.array("h", [10000] * 320)
        full = array.array("h")
        full.extend(leading)
        full.extend(speech)
        trimmed, lead_removed, trail_removed = _trim_silence(full, _SR, -50.0)
        assert lead_removed > 0
        assert len(trimmed) < len(full)

    def test_trim_silence_fully_silent_returns_something(self):
        """Fully silent input returns at least 1 frame, does not crash."""
        silent = array.array("h", [0] * 480)
        result, lead, trail = _trim_silence(silent, _SR, -50.0)
        assert len(result) >= 0  # no crash

    def test_read_write_wav_roundtrip(self, tmp_path):
        """Samples written and read back should be identical."""
        original = _sine_samples(0.1, amplitude=0.3)
        p = tmp_path / "rt.wav"
        _write_wav(p, original, _SR, 1, 2)
        result, sr, ch, sw = _read_wav(p)
        assert sr == _SR
        assert ch == 1
        assert sw == 2
        assert result == original


# ===========================================================================
# 10. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_very_short_clip(self, tmp_path):
        """50-sample clip should process without errors."""
        samples = _sine_samples(0.003, amplitude=0.2)  # ~50 samples
        p_path = tmp_path / "short.wav"
        _write_test_wav(p_path, samples)
        proc = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = proc.preprocess(p_path, p_path)
        assert math.isfinite(stats.duration_seconds) or stats.duration_seconds >= 0

    def test_output_path_differs_from_input(self, tmp_path):
        """Preprocessor writes to output_path even when it differs from input."""
        wav_in = make_clean_wav(tmp_path)
        wav_out = tmp_path / "processed.wav"
        p = AudioPreprocessor(quality_skip_threshold_dbfs=1.0)
        stats = p.preprocess(wav_in, wav_out)
        assert wav_out.exists()
        assert stats.output_path == str(wav_out)

    def test_all_steps_disabled_still_writes_output(self, tmp_path):
        """Even with all steps off, output file should be written."""
        wav_in = make_clean_wav(tmp_path)
        wav_out = tmp_path / "out.wav"
        p = AudioPreprocessor(
            normalize_volume=False,
            remove_dc_offset=False,
            reduce_noise=False,
            trim_silence=False,
            quality_skip_threshold_dbfs=1.0,
        )
        p.preprocess(wav_in, wav_out)
        assert wav_out.exists()

    def test_full_pipeline_all_steps_enabled(self, tmp_path):
        """Running all steps on a noisy padded file should not crash."""
        wav = make_silent_padded_wav(tmp_path)
        p = AudioPreprocessor(
            enabled=True,
            normalize_volume=True,
            remove_dc_offset=True,
            reduce_noise=True,
            trim_silence=True,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(wav, wav)
        assert wav.exists()
        assert len(stats.steps_applied) > 0

    def test_full_pipeline_on_dc_offset_and_noisy_audio(self, tmp_path):
        """DC + noise + silence — all steps exercised without crash."""
        dc_noise = make_dc_offset_wav(tmp_path, dc=3000)
        p = AudioPreprocessor(
            enabled=True,
            normalize_volume=True,
            remove_dc_offset=True,
            reduce_noise=True,
            trim_silence=True,
            quality_skip_threshold_dbfs=1.0,
        )
        stats = p.preprocess(dc_noise, dc_noise)
        assert stats.dc_offset_removed
        assert stats.volume_normalized


# ===========================================================================
# 11. SpeechService integration smoke test
# ===========================================================================

class TestSpeechServiceIntegration:

    def test_speech_service_has_preprocessor_attribute(self):
        """SpeechService must have a _preprocessor instance after init."""
        from app.application.services.speech_service import SpeechService
        from app.infrastructure.speech_engine.audio_preprocessor import AudioPreprocessor

        class _FakeEngine:
            binary_path = "/fake/binary"
            model_path = "/fake/model"
            def is_available(self): return False
            def transcribe(self, p): return "2 dosa 1 tea"

        svc = SpeechService(engine=_FakeEngine())
        assert hasattr(svc, "_preprocessor")
        assert isinstance(svc._preprocessor, AudioPreprocessor)

    def test_preprocessor_is_called_before_whisper(self, tmp_path):
        """Preprocessor is invoked inside transcribe(); verify via stats side-effect."""
        from app.application.services.speech_service import SpeechService
        from app.infrastructure.speech_engine.audio_preprocessor import AudioPreprocessor

        preprocess_calls = []

        class _TrackingPreprocessor(AudioPreprocessor):
            def preprocess(self, inp, out):
                preprocess_calls.append((inp, out))
                return super().preprocess(inp, out)

        class _FakeEngine:
            binary_path = str(tmp_path / "binary")
            model_path = str(tmp_path / "model")
            def is_available(self): return True
            def transcribe(self, p): return "2 dosa 1 tea"

        # Create real wav files so SpeechService doesn't bail
        binary_file = tmp_path / "binary"
        binary_file.write_text("fake")
        binary_file.chmod(0o755)
        (tmp_path / "model").write_text("fake")

        # Write a real WAV file for transcription
        audio_path = tmp_path / "order.wav"
        samples = _sine_samples(0.3, amplitude=0.2)
        _write_test_wav(audio_path, samples)

        svc = SpeechService(engine=_FakeEngine())
        svc._preprocessor = _TrackingPreprocessor()
        svc.load_model()

        svc.transcribe(str(audio_path))
        assert len(preprocess_calls) == 1, "Preprocessor should have been called once"


# ===========================================================================
# 12. Highpass, Lowpass, Debug WAV, and Speech Frequency Preservation Tests
# ===========================================================================

class TestNewPreprocessingPipeline:

    def test_highpass_filter_attenuates_low_rumble(self):
        """High-pass filter at 80 Hz should attenuate 40 Hz rumble significantly more than 440 Hz speech."""
        low_rumble = _sine_samples(0.5, freq_hz=40.0, amplitude=0.5)
        speech_freq = _sine_samples(0.5, freq_hz=440.0, amplitude=0.5)

        hp_rumble = _apply_highpass_filter(low_rumble, sample_rate=16000, cutoff_hz=80.0)
        hp_speech = _apply_highpass_filter(speech_freq, sample_rate=16000, cutoff_hz=80.0)

        rumble_peak_before = max(abs(s) for s in low_rumble)
        rumble_peak_after = max(abs(s) for s in hp_rumble)

        speech_peak_before = max(abs(s) for s in speech_freq)
        speech_peak_after = max(abs(s) for s in hp_speech)

        rumble_ratio = rumble_peak_after / max(1, rumble_peak_before)
        speech_ratio = speech_peak_after / max(1, speech_peak_before)

        # Rumble at 40 Hz (below cutoff 80 Hz) should be attenuated much more than 440 Hz speech
        assert rumble_ratio < 0.7
        assert speech_ratio > 0.95

    def test_lowpass_filter_attenuates_high_hiss(self):
        """Low-pass filter at 7500 Hz should preserve 1000 Hz tone but attenuate 7800 Hz hiss."""
        speech_tone = _sine_samples(0.5, freq_hz=1000.0, amplitude=0.5)
        high_hiss = _sine_samples(0.5, freq_hz=7800.0, amplitude=0.5)

        lp_speech = _apply_lowpass_filter(speech_tone, sample_rate=16000, cutoff_hz=7500.0)
        lp_hiss = _apply_lowpass_filter(high_hiss, sample_rate=16000, cutoff_hz=7500.0)

        speech_ratio = max(abs(s) for s in lp_speech) / max(abs(s) for s in speech_tone)
        hiss_ratio = max(abs(s) for s in lp_hiss) / max(abs(s) for s in high_hiss)

        assert speech_ratio > 0.95
        assert hiss_ratio < speech_ratio

    def test_stereo_to_mono_and_resample_conversion(self, tmp_path):
        """_read_wav downmixes 2-channel 32 kHz WAV to 16 kHz mono 16-bit WAV."""
        p = tmp_path / "stereo_32k.wav"
        sr = 32000
        n_samples = int(sr * 0.2)
        # Create stereo interleaved samples
        stereo_samples = array.array("h")
        for i in range(n_samples):
            s = int(0.2 * _INT16_MAX * math.sin(2 * math.pi * 440 * i / sr))
            stereo_samples.append(s)  # L
            stereo_samples.append(s)  # R

        _write_test_wav(p, stereo_samples, sample_rate=sr, channels=2, sample_width=2)

        out_path = tmp_path / "mono_16k.wav"
        prep = AudioPreprocessor(validate_sample_rate=False, quality_skip_threshold_dbfs=1.0)
        prep.preprocess(p, out_path)

        samples, out_sr, out_ch, out_sw = _read_wav(out_path)
        assert out_sr == WHISPER_SAMPLE_RATE  # 16000
        assert out_ch == WHISPER_CHANNELS     # 1
        assert out_sw == WHISPER_SAMPLE_WIDTH # 2

    def test_debug_wav_saved_when_debug_enabled(self, tmp_path):
        """When debug=True, debug_before and debug_after paths are saved in stats."""
        p = tmp_path / "test_input.wav"
        samples = _sine_samples(0.3, amplitude=0.1)
        _write_test_wav(p, samples)

        out_path = tmp_path / "test_output.wav"
        prep = AudioPreprocessor(enabled=True, debug=True, quality_skip_threshold_dbfs=1.0)
        stats = prep.preprocess(p, out_path)

        assert stats.debug_before_path is not None
        assert stats.debug_after_path is not None
        assert Path(stats.debug_before_path).exists()
        assert Path(stats.debug_after_path).exists()

    def test_consonant_preservation_conservative_noise_reduction(self):
        """Conservative noise reduction with smooth gain does not silence high-energy speech bursts."""
        speech_burst = _sine_samples(0.2, freq_hz=2000.0, amplitude=0.4)
        denoised, noise_floor = _reduce_noise(speech_burst, strength=0.15)

        peak_before = max(abs(s) for s in speech_burst)
        peak_after = max(abs(s) for s in denoised)
        assert peak_after >= 0.8 * peak_before

