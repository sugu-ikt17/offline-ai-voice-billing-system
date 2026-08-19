"""Application configuration, loaded from environment variables / .env file.

Whisper.cpp settings can be overridden by environment variables:
  WHISPER_BINARY_PATH  — absolute path to the compiled whisper.cpp binary
  WHISPER_MODEL_PATH   — absolute path to the GGML model file
  WHISPER_MODEL_NAME   — model filename (default: ggml-base.bin)
                          used only when WHISPER_MODEL_PATH is not set explicitly

Faster-Whisper performance tuning:
  WHISPER_BEAM_SIZE  — beam search width (default: 1 = greedy, fastest; 5 = accurate)
                       Set to 1 for sub-second inference on GPU; raise to 5 for max accuracy.
  WHISPER_LANGUAGE   — ISO language code (e.g. 'ta' for Tamil, 'en' for English).
                       Set to 'auto' or '' to enable automatic language detection.
                       Skipping auto-detection saves ~50 ms per request.

Menu Vocabulary Corrector:
  VOCAB_CORRECTOR_THRESHOLD — float 0–1, minimum difflib similarity for a
                              fuzzy menu-word correction to be accepted.
                              Default: 0.72

Audio Preprocessing (all steps independently toggled by env var):
  AUDIO_PREPROCESS_ENABLED         — master on/off switch (default: true)
  AUDIO_NORMALIZE_VOLUME           — peak-normalize to target dBFS (default: true)
  AUDIO_NORMALIZE_TARGET_DBFS      — target peak level in dBFS (default: -6.0)
  AUDIO_REMOVE_DC_OFFSET           — subtract mean sample value (default: true)
  AUDIO_REDUCE_NOISE               — spectral-subtraction noise reduction (default: true)
  AUDIO_NOISE_REDUCTION_STRENGTH   — noise gate alpha 0–1 (default: 0.15)
  AUDIO_TRIM_SILENCE               — strip leading/trailing silence (default: true)
  AUDIO_SILENCE_THRESHOLD_DBFS     — silence level in dBFS below which to trim (default: -50.0)
  AUDIO_VALIDATE_SAMPLE_RATE       — enforce 16 kHz mono before transcription (default: true)
  AUDIO_QUALITY_SKIP_THRESHOLD_DBFS— skip preprocessing when signal is already
                                      loud/clean enough (default: -6.0)

Performance / Debug:
  DEBUG_TIMING  — when true, emit per-phase timing breakdown at DEBUG level
                  (audio write, conversion, preprocessing, inference, normalization,
                   vocabulary correction). Default: false.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Offline AI Voice Billing System"
    api_v1_prefix: str = "/api/v1"

    # Billing
    tax_rate: float = 0.05  # 5% default tax

    # Faster-Whisper — override via WHISPER_MODEL_NAME / WHISPER_DEVICE / WHISPER_COMPUTE_TYPE env vars
    # Default: base — fast CPU inference for short shop-order commands (3–6 s audio).
    # For higher accuracy at the cost of latency, set WHISPER_MODEL_NAME=distil-large-v3.
    whisper_model_name: str = "distil-large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_binary_path: str = str(BASE_DIR / "models" / "whisper" / "main")
    whisper_model_path: str = str(BASE_DIR / "models" / "whisper" / "ggml-base.bin")

    # Faster-Whisper performance tuning
    # beam_size=1 (greedy decode) is 3–5x faster than beam_size=5 for short
    # shop-order audio clips (< 10 s) with negligible accuracy loss.
    whisper_beam_size: int = 1
    # ISO language code. Avoids per-request language detection overhead.
    # Set to empty string or 'auto' to re-enable automatic detection.
    whisper_language: str = "ta"

    # Sarvam AI STT settings (Cloud-based STT requiring active internet connectivity)
    sarvam_api_key: str = ""
    sarvam_model: str = "saaras:v3"
    sarvam_language_code: str = "ta-IN"
    sarvam_mode: str = "transcribe"

    # Audio uploads (temporary storage before transcription)
    audio_upload_dir: str = str(BASE_DIR / "data" / "audio_uploads")

    # Menu Vocabulary Corrector — fuzzy similarity threshold (0.0–1.0).
    # Raise to require closer matches; lower to accept more corrections.
    # Override via VOCAB_CORRECTOR_THRESHOLD env var.
    vocab_corrector_threshold: float = 0.72

    # ----------------------------------------------------------------
    # Audio Preprocessing
    # Each step can be independently disabled via an environment variable.
    # Defaults are tuned for Raspberry Pi Zero 2 W (lightweight, no AI).
    # ----------------------------------------------------------------

    #: Master switch — set AUDIO_PREPROCESS_ENABLED=false to bypass entirely.
    audio_preprocess_enabled: bool = True

    #: Peak-volume normalization — brings quiet recordings up to target level.
    #: -6.0 dBFS gives 6 dB of headroom above IIR filter transients and noise-gate
    #: transitions that can momentarily exceed the pre-filter peak.  -3.0 was too
    #: close to 0 dBFS and caused downstream clipping artifacts.
    audio_normalize_volume: bool = True
    audio_normalize_target_dbfs: float = -6.0  # target peak in dBFS (was -3.0)

    #: DC offset removal — subtracts the mean sample value (cheap, one pass).
    audio_remove_dc_offset: bool = True

    #: Gentle high-pass filter — removes low-frequency rumble / AC mains hum (70-100 Hz).
    audio_highpass_enabled: bool = True
    audio_highpass_cutoff_hz: float = 80.0

    #: Gentle low-pass filter — removes high-frequency hiss (7-8 kHz).
    audio_lowpass_enabled: bool = True
    audio_lowpass_cutoff_hz: float = 7500.0

    #: Lightweight spectral-gate noise reduction (no AI model required).
    #: Adjust strength (0.0-1.0) via AUDIO_NOISE_REDUCTION_STRENGTH env var without code changes.
    audio_reduce_noise: bool = True
    audio_noise_reduction_strength: float = 0.15  # gate alpha, 0.0–1.0

    #: Silence trimming — strips leading/trailing silence frames.
    audio_trim_silence: bool = True
    audio_silence_threshold_dbfs: float = -50.0  # RMS level below which = silence

    #: Sample-rate / channel validation (raises if wrong format reaches Whisper).
    audio_validate_sample_rate: bool = True

    #: If the audio peak is already above this level, skip preprocessing to
    #: avoid double-processing clean recordings.  Set to -0.1 to always process.
    #: IMPORTANT: must be above the normalize_target_dbfs (-6.0 dBFS) so that
    #: recordings that are already loud (e.g. clipping at 0.0 dBFS from browser
    #: AGC) are NOT silently skipped — they still need DC removal and filtering.
    #: Previous value of -6.0 dBFS matched the normalize target exactly, causing
    #: near-full-scale recordings to bypass all preprocessing and reach Whisper
    #: with distortion intact.
    audio_quality_skip_threshold_dbfs: float = -1.0  # was -6.0; skip only near-true-clipping

    # ----------------------------------------------------------------
    # Performance / Debug
    # ----------------------------------------------------------------

    #: Master debug switch — set DEBUG=true in environment or .env file.
    debug: bool = True

    #: When True, emit a per-phase timing breakdown at DEBUG level covering:
    #:   audio write, format conversion, preprocessing, Whisper inference,
    #:   normalization, and vocabulary correction.
    debug_timing: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
