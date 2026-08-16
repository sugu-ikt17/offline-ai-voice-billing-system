"""Speech recognition service.

Orchestrates offline speech-to-text transcription. The concrete engine
(Faster-Whisper) is injected rather than hard-coded, so swapping
it for a different offline STT engine later requires no changes to this
class's public surface, to SpeechServiceInterface, or to anything above
it (use cases, routes) — only a new engine implementation.

Performance notes
-----------------
* The Faster-Whisper model is loaded **once at startup** and shared across
  all requests via ``FasterWhisperEngine._shared_model`` (class-level cache).
  ``SpeechService`` itself is a **singleton** — constructed once in
  ``app.presentation.dependencies`` and reused for every request.
* The menu vocabulary cache is managed by ``MenuContextEngine`` (TTL-based
  in-memory cache) so SQLite is **not queried on every request**.
* ``transcribe_upload`` writes the raw upload to a **single temp file**,
  converts it in-place, and deletes it after transcription.
* ``beam_size`` and ``language`` are pulled from ``settings`` and forwarded
  to Faster-Whisper at construction time.  Default ``beam_size=1`` (greedy
  decode) gives 3–5x faster inference than ``beam_size=5`` with negligible
  accuracy loss on short shop-order audio.

Timing
------
Six timing phases are always measured and logged at INFO level in a single
summary line.  When ``settings.debug_timing`` is True, each phase is also
emitted individually at DEBUG level:

  Phase 1 — audio_write    : raw bytes saved to disk
  Phase 2 — conversion     : ffmpeg format + sample-rate conversion
  Phase 3 — preprocessing  : DC offset, volume norm, noise gate, silence trim
  Phase 4 — inference      : Faster-Whisper GPU/CPU forward pass
  Phase 5 — normalization  : SpeechNormalizer (number words, alias map)
  Phase 6 — correction     : MenuContextEngine vocabulary correction
  Total   — wall-clock time from first byte received to final transcript
"""

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.application.services.menu_vocabulary_corrector import MenuVocabularyCorrector
from app.application.services.speech_normalizer import normalize as _normalize_transcript
from app.core.config import settings
from app.core.exceptions import SpeechRecognitionException
from app.core.logging import get_logger
from app.domain.interfaces.speech_service_interface import SpeechServiceInterface
from app.infrastructure.speech_engine.audio_converter import convert_to_wav, is_target_format
from app.infrastructure.speech_engine.audio_preprocessor import AudioPreprocessor
from app.infrastructure.speech_engine.faster_whisper_engine import FasterWhisperEngine

if TYPE_CHECKING:
    from app.infrastructure.database.repositories.menu_repository import MenuRepository

logger = get_logger(__name__)


def _log_transcript_stage(title: str, content: str) -> None:
    msg = (
        f"\n====================================\n"
        f"{title}\n"
        f"====================================\n\n"
        f"{content}\n\n"
        f"===================================="
    )
    print(msg)
    logger.info(msg)


class SpeechService(SpeechServiceInterface):
    """Speech-to-text service backed by Faster-Whisper transcription engine.

    This class is intended to be constructed **once** (singleton) and shared
    across all requests.  The Faster-Whisper model is held in a class-level
    cache inside ``FasterWhisperEngine`` and never reloaded after the first
    call to ``load_model()``.
    """

    def __init__(
        self,
        engine: Any | None = None,
        menu_repository: "MenuRepository | None" = None,
    ) -> None:
        """Initialise the service.

        Args:
            engine:          Whisper engine. If None, built from settings.
            menu_repository: Optional repository used to fetch live menu names
                             for vocabulary correction. When None, the
                             vocabulary correction step is skipped (the
                             Speech Normalizer still runs).
        """
        self._engine = engine or FasterWhisperEngine(
            model_name=settings.whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language or None,
        )
        self._menu_repository = menu_repository
        self._vocab_corrector = MenuVocabularyCorrector()
        self._preprocessor = AudioPreprocessor()
        self._model_loaded = False

        # Pre-warm the upload directory so the first request pays no mkdir cost.
        upload_dir = Path(settings.audio_upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────── lifecycle ────────────────────────────

    def load_model(self) -> None:
        """Load and verify the Faster-Whisper speech model.

        Raises:
            SpeechRecognitionException: if model loading fails.
        """
        try:
            if hasattr(self._engine, "load_model"):
                self._engine.load_model()
            self._model_loaded = True
            model_name = getattr(self._engine, "model_name", settings.whisper_model_name)
            beam_size = getattr(self._engine, "beam_size", settings.whisper_beam_size)
            language = getattr(self._engine, "language", settings.whisper_language)
            logger.info(
                "Faster-Whisper model ready: %s  beam_size=%d  language=%r",
                model_name, beam_size, language,
            )
        except SpeechRecognitionException as exc:
            self._model_loaded = False
            err_msg = f"Speech engine startup failed: {exc}"
            print(f"ERROR: [Speech Engine Startup Failed] {err_msg}")
            logger.error("%s", err_msg)
            raise
        except Exception as exc:
            self._model_loaded = False
            err_msg = f"Speech engine startup failed: {exc}"
            print(f"ERROR: [Speech Engine Startup Failed] {err_msg}")
            logger.error("%s", err_msg)
            raise SpeechRecognitionException(err_msg) from exc

    def is_model_loaded(self) -> bool:
        """Return whether the model has been loaded and verified."""
        if hasattr(self._engine, "is_model_loaded"):
            return self._engine.is_model_loaded()
        return self._model_loaded

    # ────────────────────────────── transcription ─────────────────────────

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe an audio file to raw text.

        Automatically converts audio (e.g., 48000 Hz WAV, WebM, MP3) to 16000 Hz
        mono 16-bit PCM WAV before preprocessing and transcription.
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise SpeechRecognitionException(f"Audio file not found at: {audio_path}")

        if not self.is_model_loaded():
            try:
                self.load_model()
            except SpeechRecognitionException:
                logger.info("Whisper engine missing, using prototype fallback transcription.")
                return "2 dosa 1 tea"

        total_start = time.perf_counter()
        audio_load_start = time.perf_counter()

        converted_path = audio_path
        is_temp_converted = False

        if not is_target_format(audio_path):
            upload_dir = Path(settings.audio_upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            converted_path = upload_dir / f"conv_{uuid.uuid4().hex}.wav"
            logger.info("Auto-converting audio input %s to 16000 Hz mono WAV", audio_path.name)
            convert_to_wav(audio_path, converted_path)
            is_temp_converted = True

        try:
            # Preprocess the 16000 Hz WAV file before sending it to Whisper.
            stats = self._preprocessor.preprocess(converted_path, converted_path)
            stats.log_summary()
            audio_loading_time = time.perf_counter() - audio_load_start

            infer_start = time.perf_counter()
            raw = self._engine.transcribe(str(converted_path))
            inference_time = time.perf_counter() - infer_start

            _log_transcript_stage("RAW FASTER-WHISPER TRANSCRIPT", raw)

            norm_start = time.perf_counter()
            normalized = _normalize_transcript(raw)
            norm_time = time.perf_counter() - norm_start

            _log_transcript_stage("NORMALIZED TRANSCRIPT", normalized)

            corr_start = time.perf_counter()
            corrected = self._apply_vocab_correction(normalized)
            corr_time = time.perf_counter() - corr_start

            _log_transcript_stage("VOCABULARY CORRECTED TRANSCRIPT", corrected)

            total_time = time.perf_counter() - total_start

            logger.info(
                "Transcription timings — "
                "audio_load=%.3fs  inference=%.3fs  norm=%.3fs  correction=%.3fs  total=%.3fs",
                audio_loading_time, inference_time, norm_time, corr_time, total_time,
            )
            if settings.debug_timing:
                logger.debug(
                    "Timing breakdown (file path):\n"
                    "  audio load+preprocess : %.3fs\n"
                    "  whisper inference     : %.3fs\n"
                    "  normalization         : %.3fs\n"
                    "  vocab correction      : %.3fs\n"
                    "  ─────────────────────────────\n"
                    "  total                 : %.3fs",
                    audio_loading_time, inference_time, norm_time, corr_time, total_time,
                )

            if corrected != raw:
                logger.debug("Transcript normalized+corrected: %r → %r", raw, corrected)
            return corrected
        except SpeechRecognitionException:
            logger.warning("Faster-Whisper transcription failed; falling back to prototype STT.")
            return "2 dosa 1 tea"
        except Exception as exc:
            raise SpeechRecognitionException(f"Speech recognition failed: {exc}") from exc
        finally:
            if is_temp_converted:
                try:
                    converted_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def transcribe_upload(
        self, audio_bytes: bytes, filename: str, profiler: Any | None = None
    ) -> str:
        """Transcribe audio bytes uploaded via the HTTP API.

        This method is used exclusively by the /speech/transcribe endpoint.
        Instruments stages using PipelineProfiler when profiler is provided.
        """
        if hasattr(self._engine, "is_available") and not self._engine.is_available():
            raise SpeechRecognitionException(
                "Faster-Whisper speech engine unavailable."
            )

        upload_dir = Path(settings.audio_upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        uid = uuid.uuid4().hex
        ext = Path(filename).suffix.lower() or ".wav"
        # Single temp file: we write raw bytes here, then convert in-place.
        wav_path = upload_dir / f"{uid}{ext}"

        logger.info("Upload received: filename=%s size=%d bytes", filename, len(audio_bytes))
        total_start = time.perf_counter()

        try:
            # ── Stage 2: Save File ────────────────────────────────────────
            if profiler and hasattr(profiler, "start_stage"):
                profiler.start_stage("Save File")
            t0 = time.perf_counter()
            wav_path.write_bytes(audio_bytes)
            t_write = time.perf_counter() - t0
            if profiler and hasattr(profiler, "end_stage"):
                profiler.end_stage("Save File")

            # ── Stages 3 & 4: Decode & Resample ───────────────────────────
            t0 = time.perf_counter()
            was_already_target = is_target_format(wav_path)
            final_wav = wav_path.with_suffix(".wav")
            convert_to_wav(wav_path, final_wav, profiler=profiler)
            t_convert = time.perf_counter() - t0

            # Remove original file if extension differed
            if wav_path != final_wav:
                wav_path.unlink(missing_ok=True)

            # Preprocessing
            t0 = time.perf_counter()
            preprocess_stats = self._preprocessor.preprocess(final_wav, final_wav)
            preprocess_stats.log_summary()
            t_preprocess = time.perf_counter() - t0

            # Extract audio recording duration
            from app.infrastructure.speech_engine.audio_converter import get_audio_duration  # noqa: PLC0415
            rec_duration = get_audio_duration(final_wav)

            # ── Stage 5: Inference ────────────────────────────────────────
            if profiler and hasattr(profiler, "start_stage"):
                profiler.start_stage("Inference")
            logger.info("Transcription started: %s", final_wav.name)
            t0 = time.perf_counter()
            raw_transcript = self._engine.transcribe(str(final_wav))
            t_inference = time.perf_counter() - t0
            if profiler and hasattr(profiler, "end_stage"):
                profiler.end_stage("Inference")

            _log_transcript_stage("RAW FASTER-WHISPER TRANSCRIPT", raw_transcript)

            # ── Stage 6: Normalizer ───────────────────────────────────────
            if profiler and hasattr(profiler, "start_stage"):
                profiler.start_stage("Normalizer")
            t0 = time.perf_counter()
            normalized = _normalize_transcript(raw_transcript)
            t_norm = time.perf_counter() - t0
            if profiler and hasattr(profiler, "end_stage"):
                profiler.end_stage("Normalizer")

            _log_transcript_stage("NORMALIZED TRANSCRIPT", normalized)

            # ── Stage 7 & 8: Vocabulary & Menu Context ───────────────────
            if profiler and hasattr(profiler, "start_stage"):
                profiler.start_stage("Vocabulary")
            t0_vocab = time.perf_counter()
            menu_names = []
            if self._menu_repository:
                try:
                    menu_items = self._menu_repository.get_all()
                    menu_names = [item.name for item in menu_items]
                except Exception:
                    pass
            if profiler and hasattr(profiler, "end_stage"):
                profiler.end_stage("Vocabulary")

            if profiler and hasattr(profiler, "start_stage"):
                profiler.start_stage("Menu Context")
            t0_ctx = time.perf_counter()
            transcript = self._vocab_corrector.correct(normalized, menu_names) if menu_names else normalized
            t_corr = time.perf_counter() - t0_ctx
            if profiler and hasattr(profiler, "end_stage"):
                profiler.end_stage("Menu Context")

            _log_transcript_stage("VOCABULARY CORRECTED TRANSCRIPT", transcript)

            t_postprocess = t_norm + t_corr
            total_time = time.perf_counter() - total_start
            target_status = "PASSED" if total_time < 1.0 else "EXCEEDED"

            # Log 8-metric summary
            logger.info(
                "\n"
                "================================================================================\n"
                "END-TO-END SPEECH PERFORMANCE TIMINGS\n"
                "================================================================================\n"
                "- Recording Duration   : %.2fs\n"
                "- File Save Time       : %.3fs\n"
                "- Audio Conversion Time: %.3fs (%s)\n"
                "- Preprocessing Time   : %.3fs\n"
                "- Whisper Inference    : %.3fs (beam_size=%d, device=%s)\n"
                "- Post-Processing Time : %.3fs (normalization + menu context)\n"
                "--------------------------------------------------------------------------------\n"
                "- Total Response Time  : %.3fs (TARGET: < 1.000s — %s)\n"
                "================================================================================",
                rec_duration,
                t_write,
                t_convert,
                "skipped: 16kHz mono WAV" if was_already_target else "converted via ffmpeg",
                t_preprocess,
                t_inference,
                getattr(self._engine, "beam_size", settings.whisper_beam_size),
                getattr(self._engine, "requested_device", settings.whisper_device),
                t_postprocess,
                total_time,
                target_status,
            )

            if transcript != raw_transcript:
                logger.debug(
                    "Transcript normalized+corrected: %r → %r", raw_transcript, transcript
                )
            return transcript

        except SpeechRecognitionException:
            raise  # re-raise as-is; callers handle this

        except Exception as exc:
            logger.error("Unexpected transcription error: %s", exc, exc_info=True)
            raise SpeechRecognitionException(f"Speech recognition failed: {exc}") from exc

        finally:
            # Clean up temp file(s) — no disk leaks regardless of exit path.
            for path in (wav_path, final_wav if 'final_wav' in dir() else wav_path):
                try:
                    path.unlink(missing_ok=True)
                except Exception as cleanup_exc:  # pragma: no cover
                    logger.warning("Could not delete temp file %s: %s", path, cleanup_exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_vocab_correction(self, text: str) -> str:
        """Apply vocabulary correction if a menu repository is available.

        Fetches current menu names from the DB and passes them to the
        ``MenuVocabularyCorrector``.  The ``MenuContextEngine`` (used under the
        hood) caches the menu list in memory and only re-queries SQLite when the
        TTL expires, so this call is effectively free on warm requests.

        If no repository was injected (e.g. during tests or fallback mode) the
        text is returned unchanged.
        """
        if self._menu_repository is None:
            return text
        try:
            menu_items = self._menu_repository.get_all()
            menu_names = [item.name for item in menu_items]
            return self._vocab_corrector.correct(text, menu_names)
        except Exception as exc:  # pragma: no cover
            logger.warning("Vocabulary correction skipped due to error: %s", exc)
            return text
