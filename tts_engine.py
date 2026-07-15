import json
import hashlib
import io
import ipaddress
import os
import re
import socket
import subprocess
import threading
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TTS_HOME = PROJECT_ROOT / ".tts_models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "tts_cache"
DEFAULT_CACHE_MAX_FILES = 256
DEFAULT_CACHE_MAX_BYTES = 512 * 1024 * 1024
MIN_WAV_FILE_SIZE = 44
WAV_VALIDATION_CHUNK_FRAMES = 16_384
WAV_CONTENT_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"})
JAPANESE_GPT_SOVITS_LANGUAGES = frozenset({"ja", "all_ja"})
CACHE_KEY_VERSION = "v5"
TEMP_WAV_SUFFIXES = (".raw.wav", ".normalized.wav")
_PROCESS_KEY_LOCKS_GUARD = threading.Lock()
_PROCESS_KEY_LOCKS = {}
_PROCESS_CACHE_MAINTENANCE_LOCK = threading.RLock()
_PROCESS_ACTIVE_CACHE_PATHS = set()


def _is_decodable_wav_reader(reader: wave.Wave_read) -> bool:
    channels = reader.getnchannels()
    sample_width = reader.getsampwidth()
    frame_rate = reader.getframerate()
    remaining_frames = reader.getnframes()
    if (
        reader.getcomptype() != "NONE"
        or channels <= 0
        or sample_width <= 0
        or frame_rate <= 0
        or remaining_frames <= 0
    ):
        return False

    bytes_per_frame = channels * sample_width
    while remaining_frames:
        requested_frames = min(remaining_frames, WAV_VALIDATION_CHUNK_FRAMES)
        frame_data = reader.readframes(requested_frames)
        if not frame_data or len(frame_data) % bytes_per_frame:
            return False
        decoded_frames = len(frame_data) // bytes_per_frame
        if decoded_frames > requested_frames:
            return False
        remaining_frames -= decoded_frames
    return True


def _is_valid_wav_bytes(data: bytes) -> bool:
    if not isinstance(data, bytes) or len(data) < MIN_WAV_FILE_SIZE:
        return False
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            return _is_decodable_wav_reader(reader)
    except (EOFError, OSError, OverflowError, ValueError, wave.Error):
        return False


def _is_valid_wav_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < MIN_WAV_FILE_SIZE:
            return False
        with wave.open(str(path), "rb") as reader:
            return _is_decodable_wav_reader(reader)
    except (EOFError, OSError, OverflowError, ValueError, wave.Error):
        return False


def safe_cached_wav_path(path, cache_dir=None, max_bytes=None) -> Path | None:
    """Return a verified cached WAV path, or None for unsafe/invalid paths."""
    try:
        configured_cache = cache_dir
        if configured_cache is None:
            configured_cache = load_env_var(
                "AOKI_TTS_CACHE_DIR", str(DEFAULT_CACHE_DIR)
            )
        configured_max_bytes = max_bytes
        if configured_max_bytes is None:
            configured_max_bytes = _positive_int_env(
                "AOKI_TTS_CACHE_MAX_BYTES", DEFAULT_CACHE_MAX_BYTES
            )
        if (
            isinstance(configured_max_bytes, bool)
            or not isinstance(configured_max_bytes, int)
            or configured_max_bytes <= 0
        ):
            return None
        cache_root = Path(configured_cache or DEFAULT_CACHE_DIR).resolve(strict=True)
        candidate = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    if candidate.suffix.lower() != ".wav":
        return None
    try:
        if not candidate.is_relative_to(cache_root):
            return None
    except (OSError, ValueError):
        return None
    try:
        if candidate.stat().st_size > configured_max_bytes:
            return None
    except OSError:
        return None
    return candidate if _is_valid_wav_file(candidate) else None


def _require_valid_wav_file(path: Path, label: str) -> None:
    if not _is_valid_wav_file(path):
        raise RuntimeError(f"{label} is not a valid WAV file.")


def _best_effort_unlink(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _file_fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return "missing-or-unreadable"
    return hasher.hexdigest()


def load_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is not None:
        return value
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    return default


def _positive_int_env(key: str, default: int) -> int:
    raw_value = load_env_var(key, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be a positive integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{key} must be a positive integer.")
    return value


@dataclass(frozen=True)
class TTSConfig:
    backend: str
    model_name: str
    tts_home: Path
    cache_dir: Path
    max_chars: int
    speaker_name: str
    language: str
    xtts_use_cached_voice: bool
    xtts_refs_manifest: Path
    gpt_sovits_api_url: str
    gpt_sovits_reference_audio: Path
    gpt_sovits_reference_text: str
    gpt_sovits_prompt_language: str
    gpt_sovits_output_language: str
    gpt_sovits_timeout: int
    gpt_sovits_auto_start: bool
    gpt_sovits_root: Path
    gpt_sovits_python: Path
    gpt_sovits_config: Path
    gpt_sovits_normalize_loudness: bool
    gpt_sovits_target_lufs: float
    cache_max_files: int = DEFAULT_CACHE_MAX_FILES
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES
    use_gpu: bool = False
    xtts_temperature: float = 0.75
    xtts_top_k: int = 50
    xtts_top_p: float = 0.85
    xtts_repetition_penalty: float = 5.0
    xtts_speed: float = 1.0


class TTSEngine:
    _instance = None

    def __init__(self, config: TTSConfig):
        self.config = config
        self._tts = None
        for field_name in ("cache_max_files", "cache_max_bytes"):
            value = getattr(self.config, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"{field_name} must be a positive integer.")
        if self.config.backend == "xtts":
            if self.config.language != "ja":
                raise RuntimeError("AOKI_XTTS_LANGUAGE must be 'ja' for Japanese TTS.")
            if not 0.1 <= self.config.xtts_temperature <= 1.0:
                raise RuntimeError("AOKI_XTTS_TEMPERATURE must be between 0.1 and 1.0.")
            if not 1 <= self.config.xtts_top_k <= 100:
                raise RuntimeError("AOKI_XTTS_TOP_K must be between 1 and 100.")
            if not 0.1 <= self.config.xtts_top_p <= 1.0:
                raise RuntimeError("AOKI_XTTS_TOP_P must be between 0.1 and 1.0.")
            if not 1.0 <= self.config.xtts_repetition_penalty <= 15.0:
                raise RuntimeError(
                    "AOKI_XTTS_REPETITION_PENALTY must be between 1.0 and 15.0."
                )
            if not 0.75 <= self.config.xtts_speed <= 1.25:
                raise RuntimeError("AOKI_XTTS_SPEED must be between 0.75 and 1.25.")
        self._gpt_sovits_output_language = (
            self.config.gpt_sovits_output_language or ""
        ).strip().lower()
        if (
            self.config.backend == "gpt_sovits"
            and self._gpt_sovits_output_language not in JAPANESE_GPT_SOVITS_LANGUAGES
        ):
            supported = ", ".join(sorted(JAPANESE_GPT_SOVITS_LANGUAGES))
            raise RuntimeError(
                "GPT_SOVITS_OUTPUT_LANGUAGE must be Japanese-compatible "
                f"for this playback path ({supported})."
            )
        self._key_locks_guard = _PROCESS_KEY_LOCKS_GUARD
        self._key_locks = _PROCESS_KEY_LOCKS
        self._cache_maintenance_lock = _PROCESS_CACHE_MAINTENANCE_LOCK
        self._active_cache_paths = _PROCESS_ACTIVE_CACHE_PATHS
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_temp_files()
        self._prune_cache()
        os.environ.setdefault("TTS_HOME", str(self.config.tts_home))

    @classmethod
    def get(cls) -> "TTSEngine":
        if cls._instance is None:
            backend = (load_env_var("AOKI_TTS_BACKEND", "gpt_sovits") or "gpt_sovits").strip().lower()
            if backend not in {"pretrained", "xtts", "gpt_sovits"}:
                raise RuntimeError("AOKI_TTS_BACKEND must be 'pretrained', 'xtts', or 'gpt_sovits'.")

            if backend == "pretrained":
                model_name = load_env_var("AOKI_TTS_MODEL_NAME")
                if not model_name:
                    raise RuntimeError("AOKI_TTS_MODEL_NAME is not set. Run tts_pretrained/run_setup.sh first.")
            elif backend == "xtts":
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            else:
                model_name = "gpt-sovits-v2proplus"

            tts_home = Path(load_env_var("TTS_HOME", str(DEFAULT_TTS_HOME)))
            cache_dir = Path(load_env_var("AOKI_TTS_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
            max_chars = int(load_env_var("AOKI_TTS_MAX_CHARS", "400"))
            speaker_name = load_env_var("AOKI_XTTS_SPEAKER_NAME", "MyVoice")
            language = load_env_var("AOKI_XTTS_LANGUAGE", "ja")
            xtts_use_cached_voice = (load_env_var("AOKI_XTTS_USE_CACHED_VOICE", "0") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            use_gpu = (load_env_var("AOKI_TTS_USE_GPU", "0") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            xtts_refs_manifest = Path(load_env_var(
                "AOKI_XTTS_REFS_MANIFEST",
                str(PROJECT_ROOT / "tts_xtts" / "refs" / "refs_manifest.json"),
            ))
            gpt_sovits_root = Path(load_env_var(
                "GPT_SOVITS_ROOT",
                str(PROJECT_ROOT / "GPT-SoVITS-package" / "GPT-SoVITS-v2pro-20250604-nvidia50"),
            ))
            gpt_sovits_api_url = load_env_var("GPT_SOVITS_API_URL", "http://127.0.0.1:9880/tts")
            gpt_sovits_reference_audio = Path(load_env_var(
                "GPT_SOVITS_REFERENCE_AUDIO",
                str(PROJECT_ROOT / "reference.wav"),
            ))
            gpt_sovits_reference_text = load_env_var(
                "GPT_SOVITS_REFERENCE_TEXT",
                "",
            )
            gpt_sovits_auto_start = (load_env_var("GPT_SOVITS_AUTO_START", "1") or "1").strip().lower() in {
                "1", "true", "yes", "on",
            }
            gpt_sovits_normalize_loudness = (
                load_env_var("GPT_SOVITS_NORMALIZE_LOUDNESS", "1") or "1"
            ).strip().lower() in {"1", "true", "yes", "on"}

            config = TTSConfig(
                backend=backend,
                model_name=model_name,
                tts_home=tts_home,
                cache_dir=cache_dir,
                max_chars=max_chars,
                speaker_name=speaker_name,
                language=language,
                xtts_use_cached_voice=xtts_use_cached_voice,
                xtts_refs_manifest=xtts_refs_manifest,
                gpt_sovits_api_url=gpt_sovits_api_url,
                gpt_sovits_reference_audio=gpt_sovits_reference_audio,
                gpt_sovits_reference_text=gpt_sovits_reference_text,
                gpt_sovits_prompt_language=load_env_var("GPT_SOVITS_PROMPT_LANGUAGE", "ja"),
                gpt_sovits_output_language=load_env_var("GPT_SOVITS_OUTPUT_LANGUAGE", "ja"),
                gpt_sovits_timeout=int(load_env_var("GPT_SOVITS_TIMEOUT", "180")),
                gpt_sovits_auto_start=gpt_sovits_auto_start,
                gpt_sovits_root=gpt_sovits_root,
                gpt_sovits_python=Path(load_env_var(
                    "GPT_SOVITS_PYTHON",
                    str(gpt_sovits_root / "runtime" / "python.exe"),
                )),
                gpt_sovits_config=Path(load_env_var(
                    "GPT_SOVITS_CONFIG",
                    str(PROJECT_ROOT / "gpt_sovits_tts_infer.yaml"),
                )),
                gpt_sovits_normalize_loudness=gpt_sovits_normalize_loudness,
                gpt_sovits_target_lufs=float(load_env_var("GPT_SOVITS_TARGET_LUFS", "-16")),
                cache_max_files=_positive_int_env(
                    "AOKI_TTS_CACHE_MAX_FILES", DEFAULT_CACHE_MAX_FILES
                ),
                cache_max_bytes=_positive_int_env(
                    "AOKI_TTS_CACHE_MAX_BYTES", DEFAULT_CACHE_MAX_BYTES
                ),
                use_gpu=use_gpu,
                xtts_temperature=float(load_env_var("AOKI_XTTS_TEMPERATURE", "0.75")),
                xtts_top_k=int(load_env_var("AOKI_XTTS_TOP_K", "50")),
                xtts_top_p=float(load_env_var("AOKI_XTTS_TOP_P", "0.85")),
                xtts_repetition_penalty=float(
                    load_env_var("AOKI_XTTS_REPETITION_PENALTY", "5.0")
                ),
                xtts_speed=float(load_env_var("AOKI_XTTS_SPEED", "1.0")),
            )
            cls._instance = TTSEngine(config)
        return cls._instance

    @staticmethod
    def _is_temporary_wav(path: Path) -> bool:
        return any(path.name.lower().endswith(suffix) for suffix in TEMP_WAV_SUFFIXES)

    def _cleanup_stale_temp_files(self) -> None:
        try:
            cache_entries = tuple(self.config.cache_dir.iterdir())
        except OSError:
            return
        for path in cache_entries:
            if self._is_temporary_wav(path):
                _best_effort_unlink(path)

    def _cache_entries(self) -> list[tuple[Path, int, int]]:
        entries = []
        try:
            cache_paths = tuple(self.config.cache_dir.iterdir())
        except OSError:
            return entries

        for path in cache_paths:
            if path.suffix.lower() != ".wav" or self._is_temporary_wav(path):
                continue
            try:
                if path.is_symlink():
                    _best_effort_unlink(path)
                    continue
                if not path.is_file():
                    continue
                stat_result = path.stat()
            except OSError:
                continue
            entries.append((path, stat_result.st_size, stat_result.st_mtime_ns))
        return entries

    def _prune_cache(
        self,
        *,
        reserve_files: int = 0,
        protected_paths=(),
    ) -> tuple[int, int]:
        file_limit = max(0, self.config.cache_max_files - reserve_files)
        with self._cache_maintenance_lock:
            protected = set(self._active_cache_paths)
            for path in protected_paths:
                protected.add(Path(path).resolve(strict=False))

            entries = self._cache_entries()
            file_count = len(entries)
            total_bytes = sum(size for _, size, _ in entries)
            for path, size, modified_ns in sorted(
                entries, key=lambda item: (item[2], item[0].name)
            ):
                if file_count <= file_limit and total_bytes <= self.config.cache_max_bytes:
                    break
                if path.resolve(strict=False) in protected:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue
                file_count -= 1
                total_bytes -= size

            remaining = self._cache_entries()
            return len(remaining), sum(size for _, size, _ in remaining)

    @contextmanager
    def _cache_key_lock(self, cache_key: str):
        scoped_key = (str(self.config.cache_dir.resolve(strict=False)), cache_key)
        with self._key_locks_guard:
            lock_record = self._key_locks.get(scoped_key)
            if lock_record is None:
                lock_record = [threading.Lock(), 0]
                self._key_locks[scoped_key] = lock_record
            lock_record[1] += 1
            key_lock = lock_record[0]

        key_lock.acquire()
        try:
            yield
        finally:
            key_lock.release()
            with self._key_locks_guard:
                lock_record[1] -= 1
                if lock_record[1] == 0:
                    self._key_locks.pop(scoped_key, None)

    @contextmanager
    def _active_cache_path(self, wav_path: Path):
        resolved_path = wav_path.resolve(strict=False)
        with self._cache_maintenance_lock:
            self._active_cache_paths.add(resolved_path)
        try:
            yield
        finally:
            with self._cache_maintenance_lock:
                self._active_cache_paths.discard(resolved_path)

    @staticmethod
    def _temporary_wav_path(wav_path: Path, stage: str) -> Path:
        return wav_path.with_name(f".{wav_path.stem}.{uuid4().hex}.{stage}.wav")

    def _cache_key(self, cleaned_text: str) -> str:
        xtts_ref_fingerprints = ()
        if self.config.backend == "xtts" and not self.config.xtts_use_cached_voice:
            xtts_ref_fingerprints = tuple(
                _file_fingerprint(Path(path)) for path in self._load_xtts_refs()
            )
        components = (
            CACHE_KEY_VERSION,
            self.config.backend,
            self.config.model_name,
            self.config.speaker_name,
            self.config.language,
            str(self.config.xtts_use_cached_voice),
            str(self.config.use_gpu),
            str(self.config.xtts_temperature),
            str(self.config.xtts_top_k),
            str(self.config.xtts_top_p),
            str(self.config.xtts_repetition_penalty),
            str(self.config.xtts_speed),
            str(self.config.xtts_refs_manifest.resolve(strict=False)),
            _file_fingerprint(self.config.xtts_refs_manifest),
            *xtts_ref_fingerprints,
            self.config.gpt_sovits_api_url,
            str(self.config.gpt_sovits_reference_audio.resolve(strict=False)),
            _file_fingerprint(self.config.gpt_sovits_reference_audio),
            self.config.gpt_sovits_reference_text,
            self.config.gpt_sovits_prompt_language,
            self._gpt_sovits_output_language,
            str(self.config.gpt_sovits_config.resolve(strict=False)),
            _file_fingerprint(self.config.gpt_sovits_config),
            str(self.config.gpt_sovits_normalize_loudness),
            str(self.config.gpt_sovits_target_lufs),
            cleaned_text,
        )
        return hashlib.sha256("\0".join(components).encode("utf-8")).hexdigest()

    def _load_model(self):
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise RuntimeError("Coqui TTS is not installed. Install TTS in the project venv.") from exc
            if self.config.use_gpu:
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError("PyTorch is not installed; GPU TTS cannot start.") from exc
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "AOKI_TTS_USE_GPU=1, but CUDA is unavailable in the project venv."
                    )
            self._tts = TTS(model_name=self.config.model_name, progress_bar=False)
            if self.config.use_gpu:
                self._tts.to("cuda")
        return self._tts

    def synthesize_to_file(self, text: str) -> Path:
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.config.max_chars:
            raise ValueError(f"Text too long ({len(cleaned)} chars). Max {self.config.max_chars}.")
        if self.config.backend == "xtts" and not re.search(r"[\u3040-\u30ff]", cleaned):
            raise ValueError("XTTS accepts only reviewed Japanese text containing kana.")

        cache_key = self._cache_key(cleaned)
        wav_path = self.config.cache_dir / f"{cache_key}.wav"

        with self._cache_key_lock(cache_key), self._active_cache_path(wav_path):
            cached_path = safe_cached_wav_path(
                wav_path,
                self.config.cache_dir,
                self.config.cache_max_bytes,
            )
            if cached_path is not None:
                try:
                    cached_path.touch(exist_ok=True)
                except OSError:
                    pass
                file_count, total_bytes = self._prune_cache(
                    protected_paths=(cached_path,)
                )
                if (
                    file_count <= self.config.cache_max_files
                    and total_bytes <= self.config.cache_max_bytes
                    and cached_path.stat().st_size <= self.config.cache_max_bytes
                ):
                    return cached_path
                _best_effort_unlink(cached_path)
                raise RuntimeError("TTS cache limits cannot retain the requested audio.")

            _best_effort_unlink(wav_path)
            file_count, total_bytes = self._prune_cache(reserve_files=1)
            if (
                file_count > max(0, self.config.cache_max_files - 1)
                or total_bytes > self.config.cache_max_bytes
            ):
                raise RuntimeError("TTS cache limits could not be enforced before synthesis.")

            staging_path = None
            try:
                if self.config.backend == "gpt_sovits":
                    self._synthesize_gpt_sovits(cleaned, wav_path)
                else:
                    staging_path = self._temporary_wav_path(wav_path, "raw")
                    tts = self._load_model()
                    if self.config.backend == "xtts":
                        speaker_kwargs = {}
                        if self.config.xtts_use_cached_voice:
                            speaker_kwargs["speaker_name"] = self.config.speaker_name
                        else:
                            speaker_kwargs["speaker_name"] = None
                            speaker_wav = self._load_xtts_refs()
                            speaker_kwargs["speaker_wav"] = speaker_wav
                        wav = tts.synthesizer.tts(
                            text=cleaned,
                            language_name=self.config.language,
                            # Split complete sentences into independent inference
                            # calls so one early EOS cannot discard later text.
                            # The model-level splitter additionally protects a
                            # single long Japanese sentence (71-char limit).
                            split_sentences=True,
                            enable_text_splitting=True,
                            temperature=self.config.xtts_temperature,
                            top_k=self.config.xtts_top_k,
                            top_p=self.config.xtts_top_p,
                            repetition_penalty=self.config.xtts_repetition_penalty,
                            speed=self.config.xtts_speed,
                            **speaker_kwargs,
                        )
                        tts.synthesizer.save_wav(wav=wav, path=str(staging_path))
                    else:
                        tts.tts_to_file(text=cleaned, file_path=str(staging_path))
                    _require_valid_wav_file(staging_path, "TTS output")
                    staging_path.replace(wav_path)

                _require_valid_wav_file(wav_path, "TTS output")
                if wav_path.stat().st_size > self.config.cache_max_bytes:
                    raise RuntimeError("Generated audio exceeds AOKI_TTS_CACHE_MAX_BYTES.")

                file_count, total_bytes = self._prune_cache(
                    protected_paths=(wav_path,)
                )
                if (
                    file_count > self.config.cache_max_files
                    or total_bytes > self.config.cache_max_bytes
                ):
                    raise RuntimeError("TTS cache limits could not be enforced after synthesis.")
                return wav_path
            except Exception:
                _best_effort_unlink(wav_path)
                self._prune_cache()
                raise
            finally:
                if staging_path is not None:
                    _best_effort_unlink(staging_path)

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"[*#>_]", "", cleaned)
        cleaned = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _synthesize_gpt_sovits(self, text: str, wav_path: Path) -> None:
        raw_path = self._temporary_wav_path(wav_path, "raw")
        normalized_path = self._temporary_wav_path(wav_path, "normalized")
        try:
            _best_effort_unlink(raw_path, normalized_path)
            if not self.config.gpt_sovits_reference_audio.exists():
                raise RuntimeError(
                    f"GPT-SoVITS reference audio not found: {self.config.gpt_sovits_reference_audio}"
                )
            if not self.config.gpt_sovits_reference_text.strip():
                raise RuntimeError("GPT_SOVITS_REFERENCE_TEXT is empty.")

            self._ensure_gpt_sovits_api()
            payload = {
                "text": text,
                "text_lang": self._gpt_sovits_output_language,
                "ref_audio_path": str(self.config.gpt_sovits_reference_audio.resolve()),
                "prompt_text": self.config.gpt_sovits_reference_text,
                "prompt_lang": self.config.gpt_sovits_prompt_language,
                "text_split_method": "cut5",
                "batch_size": 1,
                "speed_factor": 1.0,
                "seed": 12345,
                "media_type": "wav",
                "streaming_mode": False,
                "parallel_infer": True,
                "repetition_penalty": 1.35,
            }
            try:
                response = requests.post(
                    self.config.gpt_sovits_api_url,
                    json=payload,
                    timeout=(10, self.config.gpt_sovits_timeout),
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                error_response = getattr(exc, "response", None)
                detail = error_response.text[:500] if error_response is not None else ""
                raise RuntimeError(f"GPT-SoVITS request failed. {detail}") from exc

            raw_content_type = response.headers.get("Content-Type") or response.headers.get(
                "content-type", ""
            )
            content_type = raw_content_type.split(";", 1)[0].strip().lower()
            if content_type not in WAV_CONTENT_TYPES:
                raise RuntimeError(
                    "GPT-SoVITS returned an invalid content type: "
                    f"{content_type or 'missing'}."
                )
            response_body = response.content
            if not isinstance(response_body, bytes):
                raise RuntimeError("GPT-SoVITS returned an invalid WAV body.")
            if len(response_body) > self.config.cache_max_bytes:
                raise RuntimeError("GPT-SoVITS audio exceeds AOKI_TTS_CACHE_MAX_BYTES.")
            if not _is_valid_wav_bytes(response_body):
                raise RuntimeError("GPT-SoVITS returned an invalid WAV body.")

            raw_path.write_bytes(response_body)
            _require_valid_wav_file(raw_path, "GPT-SoVITS response")
            if self.config.gpt_sovits_normalize_loudness:
                self._normalize_loudness(raw_path, wav_path, normalized_path)
            else:
                raw_path.replace(wav_path)
            _require_valid_wav_file(wav_path, "GPT-SoVITS output")
        except Exception:
            _best_effort_unlink(wav_path)
            raise
        finally:
            _best_effort_unlink(raw_path, normalized_path)

    def _normalize_loudness(
        self,
        source_path: Path,
        output_path: Path,
        normalized_path: Path | None = None,
    ) -> None:
        normalized_path = normalized_path or self._temporary_wav_path(
            output_path, "normalized"
        )
        try:
            _best_effort_unlink(normalized_path)
            ffmpeg_path = self.config.gpt_sovits_root / "runtime" / "ffmpeg.exe"
            if not ffmpeg_path.exists():
                raise RuntimeError(f"FFmpeg not found: {ffmpeg_path}")

            loudnorm = f"loudnorm=I={self.config.gpt_sovits_target_lufs}:TP=-1.5:LRA=11"
            subprocess.run(
                [
                    str(ffmpeg_path), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(source_path),
                    "-af", loudnorm,
                    "-ar", "32000",
                    "-ac", "1",
                    str(normalized_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            _require_valid_wav_file(normalized_path, "Normalized audio")
            normalized_path.replace(output_path)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            detail = (
                stderr.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes)
                else str(stderr)
            )[-500:]
            raise RuntimeError(f"Audio loudness normalization failed: {detail}") from exc
        finally:
            _best_effort_unlink(normalized_path)

    def _ensure_gpt_sovits_api(self) -> None:
        scheme, host, port = self._gpt_sovits_api_endpoint()
        if self._gpt_sovits_api_reachable():
            return
        if not self.config.gpt_sovits_auto_start:
            raise RuntimeError(f"GPT-SoVITS API is not reachable at {host}:{port}.")
        if scheme != "http" or not self._is_loopback_host(host):
            raise RuntimeError(
                "GPT-SoVITS auto-start is allowed only for a local HTTP API URL."
            )

        required = (
            self.config.gpt_sovits_python,
            self.config.gpt_sovits_root / "api_v2.py",
            self.config.gpt_sovits_config,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"GPT-SoVITS auto-start files are missing: {', '.join(missing)}")

        log_path = PROJECT_ROOT / "gpt_sovits_api.log"
        log_file = log_path.open("ab")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process_env = os.environ.copy()
        process_env["PYTHONIOENCODING"] = "utf-8"
        process_env["PYTHONUTF8"] = "1"
        bind_host = "127.0.0.1" if host == "localhost" else host
        try:
            subprocess.Popen(
                [
                    str(self.config.gpt_sovits_python),
                    "api_v2.py",
                    "-a", bind_host,
                    "-p", str(port),
                    "-c", str(self.config.gpt_sovits_config),
                ],
                cwd=str(self.config.gpt_sovits_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                env=process_env,
            )
        finally:
            log_file.close()

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self._gpt_sovits_api_reachable():
                return
            time.sleep(1)
        raise RuntimeError(f"GPT-SoVITS API did not start. Check {log_path}.")

    def _gpt_sovits_api_endpoint(self) -> tuple[str, str, int]:
        parsed = urlparse(self.config.gpt_sovits_api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RuntimeError("GPT_SOVITS_API_URL must be a valid HTTP(S) URL.")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise RuntimeError("GPT_SOVITS_API_URL contains an invalid port.") from exc
        return parsed.scheme, parsed.hostname.lower(), port

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _gpt_sovits_api_reachable(self) -> bool:
        _, host, port = self._gpt_sovits_api_endpoint()
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _load_xtts_refs(self):
        if not self.config.xtts_refs_manifest.exists():
            raise RuntimeError(f"XTTS refs manifest not found: {self.config.xtts_refs_manifest}")
        data = json.loads(self.config.xtts_refs_manifest.read_text(encoding="utf-8"))
        refs = data.get("refs", [])
        if not refs:
            raise RuntimeError("XTTS refs manifest is empty.")
        manifest_dir = self.config.xtts_refs_manifest.resolve().parent
        forbidden_roots = (
            self.config.cache_dir.resolve(strict=False),
            DEFAULT_CACHE_DIR.resolve(strict=False),
            (PROJECT_ROOT / "tts_xtts" / "work").resolve(strict=False),
        )
        validated = []
        total_seconds = 0.0
        for raw_path in refs:
            ref_path = Path(raw_path)
            if not ref_path.is_absolute():
                ref_path = manifest_dir / ref_path
            ref_path = ref_path.resolve(strict=False)
            if any(ref_path == root or root in ref_path.parents for root in forbidden_roots):
                raise RuntimeError(f"XTTS reference cannot use cache/test output: {ref_path}")
            if not ref_path.is_file():
                raise RuntimeError(f"XTTS reference audio not found: {ref_path}")
            try:
                with wave.open(str(ref_path), "rb") as reader:
                    if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
                        raise RuntimeError(
                            f"XTTS reference must be lossless PCM 16-bit WAV: {ref_path}"
                        )
                    if reader.getnchannels() != 1 or reader.getframerate() != 22_050:
                        raise RuntimeError(
                            "XTTS reference must be mono 22050 Hz to avoid repeated "
                            f"resampling: {ref_path}"
                        )
                    duration = reader.getnframes() / reader.getframerate()
            except (EOFError, OSError, wave.Error) as exc:
                raise RuntimeError(f"XTTS reference is not a valid WAV: {ref_path}") from exc
            if not 3.0 <= duration <= 15.0:
                raise RuntimeError(
                    f"XTTS reference duration must be 3-15 seconds: {ref_path}"
                )
            total_seconds += duration
            validated.append(str(ref_path))
        if total_seconds > 20.0:
            raise RuntimeError("XTTS references must total no more than 20 seconds.")
        return validated
