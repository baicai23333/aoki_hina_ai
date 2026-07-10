import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TTS_HOME = PROJECT_ROOT / ".tts_models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "tts_cache"


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


class TTSEngine:
    _instance = None

    def __init__(self, config: TTSConfig):
        self.config = config
        self._tts = None
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TTS_HOME", str(self.config.tts_home))

    @classmethod
    def get(cls) -> "TTSEngine":
        if cls._instance is None:
            backend = (load_env_var("AOKI_TTS_BACKEND", "pretrained") or "pretrained").strip().lower()
            if backend not in {"pretrained", "xtts"}:
                raise RuntimeError("AOKI_TTS_BACKEND must be 'pretrained' or 'xtts'.")

            if backend == "pretrained":
                model_name = load_env_var("AOKI_TTS_MODEL_NAME")
                if not model_name:
                    raise RuntimeError("AOKI_TTS_MODEL_NAME is not set. Run tts_pretrained/run_setup.sh first.")
            else:
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

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
            xtts_refs_manifest = Path(load_env_var(
                "AOKI_XTTS_REFS_MANIFEST",
                str(PROJECT_ROOT / "tts_xtts" / "refs" / "refs_manifest.json"),
            ))

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
            )
            cls._instance = TTSEngine(config)
        return cls._instance

    def _load_model(self):
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise RuntimeError("Coqui TTS is not installed. Install TTS in the project venv.") from exc
            self._tts = TTS(model_name=self.config.model_name, progress_bar=False, gpu=False)
        return self._tts

    def synthesize_to_file(self, text: str) -> Path:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.config.max_chars:
            raise ValueError(f"Text too long ({len(cleaned)} chars). Max {self.config.max_chars}.")

        cache_key = hashlib.sha256(
            f"{self.config.backend}|{self.config.model_name}|{self.config.speaker_name}|"
            f"{self.config.language}|{cleaned}".encode("utf-8")
        ).hexdigest()
        wav_path = self.config.cache_dir / f"{cache_key}.wav"
        if wav_path.exists():
            return wav_path

        tts = self._load_model()
        if self.config.backend == "xtts":
            speaker_kwargs = {"language": self.config.language}
            if self.config.xtts_use_cached_voice:
                speaker_kwargs["speaker"] = self.config.speaker_name
            else:
                speaker_wav = self._load_xtts_refs()
                speaker_kwargs["speaker_wav"] = speaker_wav
            tts.tts_to_file(text=cleaned, file_path=str(wav_path), **speaker_kwargs)
        else:
            tts.tts_to_file(text=cleaned, file_path=str(wav_path))
        return wav_path

    def _load_xtts_refs(self):
        if not self.config.xtts_refs_manifest.exists():
            raise RuntimeError(f"XTTS refs manifest not found: {self.config.xtts_refs_manifest}")
        data = json.loads(self.config.xtts_refs_manifest.read_text(encoding="utf-8"))
        refs = data.get("refs", [])
        if not refs:
            raise RuntimeError("XTTS refs manifest is empty.")
        return refs
