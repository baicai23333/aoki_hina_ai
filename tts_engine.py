import json
import hashlib
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


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
                gpt_sovits_output_language=load_env_var("GPT_SOVITS_OUTPUT_LANGUAGE", "zh"),
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
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.config.max_chars:
            raise ValueError(f"Text too long ({len(cleaned)} chars). Max {self.config.max_chars}.")

        cache_key = hashlib.sha256(
            f"{self.config.backend}|{self.config.model_name}|{self.config.speaker_name}|"
            f"{self.config.language}|{self.config.gpt_sovits_reference_audio}|"
            f"{self.config.gpt_sovits_reference_text}|{self.config.gpt_sovits_output_language}|"
            f"normalize={self.config.gpt_sovits_normalize_loudness}|"
            f"lufs={self.config.gpt_sovits_target_lufs}|{cleaned}".encode("utf-8")
        ).hexdigest()
        wav_path = self.config.cache_dir / f"{cache_key}.wav"
        if wav_path.exists():
            return wav_path

        if self.config.backend == "gpt_sovits":
            self._synthesize_gpt_sovits(cleaned, wav_path)
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
        if not self.config.gpt_sovits_reference_audio.exists():
            raise RuntimeError(f"GPT-SoVITS reference audio not found: {self.config.gpt_sovits_reference_audio}")
        if not self.config.gpt_sovits_reference_text.strip():
            raise RuntimeError("GPT_SOVITS_REFERENCE_TEXT is empty.")

        self._ensure_gpt_sovits_api()
        payload = {
            "text": text,
            "text_lang": self.config.gpt_sovits_output_language,
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

        raw_path = wav_path.with_suffix(".raw.wav")
        raw_path.write_bytes(response.content)
        if self.config.gpt_sovits_normalize_loudness:
            self._normalize_loudness(raw_path, wav_path)
            raw_path.unlink(missing_ok=True)
        else:
            raw_path.replace(wav_path)

    def _normalize_loudness(self, source_path: Path, output_path: Path) -> None:
        ffmpeg_path = self.config.gpt_sovits_root / "runtime" / "ffmpeg.exe"
        if not ffmpeg_path.exists():
            raise RuntimeError(f"FFmpeg not found: {ffmpeg_path}")

        normalized_path = output_path.with_suffix(".normalized.wav")
        loudnorm = f"loudnorm=I={self.config.gpt_sovits_target_lufs}:TP=-1.5:LRA=11"
        try:
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
            normalized_path.replace(output_path)
        except subprocess.CalledProcessError as exc:
            normalized_path.unlink(missing_ok=True)
            detail = exc.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"Audio loudness normalization failed: {detail}") from exc

    def _ensure_gpt_sovits_api(self) -> None:
        if self._gpt_sovits_api_reachable():
            return
        if not self.config.gpt_sovits_auto_start:
            raise RuntimeError("GPT-SoVITS API is not running on port 9880.")

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
        subprocess.Popen(
            [
                str(self.config.gpt_sovits_python),
                "api_v2.py",
                "-a", "127.0.0.1",
                "-p", "9880",
                "-c", str(self.config.gpt_sovits_config),
            ],
            cwd=str(self.config.gpt_sovits_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env=process_env,
        )

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self._gpt_sovits_api_reachable():
                return
            time.sleep(1)
        raise RuntimeError(f"GPT-SoVITS API did not start. Check {log_path}.")

    def _gpt_sovits_api_reachable(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 9880), timeout=0.5):
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
        return refs
