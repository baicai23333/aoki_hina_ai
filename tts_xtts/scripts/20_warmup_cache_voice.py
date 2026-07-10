#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tts_engine import load_env_var

REPORT_DIR = PROJECT_ROOT / "tts_xtts" / "reports"
WORK_DIR = PROJECT_ROOT / "tts_xtts" / "work"
REFS_MANIFEST = Path(load_env_var(
    "AOKI_XTTS_REFS_MANIFEST",
    str(PROJECT_ROOT / "tts_xtts" / "refs" / "refs_manifest.json"),
))

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_NAME = load_env_var("AOKI_XTTS_SPEAKER_NAME", "MyVoice")
LANGUAGE = load_env_var("AOKI_XTTS_LANGUAGE", "ja")


def update_env_flag(value: str):
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated = False
    output_lines = []
    for line in lines:
        if line.strip().startswith("AOKI_XTTS_USE_CACHED_VOICE="):
            output_lines.append(f"AOKI_XTTS_USE_CACHED_VOICE={value}")
            updated = True
        else:
            output_lines.append(line)
    if not updated:
        output_lines.append(f"AOKI_XTTS_USE_CACHED_VOICE={value}")

    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def load_refs():
    if not REFS_MANIFEST.exists():
        raise FileNotFoundError(f"Missing refs manifest: {REFS_MANIFEST}")
    data = json.loads(REFS_MANIFEST.read_text(encoding="utf-8"))
    refs = data.get("refs", [])
    if not refs:
        raise RuntimeError("Refs manifest has no refs.")
    return refs


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    os_tts_home = load_env_var("TTS_HOME", str(PROJECT_ROOT / ".tts_models"))
    os.environ.setdefault("TTS_HOME", os_tts_home)

    refs = load_refs()

    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError("Coqui TTS is not installed. Install TTS in the project venv.") from exc

    report_lines = []
    report_lines.append(f"Model: {XTTS_MODEL}")
    report_lines.append(f"Speaker: {SPEAKER_NAME}")
    report_lines.append(f"Language: {LANGUAGE}")
    report_lines.append(f"Refs: {len(refs)}")

    tts = TTS(model_name=XTTS_MODEL, progress_bar=False, gpu=False)

    # Precompute and cache the voice using refs so speaker_name becomes usable.
    gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(
        refs,
        max_ref_length=tts.synthesizer.tts_config.max_ref_len,
        gpt_cond_len=tts.synthesizer.tts_config.gpt_cond_len,
        gpt_cond_chunk_len=tts.synthesizer.tts_config.gpt_cond_chunk_len,
        sound_norm_refs=tts.synthesizer.tts_config.sound_norm_refs,
    )
    tts.synthesizer.tts_model.speaker_manager.speakers[SPEAKER_NAME] = {
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
    }

    warmup_path = WORK_DIR / "warmup.wav"
    tts.tts_to_file(
        text="Warm up voice cache.",
        file_path=str(warmup_path),
        speaker=SPEAKER_NAME,
        speaker_wav=refs,
        language=LANGUAGE,
    )
    report_lines.append(f"Warmup wav: {warmup_path}")

    cached_voice_ok = False
    cached_path = WORK_DIR / "cached_voice_test.wav"
    try:
        tts.tts_to_file(
            text="Cache voice test.",
            file_path=str(cached_path),
            speaker=SPEAKER_NAME,
            language=LANGUAGE,
        )
        cached_voice_ok = True
        report_lines.append("Cached voice inference: supported")
    except Exception as exc:
        report_lines.append(f"Cached voice inference: failed ({exc})")

    update_env_flag("1" if cached_voice_ok else "0")
    report_lines.append(f"AOKI_XTTS_USE_CACHED_VOICE={1 if cached_voice_ok else 0}")

    report_path = REPORT_DIR / "xtts_warmup_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
