#!/usr/bin/env python3
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "tts_pretrained" / "reports"
REPORT_FILE = REPORT_DIR / "tts_list_models.txt"
SELECTION_REPORT = REPORT_DIR / "tts_model_selection.txt"
ENV_FILE = PROJECT_ROOT / ".env"

MODEL_PREFIX = "tts_models/"


def load_model_lines():
    if not REPORT_FILE.exists():
        raise FileNotFoundError(f"Missing model list report: {REPORT_FILE}")
    lines = REPORT_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [line.strip() for line in lines if line.strip().startswith(MODEL_PREFIX)]


def is_xtts(name: str) -> bool:
    return "xtts" in name.lower()


def pick_japanese(models):
    japanese = []
    for name in models:
        lowered = name.lower()
        if is_xtts(name):
            continue
        if "/ja/" in lowered or "japanese" in lowered or "/jp/" in lowered:
            japanese.append(name)
    return japanese[0] if japanese else None


def pick_english_single_speaker(models):
    english = []
    for name in models:
        lowered = name.lower()
        if is_xtts(name):
            continue
        if "/en/" not in lowered:
            continue
        if any(token in lowered for token in ["ljspeech", "lj", "single", "vctk"]):
            english.append(name)
    if english:
        for name in english:
            if "vctk" not in name.lower():
                return name
        return english[0]
    for name in models:
        lowered = name.lower()
        if "/en/" in lowered and not is_xtts(name):
            return name
    return None


def update_env(model_name: str):
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated = False
    output_lines = []
    for line in lines:
        if line.strip().startswith("AOKI_TTS_MODEL_NAME="):
            output_lines.append(f"AOKI_TTS_MODEL_NAME={model_name}")
            updated = True
        else:
            output_lines.append(line)
    if not updated:
        output_lines.append(f"AOKI_TTS_MODEL_NAME={model_name}")

    ENV_FILE.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_model_lines()
    selected = pick_japanese(models)
    reason = "japanese model"
    if not selected:
        selected = pick_english_single_speaker(models)
        reason = "english single-speaker fallback"

    if not selected:
        raise RuntimeError("No suitable TTS model found in list.")

    update_env(selected)

    SELECTION_REPORT.write_text(
        f"Selected model: {selected}\nReason: {reason}\n",
        encoding="utf-8",
    )

    print(f"Selected model: {selected}")


if __name__ == "__main__":
    main()
