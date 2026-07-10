#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TTS_HOME="${PROJECT_ROOT}/.tts_models"

mkdir -p "${PROJECT_ROOT}/tts_xtts/raw" "${PROJECT_ROOT}/tts_xtts/refs" "${PROJECT_ROOT}/tts_xtts/work" "${PROJECT_ROOT}/tts_xtts/reports" "${PROJECT_ROOT}/tts_xtts/scripts"

cp "${PROJECT_ROOT}/new.wav" "${PROJECT_ROOT}/tts_xtts/raw/new.wav"

"${PROJECT_ROOT}/tts_xtts/scripts/10_prepare_refs.py"
"${PROJECT_ROOT}/tts_xtts/scripts/20_warmup_cache_voice.py"
