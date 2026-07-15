#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TTS_HOME="${PROJECT_ROOT}/.tts_models"
PYTHON_BIN="${AOKI_PYTHON:-${PROJECT_ROOT}/venv311/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Project Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/tts_xtts/raw" "${PROJECT_ROOT}/tts_xtts/refs" "${PROJECT_ROOT}/tts_xtts/work" "${PROJECT_ROOT}/tts_xtts/reports" "${PROJECT_ROOT}/tts_xtts/scripts"

cp "${PROJECT_ROOT}/new.wav" "${PROJECT_ROOT}/tts_xtts/raw/new.wav"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tts_xtts/scripts/30_prepare_quality_ref.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tts_xtts/scripts/20_warmup_cache_voice.py"
