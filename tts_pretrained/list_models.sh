#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TTS_HOME="${PROJECT_ROOT}/.tts_models"
REPORT_DIR="${PROJECT_ROOT}/tts_pretrained/reports"
REPORT_FILE="${REPORT_DIR}/tts_list_models.txt"

mkdir -p "${REPORT_DIR}"

echo "Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${REPORT_FILE}"
tts --list_models >> "${REPORT_FILE}"

echo "Saved to ${REPORT_FILE}"
