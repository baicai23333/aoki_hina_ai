# Coqui Pretrained TTS (CPU) Setup Report

## Versions
- Coqui TTS: not checked (run `python -m pip show TTS` in the project venv to confirm)
- Python: not checked (run `python --version` in the project venv to confirm)

## Model Selection
- Source list: `tts_pretrained/reports/tts_list_models.txt`
- Selected model: `AOKI_TTS_MODEL_NAME` from `.env`
- Selection report: `tts_pretrained/reports/tts_model_selection.txt`

## Paths
- Model cache (TTS_HOME): `/home/mutsumi/aoki_hina_ai/.tts_models`
- WAV cache: `tts_cache/`

## Smoke Test
1. Run model selection:
   ```bash
   ./tts_pretrained/run_setup.sh
   ```
2. Enable TTS in `.env`:
   ```bash
   AOKI_TTS_ENABLED=1
   ```
3. Quick synthesis (inside venv):
   ```bash
   python - <<'PY'
   from tts_engine import TTSEngine
   engine = TTSEngine.get()
   path = engine.synthesize_to_file("こんにちは、青木陽菜です。")
   print(path)
   PY
   ```
4. Open Streamlit UI and click Play, or enable Auto TTS.

## Rollback
- Set `AOKI_TTS_ENABLED=0` in `.env` to disable TTS.
