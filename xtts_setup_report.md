# XTTS (User Voice) Setup Report

## Model
- XTTS model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Backend switch: `AOKI_TTS_BACKEND=xtts`

## Paths
- TTS_HOME: `/home/mutsumi/aoki_hina_ai/.tts_models`
- Refs: `tts_xtts/refs/`
- Work: `tts_xtts/work/`
- Reports: `tts_xtts/reports/`

## Workflow
1. Prepare refs and warmup:
   ```bash
   ./tts_xtts/run_setup_xtts.sh
   ```
2. Enable XTTS:
   ```bash
   AOKI_TTS_BACKEND=xtts
   AOKI_TTS_ENABLED=1
   ```
3. Start the Streamlit UI and use Auto TTS or Play.

## Cached Voice
- `AOKI_XTTS_USE_CACHED_VOICE` is written by `20_warmup_cache_voice.py`.
- If cached voice is not supported, the engine will use `speaker_wav` on each request.

## Rollback
- Set `AOKI_TTS_ENABLED=0` or `AOKI_TTS_BACKEND=pretrained` in `.env`.
