# Aoki Hina AI

Streamlit chat app that roleplays as Aoki Hina in Chinese, using DeepSeek for chat and GPT-SoVITS for cloned-voice playback.

## Features

- Streamlit login and registration UI
- SQLite-backed user and chat history storage
- DeepSeek chat model through LangChain
- Few-shot examples for character tone
- GPT-SoVITS zero-shot voice cloning with a Japanese reference clip
- Automatic local GPT-SoVITS API startup and WAV caching
- Chinese and Japanese assistant output, with Japanese-only speech playback

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `DEEPSEEK_API_KEY`.

## Run

```powershell
streamlit run chat_client.py
```

## GPT-SoVITS Setup

Set `AOKI_TTS_ENABLED=1` and `AOKI_TTS_BACKEND=gpt_sovits` in `.env`.
Set `GPT_SOVITS_ROOT` to an installed GPT-SoVITS directory, then configure a local 3–10 second reference WAV and its exact transcript using `GPT_SOVITS_REFERENCE_AUDIO` and `GPT_SOVITS_REFERENCE_TEXT`.

With `GPT_SOVITS_AUTO_START=1`, the chat app starts the local API on port 9880 when speech is first requested. Model packages, reference recordings, generated speech, `.env`, logs, and chat databases are intentionally excluded from Git.

## Legacy XTTS Setup

The project includes XTTS helper scripts and reference audio under `tts_xtts/`.

```bash
./tts_xtts/run_setup_xtts.sh
```

This prepares reference clips and warms up the cached speaker voice when supported.

## Notes

- `.env`, local chat databases, logs, virtual environments, model caches, and generated TTS cache files are ignored by Git.
- Do not commit real API keys or private chat history.
