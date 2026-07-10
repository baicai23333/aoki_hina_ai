# Aoki Hina AI

Streamlit chat app that roleplays as Aoki Hina in Chinese, using DeepSeek for chat and Coqui TTS / XTTS for voice playback.

## Features

- Streamlit login and registration UI
- SQLite-backed user and chat history storage
- DeepSeek chat model through LangChain
- Few-shot examples for character tone
- Optional Coqui TTS playback
- XTTS voice reference setup and warmup scripts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `DEEPSEEK_API_KEY`.

## Run

```bash
streamlit run chat_client.py
```

## XTTS Setup

The project includes XTTS helper scripts and reference audio under `tts_xtts/`.

```bash
./tts_xtts/run_setup_xtts.sh
```

This prepares reference clips and warms up the cached speaker voice when supported.

## Notes

- `.env`, local chat databases, logs, virtual environments, model caches, and generated TTS cache files are ignored by Git.
- Do not commit real API keys or private chat history.
