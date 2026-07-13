# Aoki Hina AI

Streamlit chat app for a non-official fan-created AI character inspired by Aoki Hina's public materials. It uses DeepSeek for chat and GPT-SoVITS for optional local speech playback.

## Features

- Streamlit login and registration UI
- SQLite-backed user and chat history storage
- DeepSeek chat model through LangChain
- A five-stage persona pipeline with deterministic safety routing
- A verified source registry with explicit quarantine states
- Granular public fact claims kept separate from style guidance and user history
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

## Persona pipeline

The main response path is:

```text
user input
  -> deterministic scene classification
  -> verified source and evidence retrieval
  -> public fact question: deterministic rendering from verified claims
  -> other scenes: DeepSeek planning and styled generation
  -> strict DeepSeek review + deterministic identity/privacy checks
  -> translation and optional TTS
```

Persona materials live under `persona/`:

- `identity.md`, `tone.md`, `interaction_rules.md`, and `boundaries.md` define stable behavior.
- `source_registry.jsonl` records every source ID, URL, locator, verification status, and permitted use.
- `fact_claims.jsonl` contains granular public facts. A claim is active only when its citation points to a verified, fact-eligible source with a locator.
- `evidence_cards.jsonl` contains original Hina Bot interaction policies. These cards cannot carry external evidence or support real-person facts.
- `style_evidence_cards.jsonl` contains the imported 18-card public-expression pattern set. Only cards whose style-supporting references are verified enter prompts; the rest are quarantined at startup.
- `fewshot_dialogues.jsonl` contains reviewed conversation examples.
- `evaluation_cases.jsonl` is the fixed regression set for future prompt changes.
- `SOURCE_AUDIT.md` records the current audit snapshot and source-promotion process.
- `IMPORT_NOTES.md` records how the supplied Persona v1 files were normalized.

Style cards never support facts. Supported public fact questions bypass free-form generation and are rendered directly from verified claims; unsupported questions return an explicit “insufficient evidence” response without making an API request. Missing registries, malformed booleans, duplicate IDs, unknown source references, and incompatible citation types fail at startup.

Run the local tests without making an API request:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
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
