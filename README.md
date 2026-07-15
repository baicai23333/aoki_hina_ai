# Aoki Hina AI

Streamlit chat app for a non-official fan-created AI character inspired by Aoki Hina's public materials. It uses DeepSeek for chat and GPT-SoVITS for optional local speech playback.

## Features

- Streamlit login and registration UI
- SQLite-backed user and chat history storage
- User-controlled structured memory with per-account isolation and hard delete
- DeepSeek chat model through LangChain
- A five-stage persona pipeline with deterministic safety routing
- A verified source registry with explicit quarantine states
- Granular public fact claims kept separate from style guidance and user history
- Optional GPT-SoVITS speech synthesis with a licensed or original Japanese reference clip
- Automatic local GPT-SoVITS API startup and WAV caching
- Chinese and strictly reviewed Japanese assistant output, with Japanese-only speech playback
- Message-ID-scoped translation and audio metadata, including safe legacy-history migration
- Optional allowlisted pipeline diagnostics that never include chat text or raw errors

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
streamlit run app.py
```

The default page remains the chat app. The protected management console is
available from the sidebar or at `/admin`.

## Management console

The management console provides privacy-first site statistics, user search,
translation and audio status, per-user maintenance actions, database health,
and an audit trail. Chat text is not loaded or displayed by default.

Create an Argon2 admin password hash, then add the printed value and a private
admin username to `.env`:

```powershell
.\.venv\Scripts\python.exe scripts\create_admin_password_hash.py
```

```text
AOKI_ADMIN_USERNAME=your_admin_name
AOKI_ADMIN_PASSWORD_HASH=the_generated_argon2_hash
```

Restart Streamlit after changing the admin configuration. To allow an
administrator to explicitly load recent chat text, set
`AOKI_ADMIN_ALLOW_MESSAGE_CONTENT=1`; every such access is written to the admin
audit log. Destructive actions require typed confirmation and never delete
cached audio files automatically. Password resets increment the account's
session version, so already-open chat sessions are asked to sign in again.
Deleting an account removes its chat and memory rows, while the audit log keeps
the target username and deletion counts for accountability.

For a public deployment, enforce request-rate limits for `/admin` at a trusted
reverse proxy or edge service. The built-in failed-login cooldown is scoped to
one browser session and is only a usability safeguard; Streamlit's reported
client IP is not used as a security boundary. Use a unique admin password of at
least 12 characters.

## Persona pipeline

The main response path is:

```text
user input
  -> deterministic scene classification
  -> select up to six relevant, user-saved memories for ordinary chat only
  -> verified source and evidence retrieval
  -> public fact question: deterministic rendering from verified claims
  -> other scenes: DeepSeek planning and styled generation
  -> strict DeepSeek review + deterministic identity/privacy checks
  -> temperature-zero translation + deterministic checks + strict review
  -> atomic message-ID-scoped history save
  -> validated Japanese only: optional TTS attached to that message ID
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

## Offline persona evaluation

The fixed suite contains 100 cases covering all seven routes, verified and unsupported facts, style-card retrieval, identity attacks, private probes, and quarantined evidence. It runs without a network connection or model API:

```powershell
.\.venv\Scripts\python.exe eval_persona.py
```

Use `--json` for a machine-readable report. The evaluator scores routing, fact retrieval, style retrieval, source isolation, and boundary actions independently, then exits with a non-zero status if any case fails. Unknown schema fields, empty style/source assertions, duplicate IDs, and references to missing facts, cards, or sources fail fast. Regenerate the reviewed JSONL after intentionally editing the case builder with:

```powershell
.\.venv\Scripts\python.exe scripts\build_persona_evaluation_cases.py
```

See `persona/EVALUATION.md` for the schema, distribution, and current scope. Final-response helpfulness, naturalness, and Japanese translation fidelity require a separate live-model evaluation and are not claimed by this offline score.

## User-controlled memory

Signed-in users can explicitly add, update, and permanently delete four kinds of structured memory: preferred name, interests, goals, and conversation preferences. The app does not infer or save memories from chat history automatically. Reads and mutations are always scoped to the current username, and each account can store at most 50 entries.

For ordinary chat, the pipeline selects at most six items. Preferred names and conversation preferences are prioritized; interests and goals are used only when they overlap with the current topic. Public-fact answers, identity attacks, and private-information probes ignore user memory completely. Selected memories are labeled as untrusted user context and cannot support real-person facts or override identity and privacy rules.

Memory is stored in the local, unencrypted `chat_history.db`. Selected entries are sent to DeepSeek with the current request. Deleting an entry prevents future use but does not erase existing chat records or TTS audio that may already contain related text. Do not store passwords, API keys, addresses, identity documents, medical or financial data, or another person's private information.

## Translation, history, and diagnostics

Ordinary Japanese translations use the final visible Chinese response as their only source, with a separate temperature-zero translator, deterministic identity/privacy/name/number checks, and a strict reviewer that can only accept or reject. Markdown strikethrough, hidden HTML, and redaction placeholders are rejected before translation so text hidden in Chinese cannot survive in Japanese or speech. A deterministic formatting or fidelity failure can trigger one constrained retranslation; the replacement is independently rechecked, while impersonation or added private information is rejected immediately without retry. Identity, private-information, and insufficient-evidence routes use canonical bilingual response pairs without a translation-model call, and a source mismatch fails closed instead of showing stale Japanese. If translation or review still fails, the app stores and shows the Chinese answer plus a fixed public status notice; rejected candidate text, internal issue codes, and raw model errors are never shown.

Each assistant message keeps its own immutable database ID, translation status, and audio path. Only `validated` and `fixed` Japanese can be synthesized or played. A historical Chinese message containing hidden or redacted source content is excluded from future model context and suppresses its Japanese text and audio even if older metadata marked them playable. Japanese text created before this migration is retained as `legacy_unverified` for reading, but it cannot be voiced until replaced by a reviewed translation. Audio playback accepts only valid WAV files inside the configured TTS cache. The TTS master switch also hides existing players, and autoplay/pending state is isolated per signed-in account.

Set `AOKI_DEBUG_UI=1` for an optional local sidebar trace. It exposes only allowlisted route names, evidence/fact/memory IDs, boundary and validation codes, stage statuses, and rounded timings. It does not contain user or assistant text, memory values, prompts, file paths, candidate translations, or raw exceptions, and it is not written to the chat database.

## GPT-SoVITS Setup

Set `AOKI_TTS_ENABLED=1` and `AOKI_TTS_BACKEND=gpt_sovits` in `.env`.
Set `GPT_SOVITS_ROOT` to an installed GPT-SoVITS directory, then configure a local 3–10 second licensed or original reference WAV and its exact transcript using `GPT_SOVITS_REFERENCE_AUDIO` and `GPT_SOVITS_REFERENCE_TEXT`. Do not use a real person's recording without permission or present synthesized speech as that person.

With `GPT_SOVITS_AUTO_START=1`, the chat app starts the configured local API when speech is first requested; non-local endpoints are never auto-started. The response must have an accepted WAV media type and a complete, decodable PCM WAV structure; invalid or partial output is deleted. Cache keys include the relevant voice/configuration inputs, same-text synthesis is locked, and the oldest cached files are pruned according to `AOKI_TTS_CACHE_MAX_FILES` and `AOKI_TTS_CACHE_MAX_BYTES`. Model packages, reference recordings, generated speech, `.env`, logs, and chat databases are intentionally excluded from Git.

## Legacy XTTS Setup

The project includes XTTS helper scripts and reference audio under `tts_xtts/`.

```bash
./tts_xtts/run_setup_xtts.sh
```

This prepares reference clips and warms up the cached speaker voice when supported.

## Notes

- `.env`, local chat databases, logs, virtual environments, model caches, and generated TTS cache files are ignored by Git.
- Do not commit real API keys or private chat history.
