# khutbah-translate

Live captions for Friday khutbahs and Islamic lessons, on a small CPU-only box.

The speaker talks in English, Arabic, or (usually) both mixed together. The app:

1. splits the live audio into utterances with a **voice activity detector**;
2. transcribes each utterance with a **local speech-recognition model** that handles Arabic, English and code-switching;
3. recognises **Quran verses and hadith** being quoted, and shows them as cards with their established text and translations (Arabic, English and Chinese for the Quran) instead of an ad-hoc machine translation;
4. **translates** everything else into English and Chinese with an LLM (Arabic → English + Chinese, English → Chinese);
5. streams the result to a **TV caption page** and to **phones on the same network**, and keeps a **transcript** of every session.

Everything except the translation runs locally on the CPU. The target machine is an Intel N150 mini PC (4 cores, 16–32 GB RAM); development was done on an Apple M-series Mac.

```
audio source ──WS /ws/ingest──▶ VAD ─▶ STT (Qwen3-ASR + Omnilingual) + quote matching ─▶ batcher ─▶ LLM translation
                                                        │                                         │
                                  TV / phones ◀──WS /ws/view── events ──▶ SQLite log ──▶ transcript
```

---

## Contents

- [Quick start](#quick-start)
- [Running it for real](#running-it-for-real)
- [Configuration](#configuration)
- [API](#api)
- [How it works](#how-it-works)
- [Model choices and findings](#model-choices-and-findings)
- [Development and evaluation tools](#development-and-evaluation-tools)
- [Data sources and licences](#data-sources-and-licences)
- [Limitations and roadmap](#limitations-and-roadmap)

---

## Quick start

### 1. Prerequisites

| Tool | Why | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 3.12 + dependencies | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| ffmpeg | decoding audio files / microphones for the streaming client | `brew install ffmpeg` / `sudo apt install ffmpeg` |
| curl, tar, bzip2 | downloads in `setup.sh` | usually preinstalled |
| An OpenAI-compatible LLM API key | translation (optional: the app runs without it, showing the transcript only) | the default is [opencode Go](https://opencode.ai) |

No GPU, PyTorch or CUDA is needed. Speech models run through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on ONNX Runtime.

### 2. Install and download models and data

```bash
git clone https://github.com/Aelbannan/khutbah-translate.git
cd khutbah-translate
uv sync                    # creates .venv with all dependencies
./scripts/setup.sh         # ~1.2 GB download (models + Quran/hadith texts), then builds the databases
```

`setup.sh` is idempotent (re-running skips what already exists). It downloads:

- `models/silero_vad.onnx`: voice activity detection
- `models/sherpa-onnx-qwen3-asr-0.6B-int8-*`: main speech recogniser
- `models/sherpa-onnx-omnilingual-asr-*-300M-ctc-v2-int8-*`: second recogniser, used for spotting Quran/hadith quotes
- Tanzil Quran text (simple-clean, Uthmani) and Ma Jian's Chinese translation; the English (Saheeh International) is fetched from the quran.com API during the build
- nine hadith collections (Arabic + English) from fawazahmed0/hadith-api

and builds `data/quran.db` (~6,236 ayat) and `data/hadith.db` (~36,000 hadith, ~100 MB). The first hadith lookup builds and caches a search index (`data/hadith.index.npz`).

Use `./scripts/setup.sh --all-stt` to also fetch the other speech models compared during development (Moonshine, Whisper, Cohere Transcribe).

### 3. Check it

```bash
uv run pytest -q
```

(`tests/test_server.py::test_ingest_to_viewer` is skipped unless you have segmented a recording, see [tools](#development-and-evaluation-tools).)

### 4. Set the translation API key

```bash
export OPENCODE_GO_API_KEY=...        # never commit this; .env files are git-ignored
```

The key is read from `OPENCODE_GO_API_KEY`, falling back to `~/.pi/agent/auth.json` (`{"opencode-go": {"key": ...}}`) if you use the pi agent. To use any other OpenAI-compatible endpoint, set `KHUTBAH_LLM_BASE_URL` and `KHUTBAH_LLM_MODEL` (see [Configuration](#configuration)). To run without translation, set `KHUTBAH_TRANSLATE=0`.

### 5. Start the server and stream something

```bash
uv run khutbah-server                         # http://0.0.0.0:8000, loads models (~10 s)
```

In a second terminal, stream a recording in real time (any format ffmpeg reads):

```bash
uv run scripts/stream_audio.py path/to/khutbah.mp3 --title "Jumu'ah 12 Dec"
```

and open **http://localhost:8000** to watch the captions. The streaming client also prints every event to the terminal. `--speed 3` streams three times faster than real time for testing.

---

## Running it for real

### Audio sources

The server takes audio from exactly one source at a time over `WS /ws/ingest`. Ways to feed it:

| Source | Command |
|---|---|
| A microphone / USB audio interface on the server (macOS) | `uv run scripts/stream_audio.py --mic` |
| ... on Linux (PulseAudio/PipeWire default input) | `uv run scripts/stream_audio.py --mic` |
| ... on Linux, a specific ALSA device | `uv run scripts/stream_audio.py --ffmpeg-input "-f alsa -i hw:1"` |
| A network stream (e.g. the masjid's Icecast/RTMP/HLS feed) | `uv run scripts/stream_audio.py "https://example.org/live.m3u8"` |
| A browser's microphone | open `http://localhost:8000/ingest`, press Start |
| Your own client | send 16 kHz mono PCM16 frames to `/ws/ingest` (see [API](#api)) |

The best audio is a direct line feed from the masjid's mixing desk into a USB audio interface, rather than a microphone in the room.

Browsers only allow microphone access on `localhost` or over HTTPS, so the `/ingest` page works on the server itself but not from a phone over plain HTTP. Put the server behind an HTTPS reverse proxy (e.g. Caddy) if you need that.

### Viewers

- **TV**: open `http://<server-ip>:8000/?tv=1` in a full-screen browser (large text, no controls). Add `&lang=en` or `&lang=zh` for a single language.
- **Phones**: `http://<server-ip>:8000/` on the same Wi-Fi; each user picks English, 中文, or both. A phone that joins late gets the last 30 captions and cards.
- The page reconnects automatically if the server restarts.

What viewers see:

- **Captions**: translated passages (English and/or Chinese). Quotes appear inline as `⟨Qur'an 3:102⟩` / `⟨Sahih Muslim 2937⟩` markers.
- **Cards**: each recognised Quran passage with Uthmani Arabic, Saheeh International English and Ma Jian Chinese; each hadith with its Arabic, English and grading. A quote repeated within ten minutes does not produce a second card.
- **Now hearing**: the raw transcript of the latest utterance (hidden in TV mode).

### Transcripts

Every event of every session is stored in `data/sessions.db`.

- `GET /api/sessions`: list of sessions
- `GET /api/sessions/<id>/transcript.txt`: readable transcript (source text, English, Chinese, quotes)
- `GET /api/sessions/<id>`: all events as JSON

### Running on the mini PC (Linux)

```bash
# one-off
sudo apt install ffmpeg curl bzip2
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Aelbannan/khutbah-translate.git && cd khutbah-translate
uv sync && ./scripts/setup.sh
```

A systemd unit to start the server at boot (`/etc/systemd/system/khutbah.service`):

```ini
[Unit]
Description=khutbah-translate
After=network-online.target

[Service]
User=khutbah
WorkingDirectory=/home/khutbah/khutbah-translate
EnvironmentFile=/home/khutbah/khutbah-translate/.env   # OPENCODE_GO_API_KEY=..., KHUTBAH_LLM_MODEL=...
ExecStart=/home/khutbah/.local/bin/uv run khutbah-server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now khutbah
```

Start the audio source (e.g. `stream_audio.py --mic`) when the khutbah begins and stop it (Ctrl-C) afterwards; each run is one session.

Performance has so far been measured on an M3 Pro, not yet on the N150 (see [findings](#model-choices-and-findings)). If the N150 can't keep up with both recognisers, set `KHUTBAH_ALT_STT=` (empty) to run Qwen3-ASR only; you lose some quote detection on recited Arabic.

---

## Configuration

All settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `OPENCODE_GO_API_KEY` | from `~/.pi/agent/auth.json` | API key for the translation endpoint |
| `KHUTBAH_LLM_BASE_URL` | `https://opencode.ai/inference/go/openai/v1` | any OpenAI-compatible `/chat/completions` endpoint |
| `KHUTBAH_LLM_MODEL` | `qwen3.8-flash` | translation model (see [comparison](#translation-models)) |
| `KHUTBAH_TRANSLATE` | `1` | `0` disables translation (transcript and cards only) |
| `KHUTBAH_STT` | `qwen3` | main recogniser: `qwen3`, `omnilingual`, `moonshine`, `whisper-small`, `whisper-tiny`, `cohere` |
| `KHUTBAH_ALT_STT` | `omnilingual` | second recogniser for quote detection; empty to disable |
| `KHUTBAH_STT_THREADS` | `4` | CPU threads per recogniser |
| `KHUTBAH_HOST` / `KHUTBAH_PORT` | `0.0.0.0` / `8000` | where the server listens |

---

## API

### `WS /ws/ingest?title=<session title>`: audio in

- Send **binary frames of 16 kHz, mono, signed 16-bit little-endian PCM**, any size (100 ms = 3,200 bytes is a good choice).
- Send the text frame `{"type": "end"}` (or just close) to finish. The server processes the remaining audio and translations, then sends `{"type": "session", "state": "ended"}` and closes.
- The source receives the same events as viewers.
- Only one source at a time: a second connection is closed with code `4409`.

Minimal client (Python):

```python
import asyncio, json, subprocess, websockets

async def main():
    pcm = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", "talk.mp3",
                            "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], stdout=subprocess.PIPE)
    async with websockets.connect("ws://localhost:8000/ws/ingest?title=Test") as ws:
        async def recv():
            async for msg in ws:
                print(json.loads(msg))
        task = asyncio.create_task(recv())
        while chunk := pcm.stdout.read(3200):
            await ws.send(chunk)
            await asyncio.sleep(0.1)          # real-time pacing
        await ws.send(json.dumps({"type": "end"}))
        await task

asyncio.run(main())
```

### `WS /ws/view`: events out

JSON messages. On connect you first get a replay of the current session (`"replay": true`): its `session` event plus the last 30 passages and cards. Every event carries `"session": <id>`.

```jsonc
// session started / ended
{"type": "session", "state": "started", "session": 3, "title": "Jumu'ah 12 Dec"}

// an utterance was transcribed (seconds are relative to the start of the stream)
{"type": "segment", "id": 5, "start": 61.2, "end": 70.4, "lang": "ar",
 "text": "...", "alt_text": "...",          // main and second recogniser
 "marked": "... [QURAN 3:102] ...",          // text with quotes replaced by placeholders
 "stt_seconds": 1.9}

// a quotation was recognised ("new": false = shown recently, no new card needed)
{"type": "card", "kind": "quran", "ref": "3:102", "segment": 5, "new": true,
 "surah": 3, "surah_name": {"ar": "آل عمران", "simple": "Ali 'Imran", "en": "Family of Imran", "zh": "仪姆兰的家属"},
 "ayat": [{"ayah": 102, "ar": "يَٰٓأَيُّهَا ...", "en": "O you who have believed ...", "zh": "信道的人们啊！..."}]}
{"type": "card", "kind": "hadith", "ref": "Sahih Muslim 2937", "segment": 209, "new": true,
 "collection": "muslim", "citation": "Sahih Muslim 2937", "grade": "...",
 "ar": "...", "en": "...", "zh": null}

// a passage (several utterances) was translated
{"type": "passage", "id": 2, "segments": [4, 5, 6], "start": 52.0, "end": 80.1,
 "source": "...", "en": "...", "zh": "...", "model": "deepseek-v4.1-flash", "seconds": 1.8}
// (en/zh are null if translation is off or failed; failures include "error")

// pipeline health, after every utterance (not stored)
{"type": "status", "lag": 3.2, "queued": 0}     // seconds behind live, utterances waiting
```

### HTTP

| Route | |
|---|---|
| `GET /` | caption viewer (`?tv=1`, `?lang=en\|zh\|both`) |
| `GET /ingest` | browser microphone source |
| `GET /api/sessions` | `[{"id", "title", "started", "ended"}]` (Unix times) |
| `GET /api/sessions/{id}` | session plus all its events |
| `GET /api/sessions/{id}/transcript.txt` | plain-text transcript |

---

## How it works

### Code layout

```
src/khutbah/
  vad.py        Silero VAD (via sherpa-onnx) → utterance segments, with pre-roll and a 15 s cap
  stt.py        speech-recognition backends (sherpa-onnx offline recognisers) + output clean-up
  arabic.py     Arabic normalisation (strip diacritics, unify alef/ya/ta marbuta, drop hamza)
  matcher.py    shared fuzzy matching: character n-gram index + word-level local alignment
  quran.py      Quran span finder
  hadith.py     hadith span finder
  pipeline.py   QuoteFinder (quotes → cards + placeholders) and Batcher (utterances → passages)
  translate.py  LLM translation (OpenAI-compatible), prompt and per-model settings
  live.py       the live pipeline: threads for VAD, STT+matching, translation; emits events
  store.py      SQLite event log and transcript export
  server.py     FastAPI app: WebSockets, HTTP API, static pages
  web/          viewer (TV/phone) and browser-microphone pages, plain HTML/JS
scripts/        setup, database builders, streaming client, evaluation tools
tests/
```

### Pipeline

1. **VAD.** Silero VAD cuts the stream at pauses of at least 0.6 s, keeps 0.3 s of audio before each onset, and forces a cut at 15 s so recognition latency stays bounded.
2. **Speech recognition.** Each utterance goes to Qwen3-ASR (main transcript, punctuated, good at mixed English/Arabic) and optionally to Omnilingual ASR (more accurate on recited Arabic, used only for finding quotes). Clean-up removes stray CJK characters and collapses repetition loops.
3. **Quote finding** (`pipeline.QuoteFinder`). Each transcript is searched for Quran first, then hadith in the remaining words.
   - Matching spans in the text are replaced with placeholders like `[QURAN 3:102]` or `[HADITH Sahih Muslim 2937]`, so the translator leaves them alone and the viewer shows the card instead.
   - A quote found only in the Omnilingual transcript still produces a card, and the translator is told `[QURAN 2:201 quoted here]`.
4. **Batching.** Utterances are grouped into passages of roughly 18–45 words, ending at a sentence end or a pause over 2 s. Translating tiny VAD fragments on their own gives poor results.
5. **Translation.** Each passage goes to the LLM with the previous four passages as context. The model returns JSON `{"en": ..., "zh": ...}`.
6. **Events** are broadcast to viewers and appended to SQLite.

VAD, recognition + matching, and translation each run in their own thread, so a slow translation never holds up audio intake. Only one live session runs at a time; the models are loaded once at startup.

### Quran recognition (local, no API)

Speech recognition of recited Arabic is noisy, so matching is fuzzy at both the character and word level:

1. **Normalise** the Arabic: strip diacritics and tatweel, unify alef/ya/ta marbuta forms, drop hamza.
2. **Candidate retrieval.** A TF-IDF index over character 4-grams of every ayah (a sparse matrix in NumPy) picks the 12 most similar ayat, plus their neighbours (quotes often span ayat).
3. **Local alignment.** Smith-Waterman alignment over *words*, where two words match if their fuzzy similarity is at least 0.72 (rapidfuzz). This finds the quoted span *inside* a longer utterance, e.g. English speech with one ayah embedded, and tolerates misheard words.
4. **Rules against false positives:**
   - at least 4 matched words;
   - phrases that occur in many ayat (e.g. "رضي الله عنه") don't count;
   - common dhikr and du'a formulas that happen to be Quranic (e.g. "إنا لله وإنا إليه راجعون") don't count;
   - continuity: if the previous utterance ended at ayah *n*, a shorter match (3 words) continuing at *n+1* is accepted.

Matching takes 1–4 ms per utterance.

### Hadith recognition (local, no API)

Same retrieval and alignment, with differences because hadith texts share much stock wording ("the Messenger of Allah ﷺ said", narrator chains):

- **Distinctiveness, not length.** A match counts only if the IDF weights of the matched words add up to at least 25, with at least 4 words matched. Real quotes scored 31–87 in testing; stock phrases scored under 25.
- **Weekly formulas are ignored.** Khutbat al-Hajah, the closing du'a, kaffarat al-majlis and the salawat are technically hadith but recited every week. A match that is at least 80% formula, checked on both the heard words and the matched reference text, gets no card.
- **Quotes split across utterances.** Speakers often quote a hadith in pieces with explanation in between. Weaker partial matches (≥ 3 words, IDF weight ≥ 8) for the same hadith add up over 90 seconds. They count 1.5× if the speaker just named the collection ("Abu Dawud", "رواه البخاري", ...).
- **Same hadith in several books.** When several collections score within 8% of the best match, the better-known collection is cited: Nawawi's Forty, Qudsi, Bukhari, Muslim, Abu Dawud, Tirmidhi, Nasa'i, Ibn Majah, Muwatta.
- **One card per hadith.** Repeats within ten minutes keep the placeholder but get no new card. This applies to Quran cards too.

### Translation

`translate.py` calls an OpenAI-compatible `/chat/completions` endpoint directly (no agent framework, no MCP). The system prompt tells the model to:

- return JSON for the requested languages, cleaning up the speaker's English and translating the Arabic;
- **never invent** names, narrators, sources or details, and transliterate a garbled name rather than guess;
- keep `[QURAN …]` / `[HADITH …]` placeholders exactly and not re-translate quoted text;
- not translate the same content twice when the speaker recites Arabic and then translates it themselves;
- follow Ma Jian's conventions in Chinese: 真主 for Allah, （愿主福安之） for ﷺ, （愿主喜悦他）, （愿主赐他平安）.

Captions can't wait for chain-of-thought, so reasoning is switched off per model via `reasoning_effort` (`REASONING_EFFORT` in `translate.py`). For opencode Go, each session sends a stable `x-opencode-session` header and its own `User-Agent`, as their docs ask of clients.

---

## Model choices and findings

These results come from two real recordings: a 24-minute mostly-English khutbah with Arabic quotations (350 utterances) and an 8-minute dars on Hadith Jibril (62 utterances). The recordings are not included in the repo.

### Speech recognition

All models ran int8 on CPU through sherpa-onnx. RTF is the real-time factor (processing time ÷ audio time) on an M3 Pro with 4 threads.

| Model | RTF | Verdict |
|---|---|---|
| **Qwen3-ASR 0.6B** | ~0.12 | **Main recogniser.** Handles English/Arabic code-switching in one utterance and punctuates (328 of 350 utterances). Weaker on recited Quran. |
| **Omnilingual ASR 300M CTC** | ~0.06 | **Second opinion for quotes.** The most accurate on recited Arabic: it heard "على المنارة البيضاء شرقي دمشق" exactly where Qwen3 garbled it. Unusable as the main transcript: it spells English in mixed Arabic/Latin letters ("oنeستي oف بeiنg pرoفits") and has no punctuation. |
| Moonshine base (ar/en) + Whisper-tiny language ID | fast | Needs a language decision per utterance. When that decision is wrong it hallucinates, and it can't handle mixed utterances. |
| Cohere Transcribe | slow | Large, with the same language-ID problem. |
| Whisper small | — | Rejected: it *translates* English speech into Arabic text instead of transcribing it. |

Running both Qwen3 and Omnilingual costs about 1.5× Qwen3 alone. On the Mac the pipeline keeps up at 2–3× real time.

Still to try on the N150: OpenVINO (CPU and its 24-EU iGPU) and GGUF runtimes, which may be faster than ONNX Runtime there.

### Translation models

Six models through opencode Go translated all 149 passages of both recordings (894 requests, 0 errors), with reasoning off:

| Model | Median latency | Notes |
|---|---|---|
| **claude-haiku-5-5** | 2.7 s | Most faithful and most honest: marks unclear audio as `[unclear]` instead of guessing. Leaves some names in Latin script in the Chinese, and writes placeholders with fullwidth brackets (the viewer handles both). |
| **muse-spark-1.3-contributor** | 2.2 s | Good fidelity, but invented one detail. |
| **deepseek-v4.1-flash** | 1.8 s | Good and fast. |
| qwen3.8-flash | 1.8 s | Fast, but mistranslated a hard term and once translated a placeholder. |
| glm-5.3-flash | 3.2 s | Invented hadith content. |
| minimax-m3.1-flash | 3.2 s | Weakest: English leaking into Chinese, garbled names. |

The hard test was the word *fustat* ("encampment/stronghold") in a hadith about the Muslims' camp near Damascus. Only Haiku, DeepSeek and Muse got it right; the others wrote "best of the Muslims". **Recommendation:** `claude-haiku-5-5` for quality, or `deepseek-v4.1-flash` for speed. mimo-v2.6-flash was also tried, but it keeps reasoning even at the minimum setting and timed out.

---

## Development and evaluation tools

Offline tools used to choose models and tune the matchers. They work on a recording split into utterances. Recording names default to the author's test files, so pass your own.

```bash
ffmpeg -i talk.mp3 -ac 1 -ar 16000 data/audio/talk.wav       # 16 kHz mono WAV
uv run scripts/segment_file.py data/audio/talk.wav           # VAD → data/segments/talk/NNNN.wav + manifest.json
uv run scripts/transcribe_all.py qwen3 talk                  # → data/transcripts/talk/qwen3.json (cached)
uv run scripts/transcribe_all.py omnilingual talk
uv run scripts/scan_quotes.py qwen3 talk                     # Quran + hadith found in a transcript
uv run scripts/quran_hits.py qwen3 talk                      # Quran spans only, with alignment detail
uv run scripts/run_translate.py talk                         # dry run: cards and passages, no API calls
uv run scripts/run_translate.py talk claude-haiku-5-5 deepseek-v4.1-flash   # translate everything (costs API usage)
uv run scripts/smoke_stt.py qwen3 omnilingual                # side-by-side STT on a sample of utterances
uv run scripts/probe_reasoning.py <model>                    # which parameter disables reasoning for a model
uv run scripts/label_server.py                               # local UI (port 8765) to correct transcripts for evaluation
```

`run_translate.py` writes `data/runs/<recording>/<model>.jsonl` and `cards.json`. Models run concurrently, each with its own context.

Tests: `uv run pytest -q` covers Quran matching (synthetic noisy quotes, rejection of non-Quran and formulas), hadith matching (real noisy transcripts, stock phrases, split quotes, formulas), STT clean-up filters, and the server end to end with fake recognisers and a fake translator.

---

## Data sources and licences

Downloaded by `scripts/setup.sh`, not stored in this repository:

| Data | Source | Licence / terms |
|---|---|---|
| Quran Arabic text (simple-clean, Uthmani) | [Tanzil](https://tanzil.net) | CC BY 3.0; used unmodified, attribution kept in `data/quran.db` |
| Chinese Quran translation (Ma Jian) | Tanzil | as published by Tanzil |
| English Quran translation (Saheeh International) and surah names | [quran.com API](https://api-docs.quran.com) | per quran.com |
| Hadith Arabic and English (sunnah.com translations), gradings | [fawazahmed0/hadith-api](https://github.com/fawazahmed0/hadith-api) | per repository |
| Silero VAD, Qwen3-ASR, Omnilingual ASR and other speech models | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | per model (see each model's licence) |

Hadith cards have no Chinese yet. When added, it will be machine-translated from the English and labelled as such on screen.

---

## Limitations and roadmap

- **Not yet benchmarked on the N150.** Next: measure sherpa-onnx against OpenVINO (CPU and iGPU) and GGUF runtimes, and run Omnilingual only on mostly-Arabic utterances if CPU is tight.
- **Hadith cards in Chinese:** machine-translate and label them.
- **Placeholder robustness:** switch to neutral tokens (e.g. `⟦1⟧`) that models are less tempted to translate. Prompt the models to write names in Chinese characters in Chinese output.
- **Recognition:** Qwen3-ASR hotwords (names, Islamic terms), and a combined Qwen3 + Omnilingual transcript.
- **Labelled evaluation set** (via `label_server.py`) to measure recognition and translation quality instead of spot checks.
- **HTTPS** so phones can be audio sources, and a simple start/stop control for the operator.

---

## Licence

The code is MIT-licensed (see [LICENSE](LICENSE)). Downloaded data and models keep their own licences (see [Data sources and licences](#data-sources-and-licences)).
