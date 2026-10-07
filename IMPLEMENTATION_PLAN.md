# MultiVoco Implementation Plan

Voice AI loan servicing agent. Python only, free-tier hosting, public demo link.
Source of the stack choices: `voice-ai-python-stack.pdf`.

This file is the shared reference for the two devices working on the repo, **Onion**
and **Garlic**. Read sections 1 to 4 before writing code. Section 7 says who does what,
and in what order.

| Device | Role | Workstreams |
|---|---|---|
| **Onion** | The real-time path: audio in, turn-taking, browser, hosting | Phase 0, O, A, D, H |
| **Garlic** | Language, content, and measurement: agent, speech out, data, evals | B, C, E, F, G |

---

## 1. What we are building

A browser-based voice agent that a borrower can talk to in English, Hindi (including
Hinglish), Kannada, or Bengali, with the language detected automatically. It can:

- verify the caller's identity
- answer questions about their loan (balance, next EMI, due date, payment history)
- take simple actions (payment promise, callback request)
- hand off to a human when it should
- be interrupted mid-sentence (barge-in) and recover cleanly

Around the agent, the parts that carry the project:

- a **latency waterfall** per turn, with p50/p95 time-to-first-audio
- an **eval harness** (WER, language ID accuracy, intent accuracy, task completion,
  interruption handling) with a test set per language
- **guardrails** (identity gate, tool-only figures, human handoff)

## 2. Decisions already made

| Topic | Decision | Notes |
|---|---|---|
| Pipeline | Hand-rolled loop: VAD → STT → LLM → TTS | Revisit Pipecat only after M2. Do not add it before then. |
| Server | FastAPI + WebSockets, asyncio, one process | |
| VAD | Silero VAD on CPU, in a thread pool | Must never run on the event loop |
| STT | Deepgram streaming as primary, Groq Whisper as fallback | Both behind one interface |
| LLM | Groq (Llama-class) with tool calling | Gemini free tier is the backup provider |
| TTS | Piper in the same container for English and Hindi. Azure Speech for Kannada and Bengali | Azure is a required provider, not only a backup. Both behind one interface |
| Languages | English, Hindi/Hinglish, Kannada, Bengali | Build order in section 2.1 |
| Language selection | Auto-detect by default, manual picker as fallback | Design in section 2.2 |
| Database | Postgres on Neon (free tier). SQLite for local dev and tests | Same SQLAlchemy models for both |
| Frontend | Plain JS, served as static files by FastAPI | One container, one public URL, no CORS |
| Dashboard | Plain JS page reading the metrics API | Same static mount |
| Hosting | Hugging Face Spaces (Docker) | |
| Python | 3.11 or newer (the repo pins 3.12), dependencies managed with `uv` | |

Open items, to be checked by whoever takes the workstream (quotas change often):

- Deepgram: current free credit, which model handles Hindi-English code-switching best,
  and whether any streaming model covers Kannada and Bengali
- Groq: current rate limits for the chosen model and for Whisper
- Whisper: measured WER on Kannada and Bengali. Expect it to be worse than Hindi.
- LLM: reply quality in Kannada and Bengali on Groq versus Gemini. Gemini may need to
  be the primary for these two.
- Piper: which Hindi voice is acceptable for Hinglish output, its sample rate, and
  whether usable Kannada or Bengali voices exist
- Azure Speech: free-tier character limit, Kannada (`kn-IN`) and Bengali (`bn-IN`)
  voices, and streaming latency from the Space's region
- Other Indic providers worth a comparison if the above fall short: Sarvam AI, AI4Bharat models
- Hugging Face Spaces: CPU/RAM on the free tier and the sleep policy

Record the findings in `docs/quotas.md` with the date checked.

### 2.1 Language build order

| Order | Language | Code | Why this position |
|---|---|---|---|
| 1 | English + Hindi/Hinglish | `en`, `hi` | Best provider support. Used to get the pipeline working end to end. |
| 2 | **Kannada (priority)** | `kn` | First additional language. Starts in parallel with the baseline, not after it. |
| 3 | Bengali | `bn` | Added once Kannada works, reusing the same routing and eval machinery. |

Kannada work is not deferred to the end: its dataset recording and provider checks
start on day one (Garlic, stage 0), and its STT, TTS, and agent work follow directly
after the first English/Hindi call, before the database, dashboard, and deploy
hardening. Bengali tasks are marked `[bn]` and start only after the matching Kannada
`[kn]` task is done.

Each language has one routing entry in `app/core/languages.py`. Starting point, to be
confirmed by measurement in workstreams A, B, and C:

| Language | STT | LLM | TTS |
|---|---|---|---|
| `en`, `hi` | Deepgram streaming | Groq | Piper |
| `kn` | Groq Whisper | Groq, or Gemini if quality is poor | Azure Speech |
| `bn` | Groq Whisper | Groq, or Gemini if quality is poor | Azure Speech |

Adding a fifth language later should need only: a routing entry, a script range, a
numeral map, prompt examples, and a test set. No pipeline changes.

### 2.2 Language auto-detection

1. The call page has a language picker that defaults to **Auto**. A manual choice
   skips detection and locks the session.
2. On Auto, the agent opens with a short neutral greeting. The caller's first utterance
   is buffered and sent to Whisper, which returns a detected language. The session
   locks to that language's providers.
3. On every later turn, the script of the final transcript is checked (Latin,
   Devanagari, Kannada, Bengali). If it disagrees with the session language for two
   turns in a row, the session switches.
4. If detection confidence is low, or the first utterance is too short, the agent asks
   which language the caller prefers and locks to the answer.
5. Hindi and English share one route, so code-switching between them never triggers a switch.

Known limits, to be stated in the write-up: the first turn is slower because detection
uses batch Whisper; one-word openings are often misdetected; a mid-call switch costs
up to two bad turns before it is caught.

## 3. Architecture

```
Browser                              FastAPI (one container)
┌──────────────────┐   WS /ws/call   ┌────────────────────────────────────────────┐
│ AudioWorklet mic │ ── PCM frames ─▶│ Session                                    │
│ Playback queue   │ ◀─ PCM + JSON ──│  ├─ VAD (thread pool)                      │
│ Transcript UI    │                 │  ├─ STT stream ──▶ Deepgram / Groq         │
└──────────────────┘                 │  ├─ Turn orchestrator (cancel on barge-in) │
                                     │  ├─ Agent: LLM + tools + guardrails ─▶ Groq│
┌──────────────────┐   GET /api/...  │  ├─ TTS (Piper, pool)                      │
│ Dashboard        │ ◀── JSON ───────│  └─ Trace recorder ──▶ Postgres            │
└──────────────────┘                 └────────────────────────────────────────────┘
```

One turn:

1. Client streams mic frames. VAD marks speech start and speech end.
2. Audio goes to STT while the user speaks. Speech end finalises the transcript.
3. The agent streams LLM tokens, running tools when asked.
4. Text is cut into sentences. Each sentence goes to TTS as soon as it is complete.
5. Audio chunks stream back to the client.
6. If VAD sees speech while the agent is talking: cancel the LLM and TTS tasks, tell the
   client to flush its playback queue, and record how much of the reply was actually heard.

## 4. Contracts (frozen in Phase 0)

These are what let workstreams run in parallel. They live in `app/core/` and
`docs/protocol.md`. **Changing a contract after Phase 0 needs agreement from every
workstream that uses it.** Add fields rather than renaming or removing them.

### 4.1 Component interfaces (`app/core/interfaces.py`)

The code is the source of truth. In outline:

| Interface | Built by | Used by | What it does |
|---|---|---|---|
| `VAD` | A | O | `process(frame)` returns `SPEECH_START`, `SPEECH_END`, or nothing |
| `STT` | A | O | `start(lang)`, then per utterance `send_audio(frame)`... and `finish()`. Every `finish()` yields exactly one final `Transcript` on `events()`. `close()` ends the stream. One instance per call and language. |
| `LanguageDetector` | A | O | `detect_audio(pcm)` on the first utterance, `detect_text(text)` script check on later turns. Both return a `LangGuess(lang, confidence)`. |
| `LLM` | B | B | `stream(messages, tools)` yields `TextDelta`, `ToolCall`, `Done` |
| `TTS` | C | O | `sample_rate(lang)` and `synthesize(text, lang)`, yielding PCM16 mono chunks. One call per sentence. |
| `Tool` | B | B | `spec` and `run(args, ctx)` |
| `Agent` | B | O | One per call. `greeting(ctx)`, `respond(user_text, ctx)` yielding `TextDelta`, `ToolResult`, `Handoff`, `Done`, and `commit_spoken(text, interrupted)`. |
| `TraceSink` | E | O | `call_started`, `turn_finished`, `call_ended`. Must return quickly and never raise. |

Two of these were added while building Phase 0, because the first draft had no contract
between the orchestrator and Garlic's streams:

- **`Agent`** is the only thing the orchestrator calls for a reply. It owns the history,
  prompts, tools, and guardrails, and replies in `ctx.language`. After each reply the
  orchestrator calls `commit_spoken` with the text the caller actually heard, which on
  a barge-in is shorter than what was generated. The agent stores that, not the full reply.
- **`TraceSink`** is how finished turns reach the database.

Rules for every implementation:

- Must stop promptly when its task is cancelled (`asyncio.CancelledError`), and release
  sockets and pool work in `finally`.
- Must not block the event loop. CPU work goes through `run_blocking` or `iter_blocking`
  in `app/core/pools.py`. Cancelling a task does not stop a pool thread, so keep each
  pool job to one frame or one audio chunk.
- Must not need an API key at import time.

How components are found (`app/core/registry.py`):

- Each workstream registers its implementations when its package is imported, for
  example `register("tts", "azure", lambda lang: AzureTTS())` or
  `register("agent", DEFAULT, lambda ctx: LoanAgent(ctx))`.
- The pipeline asks for them with `get_stt(lang)`, `get_llm(lang)`, `get_tts(lang)`,
  `get_vad()`, `get_language_detector()`, `get_agent(ctx)`, `get_trace_sink()`.
  Nothing else names a provider.
- `stt`, `llm`, and `tts` are chosen per language by the routing table in
  `app/core/languages.py`. To try a different provider for a language without editing
  that file, set `MULTIVOCO_ROUTE_OVERRIDES`, for example `{"kn": {"llm": "gemini"}}`.
- `MULTIVOCO_MOCK` picks which kinds are mocked: `all` (the default), `none`, or a list
  such as `llm,tts,agent`. This is what lets real components be switched on one at a
  time during integration.

Mocks (`app/core/mocks.py`): every interface has one, with no network, models, or keys.
Scripted audio is "tone is speech, zeros are silence" (`tone()`, `silence()`,
`frames()`). The mock STT, agent, and LLM return fixed lines in the right script for
each language. The mock TTS uses a different sample rate for Kannada and Bengali than
for English and Hindi, so the client's handling of a rate change is exercised.

`app/core/languages.py` holds the `Lang` enum (`en`, `hi`, `kn`, `bn`), the routing
table, the script ranges, the numeral maps, `lang_from_script(text)`, and
`to_ascii_digits(text)`.

### 4.2 WebSocket protocol (`docs/protocol.md`)

Endpoint: `/ws/call`

| Direction | Type | Payload |
|---|---|---|
| client → server | binary | PCM16 mono, 16 kHz, 512 samples per frame (32 ms, the Silero window) |
| client → server | JSON | `{"type":"start","lang":"auto"\|"en"\|"hi"\|"kn"\|"bn"}` |
| client → server | JSON | `{"type":"set_language","lang":...}` (manual override mid-call) |
| client → server | JSON | `{"type":"playback_started","turn_id":...,"t_client_ms":...}` |
| client → server | JSON | `{"type":"playback_position","turn_id":...,"ms_played":...}` (sent on flush) |
| client → server | JSON | `{"type":"end"}` |
| server → client | JSON | `{"type":"ready","session_id":...}` |
| server → client | JSON | `{"type":"transcript","role":"user"\|"agent","text":...,"final":bool,"turn_id":...}` |
| server → client | JSON | `{"type":"audio_start","turn_id":...,"sample_rate":...}` |
| server → client | binary | PCM16 mono at the announced sample rate |
| server → client | JSON | `{"type":"audio_end","turn_id":...}` |
| server → client | JSON | `{"type":"flush","turn_id":...}` (barge-in: drop queued audio now) |
| server → client | JSON | `{"type":"state","value":"listening"\|"thinking"\|"speaking"\|"handoff"}` |
| server → client | JSON | `{"type":"language","lang":...,"source":"manual"\|"audio"\|"script"\|"asked","confidence":...}` |
| server → client | JSON | `{"type":"error","code":...,"message":...}` |

### 4.3 Latency trace (`app/core/trace.py`)

One record per turn. All times are server monotonic milliseconds from session start,
except the client one.

| Mark | Meaning |
|---|---|
| `speech_end` | VAD declares end of user speech |
| `lang_detected` | language detection finished (only on turns where it ran) |
| `stt_final` | final transcript received |
| `llm_first_token` | first text delta |
| `llm_first_sentence` | first sentence handed to TTS |
| `tts_first_chunk` | first audio chunk produced |
| `audio_sent` | first audio chunk written to the socket |
| `playback_started` | client reports first audio played (client clock, stored separately) |

Headline metric: **time to first audio = `audio_sent` − `speech_end`**. Also store
`interrupted: bool`, `ms_played`, tool calls with durations, provider names, the
server region, the turn's `language`, and `language_switched: bool`. Latency is always
reported per language, since the routes use different providers.

### 4.4 Database schema (`app/db/models.py`)

- `customers` (id, name, phone_last4, dob, language)
- `loans` (id, customer_id, principal, outstanding, emi_amount, next_due_date, status)
- `payments` (id, loan_id, amount, paid_on, status)
- `calls` (id, started_at, ended_at, verified_customer_id, outcome, handoff_reason,
  requested_lang, final_lang)
- `turns` (id, call_id, idx, language, user_text, agent_text, interrupted, ms_played, trace JSON)
- `tool_calls` (id, turn_id, name, args JSON, result JSON, duration_ms)

All loan data is synthetic. Seed script: `scripts/seed.py`. Seed customers cover all
four languages, with names typical of each.

## 5. Repository layout and ownership

Each path has one owning workstream, and each workstream has one device. Onion owns
O, A, D, H and `app/core/`. Garlic owns B, C, E, F, G. Stay inside your paths. If you
need a change somewhere else, ask the other device.

```
app/
  core/          Phase 0, then frozen   interfaces, mocks, registry, protocol, trace, pools,
                                        config, languages
  main.py        O                      FastAPI app, routes, static mounts
  pipeline/      O                      session, turn orchestrator, sentence splitter
  audio/         A                      Silero VAD wrapper, frame buffering
  stt/           A                      Deepgram and Groq Whisper clients
  langid/        A                      language detector (audio and script)
  agent/         B                      prompt, LLM client, tool loop, guardrails
  tools/         B                      loan tools
  tts/           C                      Piper wrapper, Azure Speech client
  db/            E                      models, session, repositories
  metrics/       E                      aggregation, /api/metrics routes
web/
  client/        D                      call page, AudioWorklet, playback
  dashboard/     F                      latency dashboard
evals/           G                      datasets, runners, reports
scripts/         E (seed), H (others)
deploy/          H                      Dockerfile, Space config, voice download
docs/            shared                 one file per topic, owner named at the top
tests/           mirrors app/           each workstream owns its matching folder
```

## 6. Workstreams

### Phase 0: Foundation (Onion; no other code starts until it lands)

While Onion does this, Garlic does the work that needs no code: provider and quota
checks for `docs/quotas.md`, confirming native speakers, writing the eval scenarios,
and starting the Hinglish and Kannada recordings. Garlic reviews the contracts before
they are frozen, since it builds five streams against them.

- [x] Project skeleton: `pyproject.toml`, `uv` lockfile, ruff, pytest, `.env.example`, `.gitignore`
- [x] `app/core/interfaces.py`, `mocks.py`, `trace.py`, `pools.py`, `config.py`
- [x] `app/core/languages.py`: `Lang` enum, routing table, script ranges and numeral
      maps for all four languages
- [x] `app/core/registry.py`: `register(...)` and the `get_*` accessors, with per-kind mocking
- [x] `app/core/protocol.py`: typed models for every JSON message
- [x] `docs/protocol.md` (section 4.2 written out in full, with sequences and examples)
- [x] `app/db/models.py` with the tables from 4.4, running on SQLite
- [x] `app/main.py` with `/healthz` and a WebSocket echo endpoint at `/ws/echo`
- [x] A mock end-to-end test (`tests/core/test_mock_e2e.py`): scripted audio in →
      mock VAD/STT/agent/TTS → audio out, with a full trace
- [x] The same mock test run once per language, plus `lang: "auto"` detecting Hindi,
      Kannada, and Bengali
- [ ] **Garlic reviews the contracts.** Until this is ticked, Onion can still change
      `app/core/` freely on request. After it, the freeze in section 8 applies.

Done when: `pytest` passes on a fresh clone with no API keys set.

---

After Phase 0, Onion's streams and Garlic's streams run **in parallel** with each
other. Within a device, follow the order in section 7.

### O: Orchestrator (Onion, critical path)

Owns `app/pipeline/`, `app/main.py`. Depends on: Phase 0 only (builds against mocks).

- [ ] Session lifecycle: connect, `start`, teardown, cleanup of all tasks on disconnect
- [ ] Turn state machine: listening → thinking → speaking → listening
- [ ] Sentence splitter for streamed LLM text (handle the danda `।` used in Hindi and
      Bengali, abbreviations, numbers; Kannada uses Latin punctuation)
- [ ] Session language state: manual lock, first-utterance detection, script-based
      switch after two disagreeing turns, and the "which language?" fallback (section 2.2)
- [ ] First turn on Auto: buffer the utterance, detect, then start the right STT. Later
      turns stream straight to the session's STT.
- [ ] Language switch mid-call: swap STT/LLM/TTS providers between turns, never during
      one, and keep the conversation history
- [ ] LLM → TTS → socket pipeline with bounded queues (backpressure)
- [ ] Barge-in: on `SPEECH_START` while speaking, cancel LLM and TTS tasks, send `flush`,
      wait for `playback_position`, truncate the agent message in history to what was heard
- [ ] False-interrupt handling: ignore speech shorter than a threshold, and backchannels
- [ ] Trace marks from 4.3 emitted at each stage
- [ ] Per-session limits: max call length, max turns, idle timeout

Done when: with mocks, an interrupt stops outgoing audio within 100 ms of the VAD event
in tests, no task is left running after disconnect, every turn produces a full trace,
and scripted tests cover detection, a mid-call switch, and the low-confidence fallback.

### A: Audio in (VAD + STT + language detection) (Onion)

Owns `app/audio/`, `app/stt/`, `app/langid/`. Depends on: Phase 0.

- [ ] Silero VAD wrapper running in the executor, with tunable thresholds and end-of-speech silence
- [ ] Deepgram streaming client: partials, finals, reconnect, keep-alive
- [ ] Groq Whisper fallback: buffer the utterance, transcribe on speech end
- [ ] Provider switch via config, plus automatic fallback on error
- [ ] Hinglish settings: language/model choice, tested on at least 20 recorded clips
- [ ] `[kn]` Kannada STT: compare every available provider on at least 20 clips, pick
      one, set the routing entry. Check Kannada-English code-switching too.
- [ ] `[bn]` Bengali STT: same comparison and routing entry
- [ ] Language detector, audio: Whisper-based detection on the first utterance, with a
      confidence value and a minimum-duration rule
- [ ] Language detector, text: script check using the ranges in `app/core/languages.py`,
      treating Latin-script text as "no evidence" rather than English
- [ ] `scripts/stt_file.py`: transcribe a WAV from the command line, with `--lang`
      (workstream G uses this)

Done when: both providers pass the interface tests, a note in `docs/stt.md` compares
their latency and WER per language on the sample clips, and the detector passes the
interface tests for all four languages.

### B: Agent (LLM, tools, guardrails) (Garlic)

Owns `app/agent/`, `app/tools/`. Depends on: Phase 0. Fully testable in text, no audio needed.

- [ ] Groq streaming client with tool calling, retries, timeout, Gemini as backup
- [ ] Tools: `verify_identity`, `get_loan_summary`, `get_next_emi`, `get_payment_history`,
      `record_payment_promise`, `request_callback`, `handoff_to_human`
- [ ] System prompt: short spoken replies, mirrors the caller's language, speaks amounts and
      dates in a TTS-friendly way
- [ ] Guardrail 1, identity gate: account tools refuse to run until `verify_identity`
      succeeds. Enforced in code, not only in the prompt. Lock after 3 failed attempts.
- [ ] Guardrail 2, tool-only figures: every number in a reply must trace to a tool result
      in this call. On violation, regenerate once, then fall back to a safe reply.
      The agent holds each sentence until it has passed this check, and only then yields
      it as a `TextDelta`: the orchestrator speaks deltas as they arrive, so a figure
      cannot be taken back once it is sent. Replies therefore reach the orchestrator one
      checked sentence at a time, not token by token (section 11, note 1).
- [ ] Guardrail 3, handoff: explicit request, repeated misunderstanding, distress or
      dispute, out-of-scope requests
- [ ] Reply language follows the session language passed in by the orchestrator, not
      the model's own guess. Tool names, arguments, and results stay in English.
- [ ] `[kn]` Kannada: prompt examples, reply quality check on Groq versus Gemini, set
      the routing entry. Guardrail 2 must recognise Kannada numerals (೦-೯) and numbers
      written out as Kannada words.
- [ ] `[bn]` Bengali: the same, with Bengali numerals (০-৯) and number words
- [ ] Identity verification works when the caller reads digits or a date of birth in
      any of the four languages. `verify_identity` finds the customer by `phone_last4`
      plus `dob`, the only identifying fields a caller can say (section 11, note 3).
- [ ] `scripts/chat.py`: text REPL against the agent, with `--lang` (workstream G uses this)

Done when: unit tests cover each guardrail in every language, including attempts to
get account data before verification and attempts to get the model to invent a figure.

### C: TTS (Garlic)

Owns `app/tts/`. Depends on: Phase 0.

- [ ] Piper wrapper in the executor, streaming PCM chunks, cancellable mid-sentence
- [ ] Voice selection for English and Hindi, with the model download handled at build time
- [ ] Text normalisation: rupee amounts, dates, loan IDs, digits read one at a time
- [ ] Warm-up at startup so the first turn is not slow
- [ ] Loaded Piper voices and the Azure client are cached once per process inside
      `app/tts/`. The registered factory only hands out the cached instance, because
      the registry runs the factory on every `get_tts` call (section 11, note 2).
- [ ] `[kn]` Azure Speech client with streaming output and cancellation, Kannada voice
      chosen with a native speaker. **Start this straight after the Piper baseline**:
      it is the only TTS route for Kannada.
- [ ] `[kn]` Kannada text normalisation: rupee amounts, dates, digits, English words
      inside Kannada sentences
- [ ] `[bn]` Bengali voice and text normalisation on the same Azure client
- [ ] Azure as a fallback for English and Hindi if Piper fails or is too slow
- [ ] Character-usage counter for Azure, so the free tier is not exhausted mid-demo
- [ ] Benchmark per language: time to first chunk and real-time factor on 2 vCPUs for
      Piper, network time to first chunk for Azure, in `docs/tts.md`

Done when: first chunk for a 10-word sentence arrives fast enough for the latency
target on Space-sized hardware in every language, and cancellation stops synthesis
within one chunk on both providers.

### D: Browser client (Onion)

Owns `web/client/`. Depends on: Phase 0 (works against the echo endpoint and mock pipeline).

- [ ] AudioWorklet capture, resample to 16 kHz PCM16, 512-sample frames
- [ ] Playback queue that handles the announced sample rate, with gapless scheduling
- [ ] `flush` handling: stop audio immediately and report `playback_position`
- [ ] `playback_started` reporting for the client-side latency mark
- [ ] UI: call button, state indicator, live transcript, error and mic-permission states
- [ ] Language picker (Auto, English, Hindi, Kannada, Bengali), a badge showing the
      detected language from the `language` message, and a mid-call override
- [ ] Transcript renders Devanagari, Kannada, and Bengali script correctly (fonts
      loaded, no fallback boxes), including mixed-script lines
- [ ] Playback handles a sample-rate change between turns, since Piper and Azure differ
- [ ] Echo cancellation enabled in `getUserMedia`. Test barge-in with speakers, not only headphones
- [ ] Tested in Chrome, Safari, and one mobile browser

Done when: a full call works against the mock pipeline, and the agent's own voice
through laptop speakers does not trigger barge-in.

### E: Persistence and metrics API (Garlic)

Owns `app/db/`, `app/metrics/`, `scripts/seed.py`. Depends on: Phase 0.

- [ ] Async engine, SQLite locally and Neon in production, migrations with Alembic
- [ ] Seed script: about 20 synthetic customers with varied loan states, spread across
      the four languages. Each customer has a different `phone_last4` + `dob` pair.
- [x] `customers` has a unique constraint on (`phone_last4`, `dob`), with a test
      (section 11, note 3)
- [ ] Trace recorder: writes turns and tool calls off the hot path (queue + background
      writer). One recorder per process: the registered factory returns the same
      instance on every `get_trace_sink` call (section 11, note 2).
- [ ] `docs/quotas.md`: Garlic owns the file and fills in the provider rows (Deepgram,
      Groq, Whisper, Gemini, Piper, Azure Speech), each with the date checked
      (section 11, note 4)
- [ ] `GET /api/calls`, `GET /api/calls/{id}` (transcript plus per-turn waterfall)
- [ ] `GET /api/metrics/latency` (p50/p95 per stage and for time-to-first-audio,
      filterable by date, provider, and language)
- [ ] `GET /api/metrics/summary` (calls, handoff rate, interruption rate, verification
      failures, calls per language, language switches per call)

Done when: the API returns correct percentiles for a fixture of known traces, and a
database outage does not break a live call.

### F: Dashboard (Garlic)

Owns `web/dashboard/`. Depends on: Phase 0 for the schema. Build against fixture JSON
until E's endpoints exist.

- [ ] Latency overview: p50/p95 time-to-first-audio, stage breakdown
- [ ] Per-call view: transcript with a waterfall bar for each turn, interruptions marked
- [ ] Summary tiles: calls, handoff rate, interruption rate
- [ ] Language filter on every latency view, and a side-by-side comparison of the four
      languages. Never show one blended latency figure across languages.
- [ ] Per-call view shows the language of each turn and marks where a switch happened
- [ ] Region caveat shown next to latency figures (server region, provider regions)
- [ ] Eval results page reading `evals/reports/latest.json`

Done when: it renders correctly from fixtures, then from the live API with no code change.

### G: Evals (Garlic)

Owns `evals/`. Depends on: Phase 0. Dataset work has no code dependency and can start on day one.

- [ ] Hinglish test set: 100+ utterances with reference transcripts and intent labels.
      Record real audio from several speakers. Mix of pure English, pure Hindi, and code-switched.
- [ ] `[kn]` Kannada test set: 100+ utterances, same structure, including
      Kannada-English code-switching. **Start recording on day one.** Needs at least one
      native speaker to record and a second to check the transcripts.
- [ ] `[bn]` Bengali test set: same structure and the same native-speaker requirement
- [ ] All sets cover the same intents and scenarios, so results are comparable across languages
- [ ] WER runner (via A's `stt_file.py`), with a defined normalisation per language:
      mixed native and Latin script, numerals, and spelling variants. Write the rules
      down, since they change the number a lot. Report character error rate alongside
      WER for Kannada, where long agglutinated words make WER look harsh.
- [ ] Language ID suite: accuracy and confusion matrix on first utterances, broken down
      by utterance length, plus scripted mid-call switches measuring turns-to-recover
- [ ] Intent accuracy runner (via B's agent, text mode), per language
- [ ] Task completion: 20+ scripted multi-turn scenarios with pass/fail checks on tool
      calls and final state. Include adversarial ones: wrong identity, asking for someone
      else's loan, pressing for a made-up figure.
- [ ] Interruption suite: inject audio at fixed offsets into a mock session, check stop
      time, history truncation, and recovery on the next turn
- [ ] Task completion scenarios translated and run in each language, including the
      adversarial ones
- [ ] One command: `python -m evals.run --suite all --lang all` writing `evals/reports/<timestamp>.json`
- [ ] Results table in the README, one row per language, with the date and provider versions

Done when: all five suites run from one command for every language, and results are
reproducible on another device.

### H: Deploy and operations (Onion)

Owns `deploy/`, `Dockerfile`, non-seed `scripts/`. Depends on: Phase 0. The mock
pipeline can be deployed long before real providers are wired in.

- [ ] Dockerfile: slim image, Piper voices and Silero weights baked in at build time
- [ ] Hugging Face Space configured, secrets set (including the Azure Speech key and
      region), WebSocket confirmed working through the proxy
- [ ] Azure Speech resource created in the region closest to the Space
- [ ] Daily usage caps for Azure characters and Whisper audio, with a clear message
      when a language is temporarily unavailable
- [ ] Neon database created, migrations and seed applied
- [ ] Limits: calls per IP per day, concurrent sessions, call duration cap
- [ ] Wake-up handling: loading state for a sleeping Space, plus a recorded fallback demo video
- [ ] Structured logs, with no audio and no personal data in them
- [ ] `docs/quotas.md` filled in

Done when: the public link runs a mock call end to end, and a second simultaneous
caller gets a clear "busy" message rather than a broken call.

---

### Phase 2: Integration (serial, needs O plus the streams named)

Do these in order. Steps marked **joint** connect an Onion stream to a Garlic stream:
both devices need the other's latest work, and should be available at the same time
to fix what breaks.

1. [ ] Onion (O + D): real browser against the mock pipeline, including barge-in
2. [ ] Onion (O + A): real VAD and STT. Tune end-of-speech silence.
3. [ ] **Joint** (O + B): real agent. First full text-in, text-out call.
4. [ ] **Joint** (O + C): real TTS. **First real voice call, in English/Hindi (milestone M2).**
5. [ ] **Joint** (O + A + B + C): Kannada route end to end with a manual language choice.
       **First Kannada call (milestone M2k).**
6. [ ] Onion (O + A): auto-detection across English, Hindi, and Kannada, including a mid-call switch
7. [ ] **Joint** (O + E): traces persisted. Garlic switches dashboard F from fixtures to the API.
8. [ ] Onion (H): deploy the real pipeline. Measure latency from the Space, not from a laptop.
9. [ ] Garlic (G): full eval run against the deployed build, for English/Hindi and Kannada.
10. [ ] **Joint**: Bengali route switched on, added to auto-detection, and evaluated.
        Steps 5, 6, and 9 repeated for `bn`.

Steps 5 and 6 do not wait for step 7 onward to be perfect, and Bengali never blocks
a Kannada milestone.

### Phase 3: Hardening and write-up

- [ ] Latency tuning from the waterfall, per language: biggest stage first
- [ ] Native-speaker listening pass for Kannada and Bengali: naturalness, number
      reading, and whether replies sound like a real loan officer or a translation
- [ ] Barge-in tuning on real hardware: speakers, noisy rooms, backchannels
- [ ] Load check: what happens with 2 and 3 concurrent calls on the free container
- [ ] README: architecture, the eval table, the latency table with the region caveat, what is hand-built
- [ ] Blog post on the barge-in logic and the event-loop blocking lesson
- [ ] Decide on Pipecat: migrate, or document why not

## 7. What runs in parallel

Two lanes, one per device. Work inside a lane is sequential. The lanes run side by side.

```
Onion   Phase 0 ─▶ O ─▶ D ─▶ H(mock) ─▶ A ─▶ A[kn] ─▶ lang detect ─▶ H(real) ─▶ [bn] ─▶ tuning
                         │        │      │       │            │           │         │
                         M1 ──────┘      M2      M2k          M3 ─────────┘         M5
                                         │       │            │                     │
Garlic  prep ────▶ B ─▶ C(Piper) ────────┴▶ C[kn] ─▶ B[kn] ───┴▶ E ─▶ F ─▶ G ─▶ M4 ─▶ [bn] ─▶ write-up
        └─ datasets: Hinglish and Kannada recording runs in the background from day one ─┘
```

Work order for each device, by stage:

| Stage | Onion | Garlic | Ends with |
|---|---|---|---|
| 0 | Phase 0: skeleton, contracts, mocks | Quota checks, native speakers confirmed, eval scenarios written, recordings started. Review the contracts. | **M0** |
| 1 | O: turn loop, barge-in, session limits. D: browser client. H: mock pipeline on the Space. | B: agent, tools, three guardrails, in text mode. C: Piper for English and Hindi. | **M1** (Onion alone) |
| 2 | A: Silero VAD, Deepgram, Groq Whisper for English/Hindi | Finish C. Start `[kn]` Azure client. | **M2** (joint: steps 3, 4) |
| 3 | A `[kn]`: Kannada STT comparison and routing | C `[kn]`: Kannada voice and normalisation. B `[kn]`: prompts, numerals, LLM choice. | **M2k** (joint: step 5) |
| 4 | A: language detector. O: session language state. H: real deploy and limits. | E: database, trace recorder, metrics API. F: dashboard on fixtures, then the API. | **M3** (joint: step 7) |
| 5 | Latency and barge-in tuning from the waterfall. D: language picker and script rendering polish. | G: all runners, Kannada and Hinglish eval runs, README table | **M4** |
| 6 | A `[bn]`: Bengali STT. Add `bn` to detection. | C `[bn]`, B `[bn]`, G `[bn]`: voice, prompts, test set, eval run | **M5** (joint: step 10) |
| 7 | Load check, fallback video, Pipecat decision | README, blog post | **M6** |

If one device finishes a stage early, it pulls the next item from its own lane. It
does not start work in the other device's paths.

- Onion's streams and Garlic's streams are independent after Phase 0, because each one
  talks only to the contracts and the mocks.
- The two devices meet only at the joint steps: Phase 2 steps 3, 4, 5, 7, and 10.
- Soft links, none of them blocking: F uses E's endpoints once they exist, G's WER
  runner uses A's script, G's intent runner uses B's agent.
- Only two things are serial: Phase 0, and the integration steps in Phase 2.
- Language work runs inside the streams, not as a separate stream. The `[kn]` tasks in
  A, B, C, and G are independent of each other and start with the baseline. Each
  `[bn]` task waits only for the `[kn]` task in the same stream.
- The Kannada and Bengali datasets (G) need native speakers and wall-clock time more
  than code. They are the long pole, so Garlic starts them in stage 0 and keeps them
  moving in the background.

Why the split is this way: O and D share the barge-in logic (server cancel, client
flush), so one device owning both avoids a slow back-and-forth. A feeds O directly.
Everything Garlic owns can be built and tested in text or from files, with no live
audio session, so Garlic is never blocked waiting for Onion's pipeline.

## 8. Working together across devices

- **Ownership is fixed** in the table below. If Onion and Garlic agree to swap a
  stream, update the table in the same change.
- **Stay in your paths** (section 5). This is what keeps merges clean.
- **Contracts are frozen.** To change `app/core/` or `docs/protocol.md`, both devices
  agree first, then Onion makes the change.
- **`pyproject.toml` and `uv.lock` are shared.** Add dependencies with `uv add`, one
  line each. If both devices added some, keep both sets of lines in `pyproject.toml`
  and run `uv lock` again. Never merge `uv.lock` by hand.
- **Register your implementations** in your own package (section 4.1). Do not edit
  `app/core/registry.py` or `app/core/mocks.py` to wire them in.
- **Shared files** are this plan and `docs/`. In this plan, each device edits only its
  own workstream sections and its own rows in the table. In `docs/`, one file per
  topic with the owner named at the top.
- **Sync at every milestone.** Both devices bring in the other's work before a joint step.
- **Mocks stay working.** The mock end-to-end test must pass at all times. It is the
  shared definition of "not broken".
- **No keys in the repo.** Each device has its own `.env`. Only `.env.example` is tracked.
- **Tests run without keys.** Tests that hit real providers are marked `@pytest.mark.live`
  and skipped by default.
- **Tick your own checkboxes** in section 6 as tasks land. Edit only your own section.
- Branching, commits, and merging are up to the team. This plan does not prescribe a flow.

| Stream | Owner | Status |
|---|---|---|
| Phase 0 | Onion | built, 63 tests passing, waiting for Garlic's contract review |
| O Orchestrator | Onion | not started |
| A VAD + STT + language detection | Onion | not started |
| D Browser client | Onion | not started |
| H Deploy | Onion | not started |
| B Agent | Garlic | not started |
| C TTS | Garlic | not started |
| E DB + metrics | Garlic | not started |
| F Dashboard | Garlic | not started |
| G Evals | Garlic | not started |

## 9. Milestones

| | Milestone | Proves |
|---|---|---|
| M0 | Phase 0 merged, mock test green | Both devices can build against the contracts |
| M1 | Browser talks to the mock pipeline on the public Space, barge-in works | Transport, cancellation, and hosting are sound |
| M2 | First real voice call in English/Hindi, locally | All four providers work together |
| M2k | First real Kannada call, manual language choice | The second provider route works |
| M3 | Auto-detection working, real pipeline deployed, traces in the dashboard | Latency is measured per language, not guessed |
| M4 | Eval run for English/Hindi and Kannada published in the README | The numbers that carry the project exist |
| M5 | Bengali live and evaluated | Adding a language is a routing entry plus a test set |
| M6 | Write-up, fallback video, Pipecat decision | Ready to show |

Rough pacing with the two devices: M0 in the first few days, M1 and M2 by the end of
week 2, M2k in week 3, M3 in week 4, M4 in week 5, M5 in week 6, M6 in week 7. The
stages in section 7 map onto these one to one.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Silero or Piper blocks the event loop | Executor only, enforced by the interface rules. Add a test that fails if the loop stalls beyond a threshold. |
| Free tiers throttle or sleep | Fallback providers behind the interfaces, session limits, recorded demo video |
| Agent's own voice triggers barge-in | Browser echo cancellation, minimum speech duration, tested on speakers |
| Hinglish WER is poor | Compare providers early in A, pick on data. Report the number honestly either way. |
| Kannada or Bengali STT is too inaccurate to hold a call | Garlic runs a quick check on a handful of clips in stage 0, before anything is built on it. Onion does the full comparison in stage 3. Try the other Indic providers listed in section 2. If nothing is usable, ship that language as "experimental" with its numbers shown. |
| No native speaker available for a language | That language cannot be evaluated or tuned. Confirm speakers for Kannada and Bengali before starting their datasets. |
| Auto-detection picks the wrong language | Manual picker and mid-call override, script-based correction, ask on low confidence. Language ID accuracy is measured in G. |
| Kannada and Bengali latency is worse (batch STT, network TTS) | Report per language. Do not let it drag down or hide behind the English/Hindi figure. |
| Azure free tier runs out | Usage counter and daily cap in C and H. Piper still serves English and Hindi. |
| Agent replies in the wrong language or mixes scripts | Session language is passed to the LLM explicitly and checked on output before TTS. |
| Piper is too slow on the free CPU | Garlic benchmarks it in stage 1, as the first task in C. Switch to the backup provider if it misses the target. |
| Latency looks bad from a far region | Report the server and provider regions next to every figure |
| Model invents loan figures | Guardrail 2 is enforced in code and covered by adversarial evals |
| Contract churn breaks parallel work | Phase 0 freeze, additive changes only |

## 11. Notes from Garlic for Onion

Written by Garlic on 2026-10-07, from the contract review. Each note says what Garlic
has decided inside its own streams, and what, if anything, it asks of Onion. Nothing
here changes an interface.

1. **Agent text arrives a sentence at a time.** Guardrail 2 has to check every figure
   before it is spoken, and a `TextDelta` cannot be taken back once the orchestrator
   has it. So the agent (B) holds each sentence until it passes, then yields it whole.
   - For O: do not rely on token-sized deltas. A delta may be a full sentence, and the
     sentence splitter must cope with that.
   - For the trace: `llm_first_token` will be the time of the first checked sentence,
     not the model's first token, so the `llm` stage includes the check and the
     `sentence` stage will be close to zero. Section 4.3 still says "first text
     delta", which stays true. Garlic will state this next to the latency figures in
     the dashboard and the write-up.
   - Asked of Onion: nothing, unless you want the model's raw first token as its own
     mark. That would be an added `Mark`, and Garlic would report it through an added
     event.

2. **Garlic's factories return shared instances.** The registry runs the factory on
   every `get_*` call. Piper voices (C) and the trace recorder (E) are costly to
   build, so each is created once per process inside Garlic's packages, and the
   factory hands out that one instance.
   - For O: call `get_tts(lang)` and `get_trace_sink()` as often as is convenient, per
     call or per turn. Do not close or tear down what they return at the end of a call:
     other sessions are using the same object.
   - The agent is the exception. `get_agent(ctx)` returns a new one per call, as the
     interface says.
   - Asked of Onion: nothing.

3. **Callers are identified by `phone_last4` plus `dob`.** These are the only
   identifying fields in `customers` that a caller can say aloud. Garlic added a unique
   constraint on the pair in `app/db/models.py` (E's path), with a test, and the seed
   script will give every customer a different pair.
   - This is an added constraint: no column or table was renamed or removed, and the
     table list in section 4.4 is unchanged.
   - Asked of Onion: pull the change before the freeze, and say so if you object.

4. **`docs/quotas.md` belongs to Garlic.** Phase 0 gives the provider and quota checks
   to Garlic, but workstream H also lists "`docs/quotas.md` filled in". Garlic takes
   the file and the provider rows, and has added the task to workstream E's list.
   - Asked of Onion: add the Hugging Face Spaces rows (CPU, RAM, sleep policy) and the
     Azure region to that file when you do H, and reword or remove H's checkbox.
     Garlic has not edited H's section.
