# MultiVoco Implementation Plan

Voice AI loan servicing agent. Python only, free-tier hosting, public demo link.
Source of the stack choices: `voice-ai-python-stack.pdf`.

This file is the shared reference for everyone working on the repo, on any device.
Read sections 1 to 4 before writing code. Then pick a workstream from section 6.

---

## 1. What we are building

A browser-based voice agent that a borrower can talk to in English or Hinglish. It can:

- verify the caller's identity
- answer questions about their loan (balance, next EMI, due date, payment history)
- take simple actions (payment promise, callback request)
- hand off to a human when it should
- be interrupted mid-sentence (barge-in) and recover cleanly

Around the agent, the parts that carry the project:

- a **latency waterfall** per turn, with p50/p95 time-to-first-audio
- an **eval harness** (WER, intent accuracy, task completion, interruption handling) on a Hinglish test set
- **guardrails** (identity gate, tool-only figures, human handoff)

## 2. Decisions already made

| Topic | Decision | Notes |
|---|---|---|
| Pipeline | Hand-rolled loop: VAD → STT → LLM → TTS | Revisit Pipecat only after M2. Do not add it before then. |
| Server | FastAPI + WebSockets, asyncio, one process | |
| VAD | Silero VAD on CPU, in a thread pool | Must never run on the event loop |
| STT | Deepgram streaming as primary, Groq Whisper as fallback | Both behind one interface |
| LLM | Groq (Llama-class) with tool calling | Gemini free tier is the backup provider |
| TTS | Piper in the same container, in a thread/process pool | Azure Speech is the backup |
| Database | Postgres on Neon (free tier). SQLite for local dev and tests | Same SQLAlchemy models for both |
| Frontend | Plain JS, served as static files by FastAPI | One container, one public URL, no CORS |
| Dashboard | Plain JS page reading the metrics API | Same static mount |
| Hosting | Hugging Face Spaces (Docker) | |
| Python | 3.11, dependencies managed with `uv` | |

Open items, to be checked by whoever takes the workstream (quotas change often):

- Deepgram: current free credit, and which model handles Hindi-English code-switching best
- Groq: current rate limits for the chosen model and for Whisper
- Piper: which Hindi voice is acceptable for Hinglish output, and its sample rate
- Hugging Face Spaces: CPU/RAM on the free tier and the sleep policy

Record the findings in `docs/quotas.md` with the date checked.

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

```python
class VAD(Protocol):
    async def process(self, frame: bytes) -> VADEvent | None: ...   # SPEECH_START / SPEECH_END

class STT(Protocol):
    async def start(self, language_hint: str | None) -> None: ...
    async def send_audio(self, frame: bytes) -> None: ...
    async def finish(self) -> None: ...                             # flush on speech end
    def events(self) -> AsyncIterator[Transcript]: ...              # partial and final

class LLM(Protocol):
    def stream(self, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[LLMEvent]: ...
    # LLMEvent = TextDelta | ToolCall | Done

class TTS(Protocol):
    sample_rate: int
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...    # PCM16 mono chunks

class Tool(Protocol):
    spec: ToolSpec
    async def run(self, args: dict, ctx: CallContext) -> dict: ...
```

Rules for every implementation:

- Must stop promptly when its task is cancelled (`asyncio.CancelledError`), and release
  sockets and pool work in `finally`.
- Must not block the event loop. CPU work goes through the shared executor in `app/core/pools.py`.
- Every interface ships with a mock in `app/core/mocks.py` (scripted transcripts, canned
  LLM replies, a sine-wave TTS). Workstreams develop against the mocks, not against each other.

### 4.2 WebSocket protocol (`docs/protocol.md`)

Endpoint: `/ws/call`

| Direction | Type | Payload |
|---|---|---|
| client → server | binary | PCM16 mono, 16 kHz, 512 samples per frame (32 ms, the Silero window) |
| client → server | JSON | `{"type":"start","lang_hint":"hi-en"}` |
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
| server → client | JSON | `{"type":"error","code":...,"message":...}` |

### 4.3 Latency trace (`app/core/trace.py`)

One record per turn. All times are server monotonic milliseconds from session start,
except the client one.

| Mark | Meaning |
|---|---|
| `speech_end` | VAD declares end of user speech |
| `stt_final` | final transcript received |
| `llm_first_token` | first text delta |
| `llm_first_sentence` | first sentence handed to TTS |
| `tts_first_chunk` | first audio chunk produced |
| `audio_sent` | first audio chunk written to the socket |
| `playback_started` | client reports first audio played (client clock, stored separately) |

Headline metric: **time to first audio = `audio_sent` − `speech_end`**. Also store
`interrupted: bool`, `ms_played`, tool calls with durations, provider names, and the
server region.

### 4.4 Database schema (`app/db/models.py`)

- `customers` (id, name, phone_last4, dob, language)
- `loans` (id, customer_id, principal, outstanding, emi_amount, next_due_date, status)
- `payments` (id, loan_id, amount, paid_on, status)
- `calls` (id, started_at, ended_at, verified_customer_id, outcome, handoff_reason)
- `turns` (id, call_id, idx, user_text, agent_text, interrupted, ms_played, trace JSON)
- `tool_calls` (id, turn_id, name, args JSON, result JSON, duration_ms)

All loan data is synthetic. Seed script: `scripts/seed.py`.

## 5. Repository layout and ownership

Each path has one owning workstream. Stay inside your paths. If you need a change
somewhere else, ask the owner.

```
app/
  core/          Phase 0, then frozen   interfaces, mocks, trace, pools, config
  main.py        O                      FastAPI app, routes, static mounts
  pipeline/      O                      session, turn orchestrator, sentence splitter
  audio/         A                      Silero VAD wrapper, frame buffering
  stt/           A                      Deepgram and Groq Whisper clients
  agent/         B                      prompt, LLM client, tool loop, guardrails
  tools/         B                      loan tools
  tts/           C                      Piper wrapper, backup provider
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

### Phase 0: Foundation (one device, do this first, nothing else starts until it lands)

- [ ] Project skeleton: `pyproject.toml`, `uv` lockfile, ruff, pytest, `.env.example`, `.gitignore`
- [ ] `app/core/interfaces.py`, `mocks.py`, `trace.py`, `pools.py`, `config.py`
- [ ] `docs/protocol.md` (section 4.2 written out in full, with example messages)
- [ ] `app/db/models.py` with the tables from 4.4, running on SQLite
- [ ] `app/main.py` with `/healthz` and a WebSocket echo endpoint
- [ ] A mock end-to-end test: scripted audio in → mock VAD/STT/LLM/TTS → audio out

Done when: `pytest` passes on a fresh clone with no API keys set.

---

Everything below can run **in parallel** once Phase 0 is in.

### O: Orchestrator (critical path, give it to the most experienced person)

Owns `app/pipeline/`, `app/main.py`. Depends on: Phase 0 only (builds against mocks).

- [ ] Session lifecycle: connect, `start`, teardown, cleanup of all tasks on disconnect
- [ ] Turn state machine: listening → thinking → speaking → listening
- [ ] Sentence splitter for streamed LLM text (handle Hindi danda `।`, abbreviations, numbers)
- [ ] LLM → TTS → socket pipeline with bounded queues (backpressure)
- [ ] Barge-in: on `SPEECH_START` while speaking, cancel LLM and TTS tasks, send `flush`,
      wait for `playback_position`, truncate the agent message in history to what was heard
- [ ] False-interrupt handling: ignore speech shorter than a threshold, and backchannels
- [ ] Trace marks from 4.3 emitted at each stage
- [ ] Per-session limits: max call length, max turns, idle timeout

Done when: with mocks, an interrupt stops outgoing audio within 100 ms of the VAD event
in tests, no task is left running after disconnect, and every turn produces a full trace.

### A: Audio in (VAD + STT)

Owns `app/audio/`, `app/stt/`. Depends on: Phase 0.

- [ ] Silero VAD wrapper running in the executor, with tunable thresholds and end-of-speech silence
- [ ] Deepgram streaming client: partials, finals, reconnect, keep-alive
- [ ] Groq Whisper fallback: buffer the utterance, transcribe on speech end
- [ ] Provider switch via config, plus automatic fallback on error
- [ ] Hinglish settings: language/model choice, tested on at least 20 recorded clips
- [ ] `scripts/stt_file.py`: transcribe a WAV from the command line (workstream G uses this)

Done when: both providers pass the interface tests, and a note in `docs/stt.md` compares
their latency and WER on the sample clips.

### B: Agent (LLM, tools, guardrails)

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
- [ ] Guardrail 3, handoff: explicit request, repeated misunderstanding, distress or
      dispute, out-of-scope requests
- [ ] `scripts/chat.py`: text REPL against the agent (workstream G uses this)

Done when: unit tests cover each guardrail, including attempts to get account data
before verification and attempts to get the model to invent a figure.

### C: TTS

Owns `app/tts/`. Depends on: Phase 0.

- [ ] Piper wrapper in the executor, streaming PCM chunks, cancellable mid-sentence
- [ ] Voice selection for English and Hindi, with the model download handled at build time
- [ ] Text normalisation: rupee amounts, dates, loan IDs, digits read one at a time
- [ ] Warm-up at startup so the first turn is not slow
- [ ] Backup provider behind the same interface
- [ ] Benchmark: time to first chunk and real-time factor on 2 vCPUs, in `docs/tts.md`

Done when: first chunk for a 10-word sentence arrives fast enough for the latency
target on Space-sized hardware, and cancellation stops synthesis within one chunk.

### D: Browser client

Owns `web/client/`. Depends on: Phase 0 (works against the echo endpoint and mock pipeline).

- [ ] AudioWorklet capture, resample to 16 kHz PCM16, 512-sample frames
- [ ] Playback queue that handles the announced sample rate, with gapless scheduling
- [ ] `flush` handling: stop audio immediately and report `playback_position`
- [ ] `playback_started` reporting for the client-side latency mark
- [ ] UI: call button, state indicator, live transcript, error and mic-permission states
- [ ] Echo cancellation enabled in `getUserMedia`. Test barge-in with speakers, not only headphones
- [ ] Tested in Chrome, Safari, and one mobile browser

Done when: a full call works against the mock pipeline, and the agent's own voice
through laptop speakers does not trigger barge-in.

### E: Persistence and metrics API

Owns `app/db/`, `app/metrics/`, `scripts/seed.py`. Depends on: Phase 0.

- [ ] Async engine, SQLite locally and Neon in production, migrations with Alembic
- [ ] Seed script: about 20 synthetic customers with varied loan states
- [ ] Trace recorder: writes turns and tool calls off the hot path (queue + background writer)
- [ ] `GET /api/calls`, `GET /api/calls/{id}` (transcript plus per-turn waterfall)
- [ ] `GET /api/metrics/latency` (p50/p95 per stage and for time-to-first-audio, filterable by date and provider)
- [ ] `GET /api/metrics/summary` (calls, handoff rate, interruption rate, verification failures)

Done when: the API returns correct percentiles for a fixture of known traces, and a
database outage does not break a live call.

### F: Dashboard

Owns `web/dashboard/`. Depends on: Phase 0 for the schema. Build against fixture JSON
until E's endpoints exist.

- [ ] Latency overview: p50/p95 time-to-first-audio, stage breakdown
- [ ] Per-call view: transcript with a waterfall bar for each turn, interruptions marked
- [ ] Summary tiles: calls, handoff rate, interruption rate
- [ ] Region caveat shown next to latency figures (server region, provider regions)
- [ ] Eval results page reading `evals/reports/latest.json`

Done when: it renders correctly from fixtures, then from the live API with no code change.

### G: Evals

Owns `evals/`. Depends on: Phase 0. Dataset work has no code dependency and can start on day one.

- [ ] Hinglish test set: 100+ utterances with reference transcripts and intent labels.
      Record real audio from several speakers. Mix of pure English, pure Hindi, and code-switched.
- [ ] WER runner (via A's `stt_file.py`), with a defined normalisation for mixed
      Devanagari and Latin script. Write the rule down, since it changes the number a lot.
- [ ] Intent accuracy runner (via B's agent, text mode)
- [ ] Task completion: 20+ scripted multi-turn scenarios with pass/fail checks on tool
      calls and final state. Include adversarial ones: wrong identity, asking for someone
      else's loan, pressing for a made-up figure.
- [ ] Interruption suite: inject audio at fixed offsets into a mock session, check stop
      time, history truncation, and recovery on the next turn
- [ ] One command: `python -m evals.run --suite all` writing `evals/reports/<timestamp>.json`
- [ ] Results table in the README, with the date and provider versions

Done when: all four suites run from one command and results are reproducible on
another device.

### H: Deploy and operations

Owns `deploy/`, `Dockerfile`, non-seed `scripts/`. Depends on: Phase 0. The mock
pipeline can be deployed long before real providers are wired in.

- [ ] Dockerfile: slim image, Piper voices and Silero weights baked in at build time
- [ ] Hugging Face Space configured, secrets set, WebSocket confirmed working through the proxy
- [ ] Neon database created, migrations and seed applied
- [ ] Limits: calls per IP per day, concurrent sessions, call duration cap
- [ ] Wake-up handling: loading state for a sleeping Space, plus a recorded fallback demo video
- [ ] Structured logs, with no audio and no personal data in them
- [ ] `docs/quotas.md` filled in

Done when: the public link runs a mock call end to end, and a second simultaneous
caller gets a clear "busy" message rather than a broken call.

---

### Phase 2: Integration (serial, needs O plus the streams named)

Do these in order. Each step is a short session with the owners involved.

1. [ ] O + D: real browser against the mock pipeline, including barge-in
2. [ ] O + A: real VAD and STT. Tune end-of-speech silence.
3. [ ] O + B: real agent. First full text-in, text-out call.
4. [ ] O + C: real TTS. **First real voice call (milestone M2).**
5. [ ] O + E: traces persisted. Dashboard F switches from fixtures to the API.
6. [ ] H: deploy the real pipeline. Measure latency from the Space, not from a laptop.
7. [ ] G: full eval run against the deployed build.

### Phase 3: Hardening and write-up

- [ ] Latency tuning from the waterfall: biggest stage first
- [ ] Barge-in tuning on real hardware: speakers, noisy rooms, backchannels
- [ ] Load check: what happens with 2 and 3 concurrent calls on the free container
- [ ] README: architecture, the eval table, the latency table with the region caveat, what is hand-built
- [ ] Blog post on the barge-in logic and the event-loop blocking lesson
- [ ] Decide on Pipecat: migrate, or document why not

## 7. What runs in parallel

```
Phase 0 ──┬── O  Orchestrator ─────────────┐
          ├── A  VAD + STT ────────────────┤
          ├── B  Agent + guardrails ───────┤
          ├── C  TTS ──────────────────────┼── Phase 2 integration ── Phase 3
          ├── D  Browser client ───────────┤
          ├── E  DB + metrics API ─────────┤
          ├── F  Dashboard (on fixtures) ──┤
          ├── G  Evals (dataset first) ────┤
          └── H  Deploy (mock pipeline) ───┘
```

- All nine streams are independent after Phase 0, because each one talks only to the
  contracts and the mocks.
- Soft links, none of them blocking: F uses E's endpoints once they exist, G's WER
  runner uses A's script, G's intent runner uses B's agent.
- Only two things are serial: Phase 0, and the integration steps in Phase 2.

Suggested split by number of devices:

| Devices | Split |
|---|---|
| 2 | Device 1: O, A, C, H. Device 2: B, D, E, F, G. |
| 3 | Device 1: O, H. Device 2: A, C, D. Device 3: B, E, F, G. |
| 4 | Device 1: O, H. Device 2: A, C. Device 3: B, G. Device 4: D, E, F. |

## 8. Working together across devices

- **Claim before you start.** Put your name against the workstream in the table below.
- **Stay in your paths** (section 5). This is what keeps merges clean.
- **Contracts are frozen.** To change `app/core/` or `docs/protocol.md`, raise it with
  everyone first, then one person makes the change.
- **Mocks stay working.** The mock end-to-end test must pass at all times. It is the
  shared definition of "not broken".
- **No keys in the repo.** Each device has its own `.env`. Only `.env.example` is tracked.
- **Tests run without keys.** Tests that hit real providers are marked `@pytest.mark.live`
  and skipped by default.
- **Tick your own checkboxes** in section 6 as tasks land. Edit only your own section.
- Branching, commits, and merging are up to the team. This plan does not prescribe a flow.

| Stream | Owner | Status |
|---|---|---|
| Phase 0 | | not started |
| O Orchestrator | | not started |
| A VAD + STT | | not started |
| B Agent | | not started |
| C TTS | | not started |
| D Browser client | | not started |
| E DB + metrics | | not started |
| F Dashboard | | not started |
| G Evals | | not started |
| H Deploy | | not started |

## 9. Milestones

| | Milestone | Proves |
|---|---|---|
| M0 | Phase 0 merged, mock test green | Everyone can start |
| M1 | Browser talks to the mock pipeline on the public Space, barge-in works | Transport, cancellation, and hosting are sound |
| M2 | First real voice call, locally | All four providers work together |
| M3 | Real pipeline deployed, traces in the dashboard | Latency is measured, not guessed |
| M4 | Full eval run published in the README | The numbers that carry the project exist |
| M5 | Write-up, fallback video, Pipecat decision | Ready to show |

Rough pacing with 2 to 3 people: M0 in the first few days, M1 and M2 by the end of
week 2, M3 in week 3, M4 and M5 in week 4.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Silero or Piper blocks the event loop | Executor only, enforced by the interface rules. Add a test that fails if the loop stalls beyond a threshold. |
| Free tiers throttle or sleep | Fallback providers behind the interfaces, session limits, recorded demo video |
| Agent's own voice triggers barge-in | Browser echo cancellation, minimum speech duration, tested on speakers |
| Hinglish WER is poor | Compare providers early in A, pick on data. Report the number honestly either way. |
| Piper is too slow on the free CPU | Benchmark in C during week 1. Switch to the backup provider if it misses the target. |
| Latency looks bad from a far region | Report the server and provider regions next to every figure |
| Model invents loan figures | Guardrail 2 is enforced in code and covered by adversarial evals |
| Contract churn breaks parallel work | Phase 0 freeze, additive changes only |
