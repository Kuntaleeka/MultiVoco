# WebSocket protocol

Owner: Onion. Frozen after Phase 0: fields may be added, never renamed or removed.
The message models are in `app/core/protocol.py`. If this file and the code disagree,
the code is right and this file has a bug.

## Endpoints

| Path | Purpose |
|---|---|
| `/ws/call` | A voice call. Everything below describes this endpoint. |
| `/ws/echo` | Sends back every text and binary frame unchanged. For checking transport. |
| `/healthz` | `{"status": "ok", "region": ..., "mock": ..., "active_sessions": ...}` |

## Frames

A WebSocket frame is either **binary** (audio) or **text** (one JSON object with a
`type` field). There is no other framing.

### Audio from the client

- PCM, signed 16-bit little-endian, mono, **16 000 Hz**
- Exactly **512 samples (1024 bytes, 32 ms)** per binary frame
- Sent continuously from `start` until `end`, including while the agent is speaking.
  The server needs it to detect barge-in.
- Capture with `echoCancellation: true`, or the agent will interrupt itself.

### Audio from the server

- PCM, signed 16-bit little-endian, mono
- The sample rate is given by the `audio_start` message that precedes it, and **can
  differ from one turn to the next**, because languages use different TTS providers.
- Chunk sizes vary. Every binary frame belongs to the turn of the latest `audio_start`.

## Client to server

| Message | When |
|---|---|
| `{"type": "start", "lang": "auto"}` | Once, first. `lang` is `auto`, `en`, `hi`, `kn`, or `bn`. Defaults to `auto`. |
| `{"type": "set_language", "lang": "kn"}` | The caller picks a language mid-call. Not `auto`. Takes effect on the next turn. |
| `{"type": "playback_started", "turn_id": 3, "t_client_ms": 81234.5}` | The first audio sample of turn 3 was actually played. `t_client_ms` is the client's own clock. |
| `{"type": "playback_position", "turn_id": 3, "ms_played": 640}` | Reply to `flush`: how much of turn 3's audio was played before it was dropped. |
| `{"type": "end"}` | The caller hangs up. |

## Server to client

| Message | Meaning |
|---|---|
| `{"type": "ready", "session_id": "..."}` | Reply to `start`. Begin sending audio. |
| `{"type": "state", "value": "listening"}` | `listening`, `thinking`, `speaking`, or `handoff`. For the status indicator. |
| `{"type": "language", "lang": "kn", "source": "audio", "confidence": 0.93}` | The session language was set or changed. `source` is `manual`, `audio`, `script`, or `asked`. |
| `{"type": "transcript", "role": "user", "text": "...", "final": false, "turn_id": 3}` | Live transcript. `role` is `user` or `agent`. A non-final message replaces the previous non-final one for the same role and turn. |
| `{"type": "audio_start", "turn_id": 3, "sample_rate": 24000}` | Audio for turn 3 follows as binary frames. |
| `{"type": "audio_end", "turn_id": 3}` | No more audio for turn 3. |
| `{"type": "flush", "turn_id": 3}` | Barge-in. Stop playing turn 3 **now**, drop everything queued, and reply with `playback_position`. |
| `{"type": "error", "code": "busy", "message": "..."}` | See error codes. |

`turn_id` is an integer that starts at 0 and goes up by one per agent reply. The
greeting is turn 0.

## Sequences

### A normal turn

```
client                                   server
  │ ── start {lang: "kn"} ───────────────▶ │
  │ ◀─────────────── ready {session_id} ── │
  │ ◀──── language {kn, source: manual} ── │
  │ ◀──────── audio_start {turn 0, rate} ─ │   greeting
  │ ◀──────────────────── binary audio ─── │
  │ ◀──────────────── audio_end {turn 0} ─ │
  │ ◀──────────── state {listening} ────── │
  │ ── binary audio (always flowing) ────▶ │
  │ ◀── transcript {user, final: false} ── │
  │ ◀── transcript {user, final: true} ─── │
  │ ◀──────────── state {thinking} ─────── │
  │ ◀──────── audio_start {turn 1, rate} ─ │
  │ ◀──────────── state {speaking} ─────── │
  │ ◀── transcript {agent, final: false} ─ │
  │ ◀──────────────────── binary audio ─── │
  │ ── playback_started {turn 1} ────────▶ │
  │ ◀── transcript {agent, final: true} ── │
  │ ◀──────────────── audio_end {turn 1} ─ │
  │ ◀──────────── state {listening} ────── │
```

### Barge-in

The caller starts talking while the agent is speaking.

```
  │ ◀──────────────────── binary audio ─── │   turn 1, still streaming
  │ ── binary audio (caller speaks) ─────▶ │   server VAD: speech start
  │ ◀──────────────────── flush {turn 1} ─ │   server has cancelled LLM and TTS
  │ ── playback_position {turn 1, 640} ──▶ │
  │ ◀── transcript {agent, final: true} ── │   text cut to what was heard
  │ ◀──────────── state {listening} ────── │
```

Client rules:

- On `flush`, stop the audio output immediately. Do not let a buffered chunk finish.
- Binary frames that arrive after a `flush` and before the next `audio_start` belong
  to the cancelled turn. Drop them.
- Always answer `flush` with `playback_position`, even if nothing had played yet
  (`ms_played: 0`). The server waits briefly for it, then assumes 0.

### Language detection (`lang: "auto"`)

```
  │ ── start {lang: "auto"} ─────────────▶ │
  │ ◀─────────────── ready ─────────────── │
  │ ◀──── greeting, language-neutral ───── │   turn 0
  │ ── binary audio (first utterance) ───▶ │   buffered, then detected
  │ ◀──── language {kn, source: audio} ─── │
  │ ◀── transcript {user, final: true} ─── │
  │                 ...                     │   the call continues in Kannada
```

- No `language` message is sent until a language is known. Show "detecting".
- The first user turn has no non-final transcripts, because the audio is buffered for
  detection before transcription starts.
- If detection is unsure, the agent asks which language the caller wants. The answer
  arrives as `language` with `source: "asked"`.
- If the transcript's script disagrees with the session language two turns in a row,
  the server switches and sends `language` with `source: "script"`.
- `set_language` always wins and stops further automatic switching.

### What counts as an interruption

Speech during a reply becomes a barge-in once it has lasted about 250 ms. Shorter
sounds are ignored and the reply carries on. The server stays interruptible until the
audio it sent has had time to play, not just until it has finished sending it, so a
`flush` can arrive after `audio_end` for the same turn.

## Errors and close codes

| `code` | Meaning | Connection |
|---|---|---|
| `bad_message` | Unparseable JSON, unknown `type`, wrong fields, or a wrong-sized audio frame | Closed with 4400 |
| `busy` | Too many concurrent calls | Closed with 4429 |
| `limit_reached` | Call length, turn count, or idle timeout reached | Closed with 4408 |
| `language_unavailable` | A provider for the requested language is down or over quota | Stays open. Pick another language. |
| `provider_error` | A provider failed during a turn | Stays open. The turn is abandoned. |
| `internal` | Anything else | Closed with 4500 |

A normal hang-up closes with 1000.

## Examples

```json
{"type": "start", "lang": "auto"}
{"type": "ready", "session_id": "0b9d6c1e-6a55-4c0e-9d0e-0f2f5a6f7a10"}
{"type": "language", "lang": "kn", "source": "audio", "confidence": 0.93}
{"type": "transcript", "role": "user", "text": "ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?", "final": true, "turn_id": 1}
{"type": "state", "value": "thinking"}
{"type": "audio_start", "turn_id": 1, "sample_rate": 24000}
{"type": "playback_started", "turn_id": 1, "t_client_ms": 81234.5}
{"type": "flush", "turn_id": 1}
{"type": "playback_position", "turn_id": 1, "ms_played": 640}
{"type": "error", "code": "limit_reached", "message": "Call limit of 5 minutes reached."}
```
