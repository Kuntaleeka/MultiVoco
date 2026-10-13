# TTS

Owner: Garlic (workstream C).

## Piper, English and Hindi

| | English | Hindi |
|---|---|---|
| Voice | `en_US-lessac-medium` | `hi_IN-priyamvada-medium` |
| Sample rate | 22 050 Hz | 22 050 Hz |
| Model file | 60 MB | 61 MB |

Both are female voices, to match the agent's Hindi ("कर सकती हूँ"). Piper has no Indian
English voice, so English is a US accent. Piper has no Kannada voice. It has one Bengali
voice, `bn_BD-google-medium`, with a Bangladesh accent, not tried yet.

Fetch the voices into `voices/`, which git ignores:

```sh
uv run python -m piper.download_voices en_US-lessac-medium hi_IN-priyamvada-medium --download-dir voices
```

`piper-tts` 1.8.0 runs on ONNX Runtime 1.31, with no PyTorch. It is licensed
GPL-3.0-or-later.

## Benchmark

Checked 2026-10-11 on Garlic's laptop (Windows, 16 logical CPUs), with
`uv run python -m app.tts.bench`. **This is not the deployment host.** Render's free
service has 512 MB and a shared CPU, so the timings there will be worse. Run the same
command on Render before relying on them.

One sentence of about ten words, 6 to 8 seconds of audio. Piper produces a whole
sentence at once, so "first audio" is the time to synthesise the sentence.

| ONNX Runtime settings | Memory, both voices loaded | First audio, English | First audio, Hindi |
|---|---|---|---|
| Defaults (memory arena on) | 618 MB | 263 ms | 329 ms |
| **Arena off, threads automatic (what we use)** | **238 MB** | **361 ms** | **454 ms** |
| Arena off, 2 threads | 236 MB | 586 ms | 741 ms |
| Arena off, 1 thread, no graph optimisation | 240 MB | 909 ms | 1116 ms |

- Each loaded voice costs about 88 MB with the arena off, against about 290 MB with it on.
- With the arena on, two voices do not fit in 512 MB. With it off they do, with room
  for the rest of the app. The cost is synthesis about a third slower.
- The 1-thread and 2-thread rows are the nearest guide to a shared CPU: expect roughly
  0.6 to 1.1 s to first audio for a ten-word sentence, worse if the CPU is throttled.
- Loading a voice takes about 5 s, so voices are loaded at startup, not on the first call.
- Set `MULTIVOCO_PIPER_THREADS` to cap the threads. 0, the default, lets ONNX Runtime choose.

## How it behaves

- One `PiperTTS` is shared by every call. Only one thread is inside Piper at a time,
  because its phonemiser keeps the language in a global. Concurrent calls wait their turn.
- Audio is handed over in 100 ms pieces. Cancelling a call stops it at the next piece.
  Piper cannot be stopped inside a sentence, so a sentence already being computed
  finishes in the pool and is thrown away.
- Text is cleaned before speaking (`app/tts/normalize.py`): markup, grouping commas,
  currency signs, and trailing ".00" are removed, and "EMI" is spelled out. Numbers are
  left as digits, which Piper reads aloud itself.

## Sarvam AI, Kannada and Bengali

Sarvam took Azure Speech's place on 2026-10-13, because Azure needs a card to sign up.
The client is `app/tts/sarvam.py`. The table is from Sarvam's documentation, read on
2026-10-13.

Checked with one real request per language on 2026-10-13, from Garlic's laptop in
India, one ten-word sentence each (about 7.5 s of audio):

| | First audio | Whole sentence received |
|---|---|---|
| Kannada (first request, new connection) | 858 ms | 2353 ms |
| Bengali (second request, connection reused) | 212 ms | 1667 ms |

Two requests are not a benchmark: they show that the voice name `priya` is accepted for
both languages, that audio arrives as raw PCM while it is still being generated, and
roughly what to expect. Garlic's user listened to both samples on 2026-10-13 and found
them good. A native speaker's check of each language is still open.

| | |
|---|---|
| Endpoint | `POST https://api.sarvam.ai/text-to-speech/stream`, key in `api-subscription-key` |
| Model | `bulbul:v3` (`MULTIVOCO_SARVAM_MODEL`) |
| Voice | `priya` (`MULTIVOCO_SARVAM_SPEAKER`), not chosen by a native speaker yet |
| Languages | `kn-IN`, `bn-IN`, and also `hi-IN`, `en-IN` |
| Audio | raw PCM16 (`linear16`) at 22 050 Hz, the same as Piper |
| Free credit | ₹1,000 once, on signup. At ₹30 per 10,000 characters that is about 330,000 characters. |
| Rate limit | 60 requests a minute on the starter plan |

- A daily cap, `MULTIVOCO_SARVAM_DAILY_CHARS` (30,000 by default), stops one busy day
  from spending the credit. It is counted per process and starts again on a restart.
- Out of credit, over the rate limit, or over the daily cap all raise `QuotaExceeded`,
  which the caller sees as "language unavailable".
- Sarvam does its own reading of numbers and English words (`enable_preprocessing`), so
  only markup and trailing ".00" are removed before sending.
- To route Kannada and Bengali to it, set
  `MULTIVOCO_ROUTE_OVERRIDES={"kn": {"tts": "sarvam"}, "bn": {"tts": "sarvam"}}`.

## Not done yet

- A listening check of both Piper voices, and of Hindi sentences with English words in them.
- Reading loan IDs and phone digits one at a time, and dates.
- A native speaker's choice of Sarvam voice for Kannada and for Bengali, and a proper
  timing run from the deployed host.
- Sarvam as a fallback for English and Hindi when Piper fails.
- The Piper benchmark on Render.
