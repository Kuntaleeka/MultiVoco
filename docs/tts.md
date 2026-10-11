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

## Not done yet

- A listening check of both voices, and of Hindi sentences with English words in them.
- Reading loan IDs and phone digits one at a time, and dates.
- Azure Speech for Kannada and Bengali, and as a fallback for English and Hindi.
- The benchmark on Render.
