# Speech to text: what was measured

Owner: Onion.

## First live check, 2026-10-11

This is a smoke test, not an evaluation. It shows that both clients work against the
real services and gives a first look at speed and accuracy. **Do not quote these as
accuracy figures**: there is one clip per language, and the "caller" is the Mac's
built-in voice, which is cleaner and more regular than a person on a phone. Word error
rates come from workstream G, on recorded speech.

Setup: clips sent in 32 ms frames at real-time speed from a laptop in India, over home
broadband. Deepgram `nova-3` streaming. Groq `whisper-large-v3-turbo`.

| Clip | What was said | Deepgram | Groq Whisper |
|---|---|---|---|
| English | Hello, I wanted to ask when my next EMI is due, and how much it is. | Hello. I wanted to ask when my next EMI is due, and how much it is. | Hello, I wanted to ask when my next EMI is due. And how much it is. |
| Hindi | नमस्ते, मेरी अगली ई एम आई कब है और कितनी है? | नमस्ते, मेरी अगली EMI कब है और कितनी है? | नमस्ते, मेरी अगली EMI कब है और कितनी है? |
| Hinglish | मेरा next EMI कब due है? Please बता दीजिए। | मेरा next Amy कब due है? Please बता दीजिए. | मेरा next year में कब डू है? प्लीज बता दीजिये. |
| Kannada | ನಮಸ್ಕಾರ, ನನ್ನ ಮುಂದಿನ ಇ ಎಂ ಐ ಯಾವಾಗ? | ನಮಸ್ಕಾರ ನನ್ನ ಮುಂದಿನ ಇ ಎಂ ಐ ಯಾವಾಗ | ನಮಸ್ಕಾರ ನನ್ನ ಮುಂದಿನ ಇಮಾಯಾವಾಗ. |
| Bengali | নমস্কার, আমার পরের ই এম আই কবে? | নমস্কার, আমার পরের এম আই কবে? | নমোম্যামার পারের এমাই কোবে |

| | Deepgram | Groq Whisper |
|---|---|---|
| Final transcript after the caller stops | 230 to 250 ms | 350 to 400 ms |
| First partial transcript | about 1.25 s into the speech, then about one a second | none: it is not streaming |
| Connection set-up at the start of a call | about 700 ms, once | none |

What this suggests, to be confirmed on real recordings:

- **Deepgram handles all four languages in streaming**, including Kannada and Bengali,
  and was at least as accurate as Whisper on every clip. On Kannada, Bengali, and
  Hinglish it was clearly better. If that holds, Kannada and Bengali should move from
  Whisper to Deepgram in the routing table, which also gives them live partial
  transcripts. To try it now: `MULTIVOCO_ROUTE_OVERRIDES={"kn": {"stt": "deepgram"}, "bn": {"stt": "deepgram"}}`.
- **"EMI" is the hard word.** Spoken inside a Hindi sentence it came back as "Amy"
  from Deepgram and "year" from Whisper. Deepgram has a keyword-boosting option worth
  trying for the loan vocabulary (EMI, tenure, foreclosure, and so on).
- **Whisper is a usable fallback** for English and Hindi, and weak for the rest.

## A bug the live check found

Deepgram's default end-of-speech setting (`endpointing`, 10 ms) finalised an empty
segment over the opening "Hello," and the word never appeared. The VAD does our
endpointing, so the client now sends `endpointing=false`, and the word is kept. The
tests against the local stand-in could not have caught this.

## Where the time goes in a turn

Through the whole server, with the real VAD and Deepgram and everything else mocked,
the reply began about 830 ms after the caller stopped:

| Part | Time |
|---|---|
| VAD end-of-speech silence (`MULTIVOCO_VAD_END_SILENCE_MS`) | 500 ms |
| Deepgram answering the flush | about 250 ms |
| Everything else in the server (mocked agent and TTS) | under 100 ms |

Two ways to cut this, for the tuning stage:

- Shorten the end-of-speech silence. It trades directly against cutting callers off
  when they pause.
- Ask Deepgram to flush as soon as the silence starts, not when it ends, so the two
  waits overlap instead of adding up.

## Repeating it

```sh
uv run python -m scripts.stt_file clip.wav --lang kn --provider deepgram --json
uv run python -m scripts.stt_file clip.wav --lang kn --provider groq_whisper --json
```

The test clips are not in the repository. They were made on a Mac with
`say -v <voice> -o clip.aiff "<text>"` and converted with
`afconvert -f WAVE -d LEI16@16000 -c 1 clip.aiff clip.wav`, using the voices Lekha
(Hindi), Soumya (Kannada), and Piya (Bengali). `samples/` is git-ignored.
