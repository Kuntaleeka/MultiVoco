# Language detection: what was tried and what works

Owner: Onion. Measured on 2026-10-11.

**Read this first:** every figure here comes from five clips, one per language, spoken
by the Mac's built-in voices. That is enough to choose an approach and not enough to
claim an accuracy. Workstream G's language ID suite, on recorded speech, is what
decides whether this is good enough.

## The plan's approach does not work

The plan was to let Whisper detect the language of the first utterance. It cannot tell
Kannada or Bengali from Hindi:

| Clip | Whisper (turbo) said | Whisper (large-v3) said | Deepgram's own detection said |
|---|---|---|---|
| English | English | English | English |
| Hindi | Hindi | Hindi | Hindi |
| Hinglish | Hindi | Hindi | English |
| Kannada | **Hindi** | **Telugu** | **Hindi** |
| Bengali | **Hindi** | **Hindi** | **Hindi** |

Whisper was confident each time, so its score gives no warning. It also wrote the
Kannada and Bengali speech out in Devanagari, so the script check on later turns would
not have caught the mistake either.

## What works: a ballot of recognisers

Deepgram transcribes each of our languages well when told which one to expect. So the
first utterance is transcribed three times at once, as Hindi/English (`multi`), as
Kannada, and as Bengali, and the recogniser that is most confident in its own
transcript wins. The others are forcing the sounds into the wrong language, and their
confidence shows it.

Confidence of each recogniser, with the winner in bold:

| Clip | `multi` | `kn` | `bn` | Result |
|---|---|---|---|---|
| English, full sentence | **1.00** | 0.54 | 0.80 | English |
| Hindi, full sentence | **1.00** | 0.82 | 0.59 | Hindi |
| Hinglish, full sentence | **0.92** | 0.64 | 0.72 | Hindi |
| Kannada, full sentence | 0.81 | **0.97** | 0.79 | Kannada |
| Bengali, full sentence | 0.76 | 0.53 | **0.95** | Bengali |

The winner's lead over the runner-up is the evidence. A lead of 0.10 or more counts as
sure (`MULTIVOCO_LANGID_SURE_MARGIN`). On these clips the smallest winning lead was 0.16.

Hindi and English share one recogniser. Devanagari in its transcript means Hindi,
otherwise English. The two also share one route through the pipeline, so mixing them
up costs little.

## Short openings are the weak point

| How much speech | Result on the five clips |
|---|---|
| A full sentence (2.4 s and up) | 5 of 5 right and sure |
| 1.2 s | 4 right and sure. Bengali too close to call, so the caller would be asked. |
| 0.7 s (one word) | **3 of 5 wrong, and 2 of those confidently so** |

"नमस्ते" and "नमस्कार" are greetings in all three of Hindi, Kannada, and Bengali, so one
word of it is not evidence of anything. The pipeline therefore does not trust a guess
from less than 1 second of speech (`PipelineConfig.min_detect_ms`), and asks the caller
"English, Hindi, Kannada, or Bangla?" instead. The answer is read from the same ballot:
a language named in the winning transcript wins, otherwise the winning recogniser's
language does.

## Speed

Detection adds to the first reply only. Measured through the whole server, the first
reply began 2.2 to 4.8 seconds after the caller stopped, against about 0.8 seconds for
a call where the language was picked by hand. Later turns are not affected.

The cost is Deepgram's batch endpoint, which took between 0.25 and 2 seconds for one
request on the same audio from one run to the next. Three run at once, and the ballot
waits for the slowest.

Already done to help:

- The connections are opened while the greeting plays, not after the caller speaks.
- The winning transcript is used as the caller's first turn, so the utterance is not
  transcribed a second time. This also fixed a bug: replaying the buffered audio into
  the streaming recogniser in one burst returned only the first two words.

The real fix, not built yet: run the three recognisers as live streams while the
caller is still speaking, instead of sending the recording afterwards. Streaming
answers about 250 ms after the caller stops, every time, so detection would add almost
nothing. The streaming client already exists (`app/stt/deepgram.py`). It needs a small
addition to the detector interface and to the orchestrator.

## What a wrong guess costs

- **Between Hindi and English:** nothing. One route serves both.
- **Kannada or Bengali heard as something else:** the caller gets a reply in the wrong
  language. They can change it with the language picker. The script check on later
  turns will not rescue this reliably, because a recogniser set to the wrong language
  writes in its own script.
- **Not sure:** one extra exchange, the language question.

## Repeating it

```sh
uv run python -m scripts.langid_file clip.wav
uv run python -m scripts.langid_file clip.wav --seconds 1.2 --json
```

It prints the guess, its confidence, and every recogniser's vote with its transcript.
