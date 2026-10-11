"""Detect the language of a WAV file, the way the pipeline does on a first utterance.

    uv run python -m scripts.langid_file clip.wav
    uv run python -m scripts.langid_file clip.wav --seconds 1.5 --json

--seconds uses only the start of the file, to see how a short utterance fares. The
output is the guess, its confidence (0.6 and up counts as sure), and each recogniser's
vote. Needs the Deepgram key in the environment or .env.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.core.interfaces import INPUT_SAMPLE_RATE, ProviderError
from app.langid.detector import BallotLanguageDetector, close_client
from scripts.stt_file import read_wav


async def detect_file(path: Path, seconds: float | None) -> dict:
    pcm = read_wav(path)
    if seconds is not None:
        pcm = pcm[: int(seconds * INPUT_SAMPLE_RATE) * 2]
    detector = BallotLanguageDetector()
    began = time.perf_counter()
    try:
        guess = await detector.detect_audio(pcm)
    finally:
        await close_client()
    return {
        "file": str(path),
        "audio_ms": round(len(pcm) / 2 / INPUT_SAMPLE_RATE * 1000),
        "lang": guess.lang.value if guess.lang else None,
        "confidence": guess.confidence,
        "detect_ms": round((time.perf_counter() - began) * 1000),
        "votes": [
            {"code": v.code, "confidence": round(v.confidence, 3), "text": v.text}
            for v in detector.last_votes
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--seconds", type=float, help="use only this much of the start")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args()
    try:
        result = asyncio.run(detect_file(args.wav, args.seconds))
    except ProviderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    sure = "sure" if result["confidence"] >= 0.6 else "not sure: the caller would be asked"
    print(f"{result['lang']}  confidence {result['confidence']} ({sure}), {result['detect_ms']} ms")
    for vote in result["votes"]:
        print(f"  {vote['code']:6} {vote['confidence']:.2f}  {vote['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
