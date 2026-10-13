"""Write one spoken sample per language, for a listening check, and time each one.

    uv run python -m app.tts.sample                 # every language, to voices/samples/
    uv run python -m app.tts.sample --lang kn --runs 3

English and Hindi are spoken by Piper, Kannada and Bengali by Sarvam, as in a call.
Each sentence has an amount with paise, a date, and "EMI": the things most likely to be
read wrongly. A Sarvam sample costs about 80 characters of credit per run.
"""

import argparse
import asyncio
import statistics
import sys
import time
import wave
from pathlib import Path

from app.core.interfaces import ProviderError
from app.core.languages import Lang
from app.core.pools import shutdown_pools
from app.tts import piper, sarvam, shutdown

SENTENCES = {
    Lang.EN: "Your next EMI of 8450.50 rupees is due on 2026-11-05. Is there anything else?",
    Lang.HI: "आपकी अगली EMI 8450.50 रुपये की है, और 2026-11-05 को देय है। क्या मैं और कुछ मदद करूँ?",
    Lang.KN: "ನಿಮ್ಮ ಮುಂದಿನ EMI 11450.50 ರೂಪಾಯಿ, 2026-11-05 ರಂದು ಪಾವತಿಸಬೇಕು. ಇನ್ನೇನಾದರೂ ಸಹಾಯ ಬೇಕೇ?",
    Lang.BN: "আপনার পরের EMI 10400.50 টাকা, 2026-11-05 তারিখে দিতে হবে। আর কিছু সাহায্য লাগবে?",
}
VOICES = {Lang.EN: piper, Lang.HI: piper, Lang.KN: sarvam, Lang.BN: sarvam}


async def speak(lang: Lang) -> tuple[bytes, float, float]:
    """(audio, ms to first audio, ms in total)."""
    tts = VOICES[lang]()
    started = time.perf_counter()
    first, pcm = None, b""
    async for piece in tts.synthesize(SENTENCES[lang], lang):
        first = first or (time.perf_counter() - started) * 1000
        pcm += piece
    return pcm, first or 0.0, (time.perf_counter() - started) * 1000


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=[lang.value for lang in Lang], action="append")
    parser.add_argument("--runs", type=int, default=1, help="timed runs per language")
    parser.add_argument("--out", type=Path, default=Path("voices/samples"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    for lang in [Lang(code) for code in args.lang or [lang.value for lang in Lang]]:
        tts = VOICES[lang]()
        name = type(tts).__name__
        try:
            await speak(lang)  # the first run loads the voice or opens the connection
            runs = [await speak(lang) for _ in range(args.runs)]
        except ProviderError as error:
            print(f"{lang.value}: {name} failed: {error}")
            continue
        pcm = runs[-1][0]
        rate = tts.sample_rate(lang)
        path = args.out / f"{lang.value}_{name.removesuffix('TTS').lower()}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(pcm)
        print(
            f"{lang.value}: {name}, first audio {statistics.median(r[1] for r in runs):.0f} ms, "
            f"whole sentence {statistics.median(r[2] for r in runs):.0f} ms, "
            f"{len(pcm) / 2 / rate:.1f} s of audio, {path}"
        )
    await shutdown()
    shutdown_pools()


if __name__ == "__main__":
    asyncio.run(main())
