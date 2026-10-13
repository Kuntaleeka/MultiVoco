"""Transcribe a WAV file with a real STT provider, the way the pipeline would.

    uv run python -m scripts.stt_file clip.wav --lang hi
    uv run python -m scripts.stt_file clip.wav --lang kn --provider groq_whisper
    uv run python -m scripts.stt_file clip.wav --lang en --json

The file is sent in 32 ms frames at real-time speed, so the timings are what a caller
would see. --fast sends it at eight times real time instead. The last line of output is the
final transcript, or a JSON object with --json.

Needs the provider's key in the environment or .env. Any 16-bit WAV is accepted and
converted to 16 kHz mono.
"""

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

from app.core.interfaces import FRAME_MS, INPUT_SAMPLE_RATE, ProviderError
from app.core.languages import Lang
from app.core.mocks import frames


def read_wav(path: Path) -> bytes:
    """A 16-bit PCM WAV at any rate and channel count -> PCM16 mono at 16 kHz."""
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != INPUT_SAMPLE_RATE:
        samples = resample(samples, rate, INPUT_SAMPLE_RATE)
    return np.clip(np.round(samples), -32768, 32767).astype("<i2").tobytes()


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Low-pass (when going down) and then interpolate. Good enough for speech."""
    if target < rate:
        taps = 127
        n = np.arange(taps) - (taps - 1) / 2
        cutoff = 0.45 * target / rate  # as a fraction of the input sample rate
        kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        samples = np.convolve(samples, kernel / kernel.sum(), mode="same")
    positions = np.arange(int(len(samples) * target / rate)) * (rate / target)
    return np.interp(positions, np.arange(len(samples)), samples)


def make_stt(provider: str):
    if provider == "deepgram":
        from app.stt.deepgram import DeepgramSTT

        return DeepgramSTT()
    from app.stt.groq_whisper import GroqWhisperSTT

    return GroqWhisperSTT()


async def transcribe_file(path: Path, lang: Lang, provider: str, fast: bool, quiet: bool) -> dict:
    pcm = read_wav(path)
    stt = make_stt(provider)
    began = time.perf_counter()
    await stt.start(lang)
    connected = time.perf_counter()
    result: dict = {
        "file": str(path),
        "provider": provider,
        "lang": lang.value,
        "audio_ms": round(len(pcm) / 2 / INPUT_SAMPLE_RATE * 1000),
        "connect_ms": round((connected - began) * 1000),
        "first_partial_ms": None,
    }

    async def read() -> None:
        async for transcript in stt.events():
            if not transcript.final:
                if result["first_partial_ms"] is None:
                    result["first_partial_ms"] = round((time.perf_counter() - connected) * 1000)
                if not quiet:
                    print(f"  … {transcript.text}", file=sys.stderr)
                continue
            result["text"] = transcript.text
            result["confidence"] = transcript.confidence
            return

    reader = asyncio.create_task(read())
    try:
        for frame in frames(pcm):
            await stt.send_audio(frame)
            # Even --fast is paced, at eight times real time: a streaming recogniser
            # treats audio arriving faster than it can take it as a failure.
            await asyncio.sleep(FRAME_MS / 1000 / (8 if fast else 1))
        speech_end = time.perf_counter()
        await stt.finish()
        await asyncio.wait_for(reader, 30)
        # The number that matters for latency: caller stops speaking -> final transcript.
        result["final_after_speech_end_ms"] = round((time.perf_counter() - speech_end) * 1000)
    finally:
        reader.cancel()
        await stt.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--lang", choices=[lang.value for lang in Lang], default="en")
    parser.add_argument("--provider", choices=["deepgram", "groq_whisper"], default="deepgram")
    parser.add_argument("--fast", action="store_true", help="do not pace the audio in real time")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args()
    try:
        result = asyncio.run(
            transcribe_file(args.wav, Lang(args.lang), args.provider, args.fast, args.json)
        )
    except ProviderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(
            f"{result['provider']} {result['lang']}: {result['audio_ms']} ms of audio, "
            f"final {result['final_after_speech_end_ms']} ms after speech end",
            file=sys.stderr,
        )
        print(result["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
