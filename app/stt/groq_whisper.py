"""Whisper on Groq. Not streaming: the utterance is buffered and sent at speech end.

There are no partial transcripts, and the whole transcription time lands after the
caller stops speaking. It is the fallback for English and Hindi and, until something
better is measured, the route for Kannada and Bengali.
"""

import asyncio
import io
import logging
import math
import wave

import httpx2 as httpx
import numpy as np

from app.core.config import get_settings
from app.core.interfaces import INPUT_SAMPLE_RATE, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.stt.base import QueueSTT
from app.stt.config import STTSettings, get_stt_settings

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0


def to_wav(pcm: bytes, sample_rate: int = INPUT_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buffer.getvalue()


async def transcribe(
    pcm: bytes,
    language: Lang | None,
    *,
    api_key: str | None = None,
    settings: STTSettings | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, float | None, str | None]:
    """Transcribe PCM16 mono 16 kHz. Returns (text, confidence, language Whisper heard).

    With language=None, Whisper detects the language itself.
    """
    s = settings or get_stt_settings()
    key = api_key if api_key is not None else get_settings().groq_api_key
    if not key:
        raise ProviderError("MULTIVOCO_GROQ_API_KEY is not set")
    data = {"model": s.groq_model, "response_format": "verbose_json", "temperature": "0"}
    if language is not None:
        data["language"] = language.value
    try:
        response = await (client or _get_client()).post(
            s.groq_url,
            headers={"Authorization": f"Bearer {key}"},
            data=data,
            files={"file": ("utterance.wav", to_wav(pcm), "audio/wav")},
            timeout=s.groq_timeout_seconds,
        )
    except httpx.HTTPError as error:
        raise ProviderError(f"Could not reach Groq: {error!r}") from error
    if response.status_code == 429:
        raise QuotaExceeded("Groq rate limit reached")
    if response.status_code != 200:
        raise ProviderError(f"Groq returned HTTP {response.status_code}: {response.text[:200]}")
    body = response.json()
    segments = body.get("segments")
    if not segments:
        return (body.get("text") or "").strip(), None, body.get("language")
    kept = [
        seg for seg in segments if seg.get("no_speech_prob", 0.0) <= s.whisper_max_no_speech_prob
    ]
    text = " ".join((seg.get("text") or "").strip() for seg in kept).strip()
    logprobs = [seg["avg_logprob"] for seg in kept if "avg_logprob" in seg]
    confidence = math.exp(sum(logprobs) / len(logprobs)) if logprobs else None
    return text, confidence, body.get("language")


class GroqWhisperSTT(QueueSTT):
    def __init__(
        self,
        api_key: str | None = None,
        settings: STTSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self._key = api_key if api_key is not None else get_settings().groq_api_key
        self._s = settings or get_stt_settings()
        self._client = client
        self._frames: list[bytes] = []
        self._tail: asyncio.Task | None = None  # keeps finals in the order of finish() calls

    async def start(self, language: Lang) -> None:
        if not self._key:
            raise ProviderError("MULTIVOCO_GROQ_API_KEY is not set")
        self._language = language

    async def send_audio(self, frame: bytes) -> None:
        self._frames.append(frame)

    async def finish(self) -> None:
        frames, self._frames = self._frames, []
        self._tail = asyncio.create_task(self._transcribe(b"".join(frames), self._tail))

    async def _transcribe(self, pcm: bytes, before: asyncio.Task | None) -> None:
        if before is not None:
            await asyncio.gather(before, return_exceptions=True)
        if not pcm or _rms(pcm) < self._s.whisper_min_rms:
            self._emit("", final=True)  # nothing was said: do not let Whisper invent it
            return
        try:
            text, confidence, _ = await transcribe(
                pcm, self._language, api_key=self._key, settings=self._s, client=self._client
            )
        except ProviderError as error:
            self._fail(error)
            return
        self._emit(text, final=True, confidence=confidence)

    async def close(self) -> None:
        if self._tail is not None:
            self._tail.cancel()
            await asyncio.gather(self._tail, return_exceptions=True)
        self._out.put_nowait(None)
