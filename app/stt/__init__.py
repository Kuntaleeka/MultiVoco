"""Workstream A: speech to text. Importing this registers the STT providers."""

from app.core.config import get_settings
from app.core.interfaces import STT
from app.core.languages import Lang
from app.core.registry import register
from app.stt.deepgram import DeepgramSTT
from app.stt.fallback import FallbackSTT
from app.stt.groq_whisper import GroqWhisperSTT, close_client


def _deepgram(lang: Lang) -> STT:
    """Deepgram, backed by Whisper when a Groq key is there to back it with."""
    if get_settings().groq_api_key:
        return FallbackSTT(DeepgramSTT(strict=True), GroqWhisperSTT)
    return DeepgramSTT()


register("stt", "deepgram", _deepgram)
register("stt", "groq_whisper", lambda lang: GroqWhisperSTT())


async def shutdown() -> None:
    await close_client()
