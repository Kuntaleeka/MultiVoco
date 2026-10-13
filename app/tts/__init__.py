"""Speech out. Owner: workstream C (Garlic). Importing this package registers it."""

from functools import cache

from app.core.interfaces import TTS
from app.core.registry import register
from app.tts.config import get_tts_settings
from app.tts.fallback import FallbackTTS
from app.tts.piper import PiperTTS
from app.tts.sarvam import SarvamTTS, close_client

PIPER = "piper"
SARVAM = "sarvam"


@cache
def piper() -> PiperTTS:
    """The one shared instance: loaded voices are costly, and the registry asks often."""
    return PiperTTS()


@cache
def sarvam() -> SarvamTTS:
    """Shared too, so the day's character count covers every call."""
    return SarvamTTS()


@cache
def piper_with_fallback() -> TTS:
    """What the `piper` route gets: Piper, with Sarvam behind it when there is a key."""
    settings = get_tts_settings()
    if settings.sarvam_api_key and settings.piper_fallback_after_s > 0:
        return FallbackTTS(piper(), sarvam(), settings.piper_fallback_after_s)
    return piper()


register("tts", PIPER, lambda lang: piper_with_fallback())
register("tts", SARVAM, lambda lang: sarvam())


async def startup() -> None:
    """Called by app/pipeline/plugins.py before the app serves requests."""
    if get_tts_settings().tts_warm_up:
        await piper().warm_up()


async def shutdown() -> None:
    await close_client()


__all__ = ["PIPER", "SARVAM", "PiperTTS", "SarvamTTS", "piper", "sarvam", "shutdown", "startup"]
