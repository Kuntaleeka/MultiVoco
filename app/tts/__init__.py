"""Speech out. Owner: workstream C (Garlic). Importing this package registers it."""

from functools import cache

from app.core.registry import register
from app.tts.config import get_tts_settings
from app.tts.piper import PiperTTS

PIPER = "piper"


@cache
def piper() -> PiperTTS:
    """The one shared instance: loaded voices are costly, and the registry asks often."""
    return PiperTTS()


register("tts", PIPER, lambda lang: piper())


async def startup() -> None:
    """Called by app/pipeline/plugins.py before the app serves requests."""
    if get_tts_settings().tts_warm_up:
        await piper().warm_up()


__all__ = ["PIPER", "PiperTTS", "piper", "startup"]
