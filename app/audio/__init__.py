"""Workstream A: voice activity detection. Importing this registers the Silero VAD."""

from app.audio.vad import SileroVAD, warm_up
from app.core.registry import DEFAULT, register

register("vad", DEFAULT, SileroVAD)


async def startup() -> None:
    await warm_up()
