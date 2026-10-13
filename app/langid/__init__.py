"""Workstream A: language detection. Importing this registers the detector."""

from app.core.registry import DEFAULT, register
from app.langid.detector import BallotLanguageDetector, close_client

register("langid", DEFAULT, BallotLanguageDetector)


async def shutdown() -> None:
    await close_client()
