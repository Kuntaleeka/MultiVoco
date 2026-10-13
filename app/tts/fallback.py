"""Uses a second voice when the first fails, or is too slow to start."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

from app.core.interfaces import TTS, ProviderError
from app.core.languages import Lang

log = logging.getLogger(__name__)


class FallbackTTS:
    """Tries `primary`, and moves to `backup` only if no audio has gone out yet.

    Both must use the same sample rate for a language: the orchestrator announces the
    rate before the first piece arrives, so it cannot change part-way.
    """

    def __init__(self, primary: TTS, backup: TTS, first_audio_timeout_s: float) -> None:
        self._primary = primary
        self._backup = backup
        self._timeout = first_audio_timeout_s
        self.languages = set(primary.languages)

    def sample_rate(self, language: Lang) -> int:
        return self._primary.sample_rate(language)

    def _can_fall_back(self, language: Lang) -> bool:
        try:
            return language in self._backup.languages and self._backup.sample_rate(
                language
            ) == self._primary.sample_rate(language)
        except ProviderError:
            return False

    async def synthesize(self, text: str, language: Lang) -> AsyncIterator[bytes]:
        if not self._can_fall_back(language):
            async with aclosing(self._primary.synthesize(text, language)) as pieces:
                async for piece in pieces:
                    yield piece
            return

        async with aclosing(self._primary.synthesize(text, language)) as pieces:
            try:
                first = await asyncio.wait_for(anext(pieces, None), self._timeout)
            except TimeoutError:
                log.warning("primary TTS gave no audio in %.1f s, using the backup", self._timeout)
            except ProviderError as error:
                log.warning("primary TTS failed, using the backup: %s", error)
            else:
                # It has started. From here a failure is the caller's to see: switching
                # voices in the middle of a sentence would repeat or garble it.
                if first is not None:
                    yield first
                    async for piece in pieces:
                        yield piece
                return
        async with aclosing(self._backup.synthesize(text, language)) as pieces:
            async for piece in pieces:
                yield piece
