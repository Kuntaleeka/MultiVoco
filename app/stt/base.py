import asyncio
from collections.abc import AsyncIterator

from app.core.interfaces import Transcript
from app.core.languages import Lang


class QueueSTT:
    """Shared plumbing: transcripts, or one error, go into a queue that events() drains."""

    languages = set(Lang)

    def __init__(self) -> None:
        self._out: asyncio.Queue[Transcript | Exception | None] = asyncio.Queue()
        self._language = Lang.EN

    def _emit(self, text: str, *, final: bool, confidence: float | None = None) -> None:
        self._out.put_nowait(
            Transcript(text, final=final, language=self._language, confidence=confidence)
        )

    def _fail(self, error: Exception) -> None:
        self._out.put_nowait(error)

    async def events(self) -> AsyncIterator[Transcript]:
        while True:
            item = await self._out.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
