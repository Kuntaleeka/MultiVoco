"""Moves a call from one STT to another when the first one fails."""

import asyncio
import logging
from collections.abc import Callable

from app.core.interfaces import STT, ProviderError
from app.core.languages import Lang
from app.stt.base import QueueSTT

log = logging.getLogger(__name__)


class FallbackSTT(QueueSTT):
    """Uses `primary` until it raises ProviderError, then `make_secondary()` for the rest
    of the call. The utterance in progress is replayed into the secondary, so the
    caller does not have to repeat it. If the secondary fails too, the error surfaces.
    """

    def __init__(self, primary: STT, make_secondary: Callable[[], STT]) -> None:
        super().__init__()
        self._active: STT = primary
        self._make_secondary: Callable[[], STT] | None = make_secondary
        self._pump: asyncio.Task | None = None
        self._utterance: list[bytes] = []  # audio since the last final transcript
        self._owed = False  # finish() was called and its final has not arrived
        self._switching = asyncio.Lock()

    @property
    def on_fallback(self) -> bool:
        return self._make_secondary is None

    async def start(self, language: Lang) -> None:
        self._language = language
        try:
            await self._active.start(language)
        except ProviderError as error:
            await self._switch(error)
            return
        self._pump = asyncio.create_task(self._forward(self._active))

    async def send_audio(self, frame: bytes) -> None:
        self._utterance.append(frame)
        try:
            await self._active.send_audio(frame)
        except ProviderError as error:
            await self._switch(error)

    async def finish(self) -> None:
        self._owed = True
        try:
            await self._active.finish()
        except ProviderError as error:
            await self._switch(error)

    async def _forward(self, stt: STT) -> None:
        try:
            async for transcript in stt.events():
                if transcript.final:
                    self._owed = False
                    self._utterance = []
                self._out.put_nowait(transcript)
        except ProviderError as error:
            if stt is self._active:
                await self._switch(error, from_pump=True)

    async def _switch(self, error: ProviderError, from_pump: bool = False) -> None:
        async with self._switching:
            if self._make_secondary is None:
                self._fail(error)  # already on the fallback: nothing left to try
                return
            log.warning("STT failed, falling back: %s", error)
            make, self._make_secondary = self._make_secondary, None
            failed, pump = self._active, self._pump
            if pump is not None and not from_pump:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
            try:
                await failed.close()
            except ProviderError:
                pass
            try:
                self._active = make()
                await self._active.start(self._language)
                self._pump = asyncio.create_task(self._forward(self._active))
                for frame in self._utterance:
                    await self._active.send_audio(frame)
                if self._owed:
                    await self._active.finish()
            except ProviderError as second:
                self._fail(second)

    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            await asyncio.gather(self._pump, return_exceptions=True)
        try:
            await self._active.close()
        except ProviderError:
            pass
        self._out.put_nowait(None)
