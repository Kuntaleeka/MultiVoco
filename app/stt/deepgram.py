"""Deepgram streaming STT over its WebSocket API.

Deepgram cuts a stream into finalised segments on its own schedule. We do the
endpointing ourselves (the VAD), so segments are joined until finish() asks Deepgram to
flush, and the answer to that flush closes the utterance.
"""

import asyncio
import json
import logging
import time
from collections import deque
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from app.core.config import get_settings
from app.core.interfaces import INPUT_SAMPLE_RATE, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.stt.base import QueueSTT
from app.stt.config import STTSettings, get_stt_settings

log = logging.getLogger(__name__)

_QUOTA_STATUSES = {402, 429}


class _Flush:
    """One finish() waiting for its final transcript."""

    done = False


class DeepgramSTT(QueueSTT):
    def __init__(
        self,
        api_key: str | None = None,
        settings: STTSettings | None = None,
        strict: bool = False,
    ) -> None:
        """`strict` is for use behind a fallback: when Deepgram does not answer a flush
        in time, raise instead of passing off a transcript that may be cut short."""
        super().__init__()
        self._key = api_key if api_key is not None else get_settings().deepgram_api_key
        self._s = settings or get_stt_settings()
        self._strict = strict
        # Audio waits here for the writer task, so a stalled connection can never hold
        # up the caller's audio loop. If it fills, Deepgram is not keeping up.
        self._outbox: asyncio.Queue[bytes | str] = asyncio.Queue(
            maxsize=self._s.deepgram_max_queued_frames
        )
        self._broken: ProviderError | None = None
        self._ws: websockets.ClientConnection | None = None
        self._tasks: list[asyncio.Task] = []
        self._closing = False
        self._last_sent = 0.0
        self._segments: list[str] = []  # finalised pieces of the current utterance
        self._confidences: list[float] = []
        self._interim = ""  # the latest not-yet-final piece
        self._flushes: deque[_Flush] = deque()
        self._writer: asyncio.Task | None = None

    # --- Connection --------------------------------------------------------------

    def _url(self) -> str:
        language = self._s.deepgram_languages.get(self._language.value, self._language.value)
        query = {
            "model": self._s.deepgram_model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": INPUT_SAMPLE_RATE,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "endpointing": self._s.deepgram_endpointing,
        }
        return f"{self._s.deepgram_url}?{urlencode(query)}"

    async def _connect(self) -> None:
        if not self._key:
            raise ProviderError("MULTIVOCO_DEEPGRAM_API_KEY is not set")
        try:
            self._ws = await websockets.connect(
                self._url(),
                additional_headers={"Authorization": f"Token {self._key}"},
                open_timeout=self._s.connect_timeout_seconds,
            )
        except InvalidStatus as error:
            status = error.response.status_code
            kind = QuotaExceeded if status in _QUOTA_STATUSES else ProviderError
            raise kind(f"Deepgram refused the connection: HTTP {status}") from error
        except (OSError, TimeoutError, websockets.WebSocketException) as error:
            raise ProviderError(f"Could not reach Deepgram: {error}") from error
        self._last_sent = time.monotonic()
        self._tasks = [t for t in self._tasks if not t.done()]
        self._tasks.append(asyncio.create_task(self._read(self._ws)))
        self._tasks.append(asyncio.create_task(self._keep_alive(self._ws)))

    async def _send(self, data: bytes | str) -> None:
        """Send, reconnecting once if the connection has dropped."""
        for attempt in (1, 2):
            if self._ws is None:
                await self._connect()
            try:
                assert self._ws is not None
                await self._ws.send(data)
                self._last_sent = time.monotonic()
                return
            except ConnectionClosed as error:
                self._ws = None
                if attempt == 2:
                    raise ProviderError(f"Deepgram connection lost: {error}") from error

    def _enqueue(self, item: bytes | str) -> None:
        if self._broken is not None:
            raise self._broken
        try:
            self._outbox.put_nowait(item)
        except asyncio.QueueFull:
            self._broken = ProviderError("Deepgram is not taking audio fast enough")
            raise self._broken from None

    async def _write(self) -> None:
        while True:
            item = await self._outbox.get()
            try:
                await self._send(item)
            except ProviderError as error:
                self._broken = error
                self._fail(error)
                return

    async def _keep_alive(self, ws: websockets.ClientConnection) -> None:
        interval = self._s.deepgram_keepalive_seconds
        try:
            while True:
                await asyncio.sleep(interval / 2)
                if time.monotonic() - self._last_sent >= interval:
                    await ws.send(json.dumps({"type": "KeepAlive"}))
                    self._last_sent = time.monotonic()
        except ConnectionClosed:
            return

    # --- Results -----------------------------------------------------------------

    def _text(self) -> str:
        return " ".join(part for part in (*self._segments, self._interim) if part)

    def _complete(self, flush: _Flush) -> None:
        """Close the current utterance: emit its one final transcript."""
        if flush.done:
            return
        flush.done = True
        confidence = sum(self._confidences) / len(self._confidences) if self._confidences else None
        self._emit(self._text(), final=True, confidence=confidence)
        self._segments, self._confidences, self._interim = [], [], ""

    async def _read(self, ws: websockets.ClientConnection) -> None:
        try:
            async for raw in ws:
                message = json.loads(raw)
                if message.get("type") != "Results":
                    continue
                alternatives = message.get("channel", {}).get("alternatives") or [{}]
                text = (alternatives[0].get("transcript") or "").strip()
                if not message.get("is_final"):
                    self._interim = text
                    if text:
                        self._emit(self._text(), final=False)
                    continue
                self._interim = ""
                if text:
                    self._segments.append(text)
                    self._confidences.append(float(alternatives[0].get("confidence") or 0.0))
                if message.get("from_finalize") and self._flushes:
                    self._complete(self._flushes.popleft())
                elif text:
                    self._emit(self._text(), final=False)
        except ConnectionClosed:
            pass
        except Exception as error:  # a malformed message must not kill the call silently
            log.exception("Deepgram reader failed")
            self._fail(ProviderError(f"Deepgram reader failed: {error}"))
            return
        finally:
            if self._ws is ws:
                self._ws = None  # the next send reconnects
        # The connection ended. Whoever is waiting for a final gets what arrived.
        while self._flushes:
            self._complete(self._flushes.popleft())
        if not self._closing:
            log.warning("Deepgram closed the connection: %s", ws.close_reason or ws.close_code)

    async def _give_up_waiting(self, flush: _Flush) -> None:
        await asyncio.sleep(self._s.finalize_timeout_seconds)
        if flush.done:
            return
        log.warning("Deepgram did not answer Finalize in time")
        if flush in self._flushes:
            self._flushes.remove(flush)
        if self._strict:
            # What has arrived may be missing its last words. Let the fallback redo it.
            flush.done = True
            self._broken = ProviderError("Deepgram did not answer a flush in time")
            self._fail(self._broken)
        else:
            self._complete(flush)

    # --- STT interface -----------------------------------------------------------

    async def start(self, language: Lang) -> None:
        self._language = language
        await self._connect()
        self._writer = asyncio.create_task(self._write())

    async def send_audio(self, frame: bytes) -> None:
        self._enqueue(frame)

    async def finish(self) -> None:
        flush = _Flush()
        self._flushes.append(flush)
        try:
            self._enqueue(json.dumps({"type": "Finalize"}))
        except ProviderError:
            self._flushes.remove(flush)
            if not self._strict:
                self._complete(flush)  # the contract: every finish() gets one final
            raise
        self._tasks.append(asyncio.create_task(self._give_up_waiting(flush)))

    async def close(self) -> None:
        self._closing = True
        ws, self._ws = self._ws, None
        tasks = [*self._tasks, *([self._writer] if self._writer is not None else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks = []
        if ws is not None:
            try:
                async with asyncio.timeout(self._s.close_timeout_seconds):
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    await ws.close()
            except (ConnectionClosed, OSError, TimeoutError):
                ws.transport.abort()  # do not make the end of a call wait on a slow peer
        self._out.put_nowait(None)
