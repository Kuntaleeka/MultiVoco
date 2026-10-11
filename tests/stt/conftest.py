import asyncio
import http
import json
from urllib.parse import parse_qs, urlparse

import pytest
import websockets

from app.core.interfaces import Transcript
from app.core.mocks import tone
from app.stt.config import STTSettings


def results(text: str, *, final: bool, from_finalize: bool = False, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "type": "Results",
            "is_final": final,
            "speech_final": final,
            "from_finalize": from_finalize,
            "channel": {"alternatives": [{"transcript": text, "confidence": confidence}]},
        }
    )


class FakeDeepgram:
    """A local WebSocket server that behaves like Deepgram's streaming endpoint.

    For each utterance it sends: an interim result on the first audio frame, a
    finalised first segment on the fourth, and the rest when asked to Finalize.
    """

    def __init__(self) -> None:
        self.utterances = [("When is my", "next EMI due?")]
        self.reject_with: http.HTTPStatus | None = None
        self.ignore_finalize = False
        self.drop_after_frames: int | None = None  # close the socket after this many frames
        self.connections = 0
        self.queries: list[dict[str, str]] = []
        self.authorization: list[str | None] = []
        self.control: list[str] = []
        self.frames = 0
        self._utterance = 0
        self._server = None
        self.url = ""

    async def __aenter__(self) -> "FakeDeepgram":
        self._server = await websockets.serve(
            self._handle, "127.0.0.1", 0, process_request=self._gate
        )
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/listen"
        return self

    async def __aexit__(self, *exc) -> None:
        self._server.close()
        await self._server.wait_closed()

    def _gate(self, connection, request):
        self.authorization.append(request.headers.get("Authorization"))
        self.queries.append({k: v[0] for k, v in parse_qs(urlparse(request.path).query).items()})
        if self.reject_with is not None:
            return connection.respond(self.reject_with, "no")
        return None

    async def _handle(self, ws) -> None:
        self.connections += 1
        in_utterance = 0
        async for message in ws:
            first, rest = self.utterances[self._utterance % len(self.utterances)]
            if isinstance(message, bytes):
                self.frames += 1
                in_utterance += 1
                if in_utterance == 1:
                    await ws.send(results(first.split()[0], final=False))
                elif in_utterance == 4:
                    await ws.send(results(first, final=True))
                if self.drop_after_frames is not None and self.frames >= self.drop_after_frames:
                    self.drop_after_frames = None
                    await ws.close(1011, "simulated failure")
                    return
                continue
            kind = json.loads(message)["type"]
            self.control.append(kind)
            if kind == "Finalize" and not self.ignore_finalize:
                text = rest if in_utterance else ""
                await ws.send(results(text, final=True, from_finalize=True, confidence=0.7))
                self._utterance += 1
                in_utterance = 0
            elif kind == "CloseStream":
                return


@pytest.fixture
async def deepgram():
    async with FakeDeepgram() as server:
        yield server


@pytest.fixture
def stt_settings(deepgram) -> STTSettings:
    return STTSettings(
        deepgram_url=deepgram.url,
        deepgram_keepalive_seconds=30,
        finalize_timeout_seconds=1.0,
        connect_timeout_seconds=2.0,
    )


async def collect_until_final(stt, within: float = 3.0) -> list[Transcript]:
    """Read events() up to and including the next final transcript."""
    seen: list[Transcript] = []

    async def read() -> None:
        async for transcript in stt.events():
            seen.append(transcript)
            if transcript.final:
                return

    await asyncio.wait_for(read(), within)
    return seen


FRAME = tone(32)  # one 512-sample frame that is clearly not silence
QUIET = bytes(1024)
