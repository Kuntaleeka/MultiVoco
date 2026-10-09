import asyncio
import json
from typing import Any

import pytest

from app.core import registry
from app.core.config import get_settings
from app.core.interfaces import CallContext, LangGuess
from app.core.languages import Lang
from app.core.mocks import (
    InMemoryTraceSink,
    MockAgent,
    MockLanguageDetector,
    MockSTT,
    MockTTS,
    frames,
    silence,
    tone,
)
from app.pipeline.config import PipelineConfig
from app.pipeline.session import CallSession


class FakeSocket:
    """Records everything the session sends. Tests drive the session directly."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any] | bytes] = []
        self.close_code: int | None = None
        self._changed = asyncio.Event()

    async def receive(self) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {}

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))
        self._changed.set()

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)
        self._changed.set()

    async def close(self, code: int = 1000) -> None:
        self.close_code = code
        self._changed.set()

    @property
    def messages(self) -> list[dict[str, Any]]:
        return [item for item in self.sent if isinstance(item, dict)]

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == kind]

    def audio_bytes(self) -> int:
        return sum(len(item) for item in self.sent if isinstance(item, bytes))

    async def wait_for(self, predicate, within: float = 3.0):
        """Wait until predicate(self) is truthy, and return its value."""

        async def poll():
            while not (value := predicate(self)):
                self._changed.clear()
                await self._changed.wait()
            return value

        return await asyncio.wait_for(poll(), within)

    async def wait_for_state(self, value: str, count: int = 1, within: float = 3.0):
        def seen(sock: "FakeSocket"):
            return len([m for m in sock.of_type("state") if m["value"] == value]) >= count

        await self.wait_for(seen, within)


class Rig:
    """A session on a fake socket, with every component replaceable before start."""

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.socket = FakeSocket()
        self.sink = InMemoryTraceSink()
        self.agents: list[Any] = []
        self.tts = MockTTS(ms_per_char=2)  # short audio, so turns finish fast
        self.stt_script: dict[Lang, list[str]] = {}
        self.detector = MockLanguageDetector()
        self.agent_class = MockAgent
        self.config = PipelineConfig(position_timeout_s=0.2, playback_margin_s=0.01)
        self.settings = get_settings()
        self.session: CallSession | None = None
        self.use("trace_sink", lambda: self.sink)
        self.use("tts", lambda lang: self.tts)
        self.use("stt", lambda lang: MockSTT(self.stt_script.get(lang)))
        self.use("langid", lambda: self.detector)
        self.use("agent", self._make_agent)

    def _make_agent(self, ctx: CallContext):
        agent = self.agent_class(ctx)
        self.agents.append(agent)
        return agent

    def use(self, kind: str, factory) -> None:
        self.monkeypatch.setitem(registry._factories[kind], registry.MOCK, factory)

    def detect(self, lang: Lang | None, confidence: float = 0.95) -> None:
        self.detector = MockLanguageDetector(LangGuess(lang, confidence))

    async def start(self, lang: str = "en", **config) -> CallSession:
        if config:
            self.config = PipelineConfig(**{**self.config.__dict__, **config})
        self.session = CallSession(self.socket, self.config, self.settings)
        await self.session.handle_message(json.dumps({"type": "start", "lang": lang}))
        return self.session

    async def send(self, **message) -> None:
        await self.session.handle_message(json.dumps(message))

    async def audio(self, pcm: bytes) -> None:
        for frame in frames(pcm):
            await self.session.handle_audio(frame)
            await asyncio.sleep(0)

    async def say(self, ms: int = 600) -> None:
        """One utterance: speech, then enough silence for the VAD to end it."""
        await self.audio(tone(ms) + silence(400))

    async def greeted(self) -> None:
        """Wait until the greeting has finished and the session is listening."""
        await self.socket.wait_for_state("listening")

    @property
    def agent(self):
        return self.agents[0]

    def agent_final(self, turn_id: int):
        """A predicate for socket.wait_for: the agent's final transcript for a turn."""

        def find(sock: FakeSocket):
            for m in sock.of_type("transcript"):
                if m["role"] == "agent" and m["final"] and m["turn_id"] == turn_id:
                    return m
            return None

        return find

    async def reply(self, turn_id: int) -> dict[str, Any]:
        """Wait for a reply to finish, and return its final transcript message."""
        message = await self.socket.wait_for(self.agent_final(turn_id))
        await asyncio.sleep(0)
        return message

    def trace(self, turn_id: int):
        return next(t for t in self.sink.turns if t.idx == turn_id)

    def spoken(self) -> list[str]:
        return [m.content for m in self.agent.history if m.role == "assistant"]


@pytest.fixture
async def rig(monkeypatch):
    import app.core.mocks  # noqa: F401  (make sure the mock factories exist to replace)

    rig = Rig(monkeypatch)
    yield rig
    if rig.session is not None:
        await rig.session.teardown()
        leftover = [t for t in rig.session._tasks if not t.done()]
        assert not leftover, f"tasks still running after teardown: {leftover}"
