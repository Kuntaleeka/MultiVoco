"""Deterministic stand-ins for every component, so each workstream can build alone.

No network, no models, no keys. Scripted audio is "tone means speech, zeros mean
silence", which the energy-based MockVAD can segment.
"""

import array
import asyncio
import math
from collections.abc import AsyncIterator

from app.core.interfaces import (
    FRAME_BYTES,
    INPUT_SAMPLE_RATE,
    AgentEvent,
    CallContext,
    Done,
    LangGuess,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolSpec,
    Transcript,
    VADEvent,
    VADEventKind,
)
from app.core.languages import Lang, lang_from_script
from app.core.registry import MOCK, register
from app.core.trace import CallRecord, TurnTrace

USER_UTTERANCES: dict[Lang, str] = {
    Lang.EN: "When is my next EMI due?",
    Lang.HI: "मेरी अगली EMI कब है?",
    Lang.KN: "ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?",
    Lang.BN: "আমার পরের EMI কবে?",
}

AGENT_REPLIES: dict[Lang, str] = {
    Lang.EN: "Sure, I can help with that. Your next EMI is due on the fifth.",
    Lang.HI: "जी, मैं मदद करती हूँ। आपकी अगली EMI पाँच तारीख को है।",
    Lang.KN: "ಖಂಡಿತ, ನಾನು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ನಿಮ್ಮ ಮುಂದಿನ EMI ಐದನೇ ತಾರೀಖು.",
    Lang.BN: "নিশ্চয়ই, আমি সাহায্য করছি। আপনার পরের EMI পাঁচ তারিখে।",
}

GREETING = "Hello. Namaste. Namaskara. Nomoshkar."

# Different rates on purpose, so the client's handling of a rate change gets exercised.
TTS_SAMPLE_RATES: dict[Lang, int] = {
    Lang.EN: 22_050,
    Lang.HI: 22_050,
    Lang.KN: 24_000,
    Lang.BN: 24_000,
}


# --- Scripted audio --------------------------------------------------------------


def tone(ms: int, sample_rate: int = INPUT_SAMPLE_RATE, hz: float = 220.0) -> bytes:
    count = sample_rate * ms // 1000
    step = 2 * math.pi * hz / sample_rate
    return array.array("h", (int(12_000 * math.sin(step * i)) for i in range(count))).tobytes()


def silence(ms: int, sample_rate: int = INPUT_SAMPLE_RATE) -> bytes:
    return bytes(sample_rate * ms // 1000 * 2)


def frames(pcm: bytes) -> list[bytes]:
    """Split input audio into protocol frames, zero-padding the last one."""
    out = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if out and len(out[-1]) < FRAME_BYTES:
        out[-1] = out[-1].ljust(FRAME_BYTES, b"\x00")
    return out


# --- Components ------------------------------------------------------------------


class MockVAD:
    """Energy threshold with a short hangover before declaring the end of speech."""

    def __init__(self, threshold: float = 500.0, end_silence_frames: int = 8) -> None:
        self._threshold = threshold
        self._end_silence_frames = end_silence_frames
        self._speaking = False
        self._quiet = 0

    async def process(self, frame: bytes) -> VADEvent | None:
        samples = array.array("h", frame)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
        if rms >= self._threshold:
            self._quiet = 0
            if not self._speaking:
                self._speaking = True
                return VADEvent(VADEventKind.SPEECH_START)
        elif self._speaking:
            self._quiet += 1
            if self._quiet >= self._end_silence_frames:
                self._speaking = False
                self._quiet = 0
                return VADEvent(VADEventKind.SPEECH_END)
        return None


class MockSTT:
    """Returns scripted utterances in order, one per finish()."""

    languages = set(Lang)

    def __init__(self, script: list[str] | None = None) -> None:
        self._script = script
        self._language = Lang.EN
        self._queue: asyncio.Queue[Transcript | None] = asyncio.Queue()
        self._utterance = 0
        self._partial_sent = False

    def _text(self) -> str:
        if self._script:
            return self._script[self._utterance % len(self._script)]
        return USER_UTTERANCES[self._language]

    async def start(self, language: Lang) -> None:
        self._language = language

    async def send_audio(self, frame: bytes) -> None:
        if not self._partial_sent:
            self._partial_sent = True
            partial = " ".join(self._text().split()[:2])
            await self._queue.put(Transcript(partial, final=False, language=self._language))

    async def finish(self) -> None:
        text = self._text() if self._partial_sent else ""
        await self._queue.put(Transcript(text, final=True, language=self._language, confidence=0.9))
        self._utterance += 1
        self._partial_sent = False

    async def events(self) -> AsyncIterator[Transcript]:
        while (item := await self._queue.get()) is not None:
            yield item

    async def close(self) -> None:
        await self._queue.put(None)


class MockLanguageDetector:
    """Audio detection returns a scripted guess. Text detection is the real script check."""

    def __init__(self, audio_guess: LangGuess | None = None) -> None:
        self._audio_guess = audio_guess or LangGuess(Lang.EN, 0.9)

    async def detect_audio(self, pcm: bytes) -> LangGuess:
        await asyncio.sleep(0)
        return self._audio_guess

    def detect_text(self, text: str) -> LangGuess:
        return LangGuess(*lang_from_script(text))


class MockLLM:
    """Streams a canned reply word by word, after any scripted tool calls."""

    def __init__(self, language: Lang = Lang.EN, tool_calls: list[ToolCall] | None = None) -> None:
        self._language = language
        self._tool_calls = list(tool_calls or [])

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        if self._tool_calls and not any(m.role == "tool" for m in messages):
            for call in self._tool_calls:
                yield call
            yield Done("tool_calls")
            return
        for word in AGENT_REPLIES[self._language].split(" "):
            await asyncio.sleep(0)
            yield TextDelta(word + " ")
        yield Done("stop")


class MockTTS:
    """A sine wave whose length follows the text length, in 40 ms chunks."""

    languages = set(Lang)

    def __init__(self, ms_per_char: int = 20, chunk_delay: float = 0.0) -> None:
        self._ms_per_char = ms_per_char
        self._chunk_delay = chunk_delay

    def sample_rate(self, language: Lang) -> int:
        return TTS_SAMPLE_RATES[language]

    async def synthesize(self, text: str, language: Lang) -> AsyncIterator[bytes]:
        rate = self.sample_rate(language)
        pcm = tone(max(40, len(text) * self._ms_per_char), sample_rate=rate, hz=330.0)
        chunk_bytes = rate * 40 // 1000 * 2
        for start in range(0, len(pcm), chunk_bytes):
            await asyncio.sleep(self._chunk_delay)
            yield pcm[start : start + chunk_bytes]


class MockAgent:
    """Replies with the canned line for ctx.language and keeps what was actually heard."""

    def __init__(self, ctx: CallContext | None = None) -> None:
        self.history: list[Message] = []

    def greeting(self, ctx: CallContext) -> str:
        return GREETING

    async def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]:
        self.history.append(Message("user", user_text))
        for word in AGENT_REPLIES[ctx.language].split(" "):
            await asyncio.sleep(0)
            yield TextDelta(word + " ")
        yield Done("stop")

    def commit_spoken(self, spoken_text: str, interrupted: bool) -> None:
        self.history.append(Message("assistant", spoken_text))


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.calls: dict[str, CallRecord] = {}
        self.turns: list[TurnTrace] = []

    async def call_started(self, call: CallRecord) -> None:
        self.calls[call.call_id] = call

    async def turn_finished(self, turn: TurnTrace) -> None:
        self.turns.append(turn)

    async def call_ended(self, call: CallRecord) -> None:
        self.calls[call.call_id] = call


register("vad", MOCK, MockVAD)
register("stt", MOCK, lambda lang: MockSTT())
register("langid", MOCK, MockLanguageDetector)
register("llm", MOCK, lambda lang: MockLLM(lang))
register("tts", MOCK, lambda lang: MockTTS())
register("agent", MOCK, MockAgent)
register("trace_sink", MOCK, InMemoryTraceSink)

__all__ = [
    "AGENT_REPLIES",
    "GREETING",
    "USER_UTTERANCES",
    "InMemoryTraceSink",
    "MockAgent",
    "MockLLM",
    "MockLanguageDetector",
    "MockSTT",
    "MockTTS",
    "MockVAD",
    "frames",
    "silence",
    "tone",
]
