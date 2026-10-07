"""Component contracts. Frozen after Phase 0: add fields, never rename or remove.

Rules for every implementation:

- Stop promptly when the calling task is cancelled, and release sockets and pool work
  in `finally`.
- Never block the event loop. CPU work goes through app/core/pools.py.
- Ship nothing that needs an API key at import time.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from app.core.languages import Lang
from app.core.trace import CallRecord, TurnTrace

# Audio from the browser: PCM16 mono, 16 kHz, 512 samples per frame (32 ms).
INPUT_SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES * 1000 / INPUT_SAMPLE_RATE


# --- VAD -------------------------------------------------------------------------


class VADEventKind(StrEnum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VADEvent:
    kind: VADEventKind


@runtime_checkable
class VAD(Protocol):
    """One instance per call. Fed every input frame, in order."""

    async def process(self, frame: bytes) -> VADEvent | None: ...


# --- STT -------------------------------------------------------------------------


@dataclass(frozen=True)
class Transcript:
    text: str
    final: bool
    language: Lang
    confidence: float | None = None


@runtime_checkable
class STT(Protocol):
    """One instance per call and language. To change language, close it and get another.

    Lifecycle: start() once, then for each utterance send_audio() repeatedly and
    finish() at speech end. Every finish() produces exactly one final Transcript on
    events(), with empty text if nothing was recognised. close() ends events().
    """

    languages: set[Lang]

    async def start(self, language: Lang) -> None: ...
    async def send_audio(self, frame: bytes) -> None: ...
    async def finish(self) -> None: ...
    def events(self) -> AsyncIterator[Transcript]: ...
    async def close(self) -> None: ...


# --- Language detection ----------------------------------------------------------


@dataclass(frozen=True)
class LangGuess:
    lang: Lang | None  # None means "no evidence"
    confidence: float  # 0.0 to 1.0


@runtime_checkable
class LanguageDetector(Protocol):
    async def detect_audio(self, pcm: bytes) -> LangGuess:
        """Detect from the first utterance: PCM16 mono at INPUT_SAMPLE_RATE."""
        ...

    def detect_text(self, text: str) -> LangGuess:
        """Script check on a transcript. Latin-only text is no evidence."""
        ...


# --- LLM -------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # on assistant messages
    tool_call_id: str | None = None  # on tool messages


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Done:
    finish_reason: str = "stop"


LLMEvent = TextDelta | ToolCall | Done


@runtime_checkable
class LLM(Protocol):
    def stream(self, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[LLMEvent]: ...


# --- TTS -------------------------------------------------------------------------


@runtime_checkable
class TTS(Protocol):
    """May be one shared instance per process. Callers must not close or tear it down."""

    languages: set[Lang]

    def sample_rate(self, language: Lang) -> int: ...

    def synthesize(self, text: str, language: Lang) -> AsyncIterator[bytes]:
        """PCM16 mono chunks at sample_rate(language). One call per sentence."""
        ...


# --- Tools and the agent ---------------------------------------------------------


@dataclass
class CallContext:
    """Mutable per-call state shared by the orchestrator, the agent, and the tools."""

    call_id: str
    language: Lang
    verified_customer_id: int | None = None
    failed_verifications: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ToolResult:
    """A tool the agent ran while producing a reply. Recorded in the turn's trace."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float


@dataclass(frozen=True)
class Handoff:
    reason: str


@dataclass(frozen=True)
class ModelFirstToken:
    """The model produced its first token for this reply. Carries no text.

    Optional, at most once per respond(), before the first TextDelta. It lets the trace
    separate model latency from the time the agent spends checking a sentence.
    """


AgentEvent = TextDelta | ToolResult | Handoff | ModelFirstToken | Done


@runtime_checkable
class Agent(Protocol):
    """One instance per call. Owns the conversation history, prompts, tools, and guardrails.

    The orchestrator never talks to the LLM directly. It calls respond() once per user
    turn and speaks the TextDelta stream. Replies must be in ctx.language, which the
    orchestrator may change between turns.

    A TextDelta is text that is cleared to be spoken and cannot be taken back. Its size
    is up to the agent: it may be a token, or a whole sentence that was held until a
    guardrail passed it. Callers must not assume either.
    """

    def greeting(self, ctx: CallContext) -> str:
        """Opening line. Must be short and language-neutral when the language is unknown."""
        ...

    def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]: ...

    def commit_spoken(self, spoken_text: str, interrupted: bool) -> None:
        """Called once after each reply with the part the caller actually heard.

        On a barge-in this is shorter than what respond() produced. The agent stores
        this text, not the full reply, as its turn in the history.
        """
        ...


# --- Trace persistence -----------------------------------------------------------


@runtime_checkable
class TraceSink(Protocol):
    """Where the orchestrator sends finished calls and turns.

    Implementations must return quickly and never raise: a database outage must not
    break a live call. Queue the write and do it in the background.

    May be one shared instance per process. Callers must not close or tear it down.
    """

    async def call_started(self, call: CallRecord) -> None: ...
    async def turn_finished(self, turn: TurnTrace) -> None: ...
    async def call_ended(self, call: CallRecord) -> None: ...
