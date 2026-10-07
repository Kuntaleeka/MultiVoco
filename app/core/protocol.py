"""JSON messages on /ws/call. The prose version, with examples, is docs/protocol.md.

Binary frames are not modelled here: client frames are input audio, server frames are
the audio of the turn announced by the latest audio_start.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.core.languages import AUTO, Lang

LangOrAuto = Lang | Literal["auto"]


# --- Client to server ------------------------------------------------------------


class Start(BaseModel):
    type: Literal["start"] = "start"
    lang: LangOrAuto = AUTO


class SetLanguage(BaseModel):
    type: Literal["set_language"] = "set_language"
    lang: Lang


class PlaybackStarted(BaseModel):
    type: Literal["playback_started"] = "playback_started"
    turn_id: int
    t_client_ms: float


class PlaybackPosition(BaseModel):
    type: Literal["playback_position"] = "playback_position"
    turn_id: int
    ms_played: float


class End(BaseModel):
    type: Literal["end"] = "end"


ClientMessage = Annotated[
    Start | SetLanguage | PlaybackStarted | PlaybackPosition | End,
    Field(discriminator="type"),
]
_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """Raises pydantic.ValidationError on anything malformed or unknown."""
    return _client_adapter.validate_json(raw)


# --- Server to client ------------------------------------------------------------


class SessionState(StrEnum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HANDOFF = "handoff"


class LanguageSource(StrEnum):
    MANUAL = "manual"  # the caller picked it
    AUDIO = "audio"  # detected from the first utterance
    SCRIPT = "script"  # switched after the transcript script disagreed twice
    ASKED = "asked"  # the agent asked and the caller answered


class ErrorCode(StrEnum):
    BAD_MESSAGE = "bad_message"
    BUSY = "busy"  # too many concurrent sessions
    LIMIT_REACHED = "limit_reached"  # call length, turn count, or idle timeout
    LANGUAGE_UNAVAILABLE = "language_unavailable"  # a provider for it is down or over quota
    PROVIDER_ERROR = "provider_error"
    INTERNAL = "internal"


class Ready(BaseModel):
    type: Literal["ready"] = "ready"
    session_id: str


class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    role: Literal["user", "agent"]
    text: str
    final: bool
    turn_id: int


class AudioStart(BaseModel):
    type: Literal["audio_start"] = "audio_start"
    turn_id: int
    sample_rate: int


class AudioEnd(BaseModel):
    type: Literal["audio_end"] = "audio_end"
    turn_id: int


class Flush(BaseModel):
    type: Literal["flush"] = "flush"
    turn_id: int


class State(BaseModel):
    type: Literal["state"] = "state"
    value: SessionState


class LanguageMessage(BaseModel):
    type: Literal["language"] = "language"
    lang: Lang
    source: LanguageSource
    confidence: float | None = None


class Error(BaseModel):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: str


ServerMessage = (
    Ready | TranscriptMessage | AudioStart | AudioEnd | Flush | State | LanguageMessage | Error
)

# WebSocket close codes used by the server.
CLOSE_NORMAL = 1000
CLOSE_BAD_MESSAGE = 4400
CLOSE_LIMIT_REACHED = 4408
CLOSE_BUSY = 4429
CLOSE_INTERNAL = 4500
