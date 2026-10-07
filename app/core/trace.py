"""Per-turn latency trace. One TurnTrace per turn, filled in by the orchestrator."""

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.core.languages import Lang


class Mark(StrEnum):
    SPEECH_END = "speech_end"  # VAD declares end of user speech
    LANG_DETECTED = "lang_detected"  # only on turns where detection ran
    STT_FINAL = "stt_final"  # final transcript received
    LLM_FIRST_TOKEN = "llm_first_token"  # first text delta
    LLM_FIRST_SENTENCE = "llm_first_sentence"  # first sentence handed to TTS
    TTS_FIRST_CHUNK = "tts_first_chunk"  # first audio chunk produced
    AUDIO_SENT = "audio_sent"  # first audio chunk written to the socket


# Stage name -> (from mark, to mark). Detection time is part of the "stt" stage.
STAGES: dict[str, tuple[Mark, Mark]] = {
    "stt": (Mark.SPEECH_END, Mark.STT_FINAL),
    "llm": (Mark.STT_FINAL, Mark.LLM_FIRST_TOKEN),
    "sentence": (Mark.LLM_FIRST_TOKEN, Mark.LLM_FIRST_SENTENCE),
    "tts": (Mark.LLM_FIRST_SENTENCE, Mark.TTS_FIRST_CHUNK),
    "send": (Mark.TTS_FIRST_CHUNK, Mark.AUDIO_SENT),
}


class SessionClock:
    """Monotonic milliseconds since the session started."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000


@dataclass
class ToolCallTrace:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float


@dataclass
class TurnTrace:
    call_id: str
    idx: int
    language: Lang
    marks: dict[str, float] = field(default_factory=dict)
    user_text: str = ""
    agent_text: str = ""  # what the caller heard, after any barge-in truncation
    interrupted: bool = False
    ms_played: float | None = None
    language_switched: bool = False
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)  # {"stt": "deepgram", ...}
    region: str = ""
    # Client clock, not comparable with the server marks. Stored separately on purpose.
    client_playback_started_ms: float | None = None

    def mark(self, name: Mark, at_ms: float) -> None:
        """Record a mark. The first value wins, so callers can mark on every chunk."""
        self.marks.setdefault(name.value, at_ms)

    def get(self, name: Mark) -> float | None:
        return self.marks.get(name.value)

    @property
    def time_to_first_audio_ms(self) -> float | None:
        """The headline metric: audio_sent - speech_end."""
        start, end = self.get(Mark.SPEECH_END), self.get(Mark.AUDIO_SENT)
        if start is None or end is None:
            return None
        return end - start

    def stage_durations(self) -> dict[str, float]:
        durations: dict[str, float] = {}
        for stage, (lo, hi) in STAGES.items():
            start, end = self.get(lo), self.get(hi)
            if start is not None and end is not None:
                durations[stage] = end - start
        return durations

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["language"] = self.language.value
        data["time_to_first_audio_ms"] = self.time_to_first_audio_ms
        data["stages"] = self.stage_durations()
        return data


@dataclass
class CallRecord:
    call_id: str
    started_at: datetime
    requested_lang: str  # "auto" or a Lang value
    final_lang: Lang | None = None
    ended_at: datetime | None = None
    verified_customer_id: int | None = None
    outcome: str | None = None  # "completed", "handoff", "abandoned", "limit", "error"
    handoff_reason: str | None = None
