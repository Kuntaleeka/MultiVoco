"""One agent reply: what was spoken, how much audio that was, and what was heard."""

from dataclasses import dataclass, field

from app.core.languages import Lang
from app.core.trace import TurnTrace

# Used to place a cut inside a sentence when no finished sentence gives a measured rate.
_FALLBACK_MS_PER_CHAR = 70.0


@dataclass
class Segment:
    """One sentence handed to TTS, and the audio sent for it so far."""

    text: str
    ms: float = 0.0
    complete: bool = False  # False if synthesis was cut short


@dataclass
class Turn:
    id: int
    lang: Lang
    trace: TurnTrace
    sample_rate: int = 0
    segments: list[Segment] = field(default_factory=list)
    audio_started: bool = False  # audio_start has been sent
    first_audio_at: float | None = None  # loop time of the first chunk sent
    playback_anchor: float | None = None  # loop time the client reported playback began
    finished: bool = False
    handoff: str | None = None

    @property
    def audio_ms(self) -> float:
        return sum(segment.ms for segment in self.segments)

    def spoken_text(self, complete_only: bool = False) -> str:
        return " ".join(s.text for s in self.segments if s.complete or not complete_only)

    def heard_text(self, ms_played: float) -> str:
        """The part of the reply the caller heard, given how much audio was played.

        Whole sentences are exact. Inside a sentence the cut is an estimate from the
        speaking rate, moved back to the previous word boundary so no word is cut.
        """
        done = [s for s in self.segments if s.complete and s.text]
        rate = (
            sum(s.ms for s in done) / sum(len(s.text) for s in done)
            if done
            else _FALLBACK_MS_PER_CHAR
        )
        heard: list[str] = []
        remaining = ms_played
        for segment in self.segments:
            if segment.complete and remaining >= segment.ms - 1:
                heard.append(segment.text)
                remaining -= segment.ms
                continue
            per_char = segment.ms / len(segment.text) if segment.complete else rate
            chars = int(remaining / per_char) if per_char > 0 else 0
            if chars >= len(segment.text):
                heard.append(segment.text)
            elif chars > 0:
                head = segment.text[:chars]
                if not (head[-1].isspace() or segment.text[chars].isspace()):
                    head = head.rsplit(None, 1)[0] if " " in head.strip() else ""
                heard.append(head.strip())
            break
        return " ".join(part for part in heard if part)
