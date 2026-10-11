"""STT tunables, from MULTIVOCO_STT_* environment variables. Keys are in app/core/config.py."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIVOCO_STT_", env_file=".env", extra="ignore")

    deepgram_url: str = "wss://api.deepgram.com/v1/listen"
    deepgram_model: str = "nova-3"
    # Our language -> Deepgram's `language`. "multi" is its code-switching mode, which
    # covers Hindi and English together.
    deepgram_languages: dict[str, str] = Field(
        default_factory=lambda: {"en": "multi", "hi": "multi", "kn": "kn", "bn": "bn"}
    )
    # Deepgram's own end-of-speech detection. Its default (10 ms) finalises an empty
    # segment over a short opening word such as "Hello," and the word is lost. The VAD
    # does our endpointing, so this is off. A number of milliseconds also works.
    deepgram_endpointing: str = "false"
    # Deepgram drops a connection that gets no audio for about ten seconds.
    deepgram_keepalive_seconds: float = 4.0
    # How long to wait for the last words after asking Deepgram to flush.
    finalize_timeout_seconds: float = 1.5
    connect_timeout_seconds: float = 5.0

    groq_url: str = "https://api.groq.com/openai/v1/audio/transcriptions"
    groq_model: str = "whisper-large-v3-turbo"
    groq_timeout_seconds: float = 15.0
    # Whisper invents words on silence ("Thank you."). Two guards: audio quieter than
    # this RMS level (16-bit units) is not sent at all, and segments Whisper rates this
    # likely to be non-speech are dropped. On Groq the second guard rarely fires: pure
    # silence came back with a no-speech probability of 0.
    whisper_min_rms: float = 100.0
    whisper_max_no_speech_prob: float = 0.6


@lru_cache
def get_stt_settings() -> STTSettings:
    return STTSettings()
