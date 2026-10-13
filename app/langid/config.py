"""Language detection tunables, from MULTIVOCO_LANGID_* environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LangIDSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MULTIVOCO_LANGID_", env_file=".env", extra="ignore"
    )

    deepgram_url: str = "https://api.deepgram.com/v1/listen"
    deepgram_model: str = "nova-3"
    timeout_seconds: float = 6.0
    # Shorter audio than this is not worth a guess.
    min_audio_ms: float = 400.0
    # How far ahead of the runner-up the winner must be for the guess to count as sure.
    # A lead of exactly this much reports a confidence of SURE (0.6), which is the
    # orchestrator's threshold for acting on a guess without asking the caller.
    sure_margin: float = 0.10


@lru_cache
def get_langid_settings() -> LangIDSettings:
    return LangIDSettings()
