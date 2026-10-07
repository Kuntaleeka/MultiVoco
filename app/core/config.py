"""Settings, read from the environment (prefix MULTIVOCO_) and an optional .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

COMPONENT_KINDS = ("vad", "stt", "langid", "llm", "tts", "agent", "trace_sink")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIVOCO_", env_file=".env", extra="ignore")

    # "all", "none", or a comma list of component kinds that should use mocks.
    mock: str = "all"

    database_url: str = "sqlite+aiosqlite:///./multivoco.db"
    region: str = "local"
    executor_workers: int = 4

    max_call_seconds: int = 300
    max_turns: int = 40
    idle_timeout_seconds: int = 30
    max_concurrent_sessions: int = 2

    # {"kn": {"llm": "gemini"}} merged over DEFAULT_ROUTES in app/core/languages.py
    route_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)

    deepgram_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    azure_speech_key: str | None = None
    azure_speech_region: str | None = None

    def is_mocked(self, kind: str) -> bool:
        value = self.mock.strip().lower()
        if value == "all":
            return True
        if value in ("", "none"):
            return False
        return kind in {part.strip() for part in value.split(",")}


@lru_cache
def get_settings() -> Settings:
    return Settings()
