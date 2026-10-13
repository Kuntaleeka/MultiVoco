"""VAD tunables, from MULTIVOCO_VAD_* environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class VADSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIVOCO_VAD_", env_file=".env", extra="ignore")

    # Speech probability at or above this starts speech.
    start_threshold: float = 0.5
    # Once speaking, a frame still counts as speech down to this (hysteresis).
    end_threshold: float = 0.35
    # Frames in a row at or above start_threshold before SPEECH_START. 2 frames is 64 ms:
    # enough to ignore a click. The orchestrator's pre-roll keeps the audio from before.
    start_frames: int = 2
    # Silence after speech before SPEECH_END. Lower answers sooner but cuts callers off
    # mid-sentence. This is added to every turn's latency.
    end_silence_ms: int = 500


@lru_cache
def get_vad_settings() -> VADSettings:
    return VADSettings()
