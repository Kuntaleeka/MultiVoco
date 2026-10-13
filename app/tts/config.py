"""Workstream C settings, read from the environment (prefix MULTIVOCO_) and .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIVOCO_", env_file=".env", extra="ignore")

    # Where the Piper .onnx and .onnx.json files are. Fetch them with:
    #   uv run python -m piper.download_voices <voice> --download-dir voices
    voices_dir: Path = Path("voices")
    # One voice per language: each loaded voice costs memory (see docs/tts.md).
    piper_voice_en: str = "en_US-lessac-medium"
    piper_voice_hi: str = "hi_IN-priyamvada-medium"

    # CPU threads per synthesis. 0 lets ONNX Runtime decide.
    piper_threads: int = 0

    # If Piper has produced no audio after this long, or fails, the sentence is spoken
    # by Sarvam instead. Needs a Sarvam key. 0 turns the fallback off.
    piper_fallback_after_s: float = 4.0

    # Sarvam AI, for Kannada and Bengali. Kept here, not in the frozen core settings.
    sarvam_api_key: str | None = None
    sarvam_model: str = "bulbul:v3"
    sarvam_speaker: str = "priya"
    sarvam_sample_rate: int = 22_050
    sarvam_connect_timeout_s: float = 3.0
    sarvam_read_timeout_s: float = 8.0
    # Characters sent to Sarvam per process per day. The free credit is a one-time
    # grant, so this stops a busy day from spending it. 0 turns the cap off.
    sarvam_daily_chars: int = 30_000

    # Audio is handed over in pieces this long, so a barge-in can stop it part-way.
    tts_chunk_ms: int = 100
    # Load the voices and synthesise once at startup, so the first caller does not wait.
    tts_warm_up: bool = True


@lru_cache
def get_tts_settings() -> TTSSettings:
    return TTSSettings()
