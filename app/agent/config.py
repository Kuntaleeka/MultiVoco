"""Workstream B settings, read from the environment (prefix MULTIVOCO_) and .env.

Kept here rather than in app/core/config.py, which is frozen. The API keys themselves
are in the core settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIVOCO_", env_file=".env", extra="ignore")

    # Checked live on 2026-10-11, free tier. Groq no longer serves Llama chat models.
    # gemini-3.8-flash timed out or returned 503 and 429, and gemini-3.5-flash took 3 to
    # 7 s to start, so the lite model is the backup.
    groq_model: str = "openai/gpt-oss-120b"
    # A second Groq model for when the first is over its limit: the free tier allows 8000
    # tokens a minute per model, and one agent request costs about 900.
    # Off by default. `openai/gpt-oss-20b` was tried here and is not safe in front of a
    # caller: in a real call it wrote the caller's lines as well as its own (section 12,
    # note 20). With this empty, a rate-limited Groq falls through to Gemini.
    groq_backup_model: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # Both default models think before answering. Low keeps time to first token short.
    # Empty leaves the parameter out, for a model that rejects it.
    llm_reasoning_effort: str = "low"
    llm_temperature: float = 0.3
    # Covers the model's thinking as well as the spoken reply.
    llm_max_tokens: int = 500

    llm_connect_timeout_s: float = 3.0
    # Longest wait for the response to start, and between two chunks of it.
    llm_read_timeout_s: float = 6.0
    # Extra tries on one provider, only while nothing has been yielded yet.
    llm_retries: int = 1
    llm_retry_delay_s: float = 0.25


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()
