import os

# Before anything imports app.core.config: tests never use real providers or a real
# database, whatever is in the developer's .env.
os.environ["MULTIVOCO_MOCK"] = "all"
os.environ["MULTIVOCO_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["MULTIVOCO_ROUTE_OVERRIDES"] = "{}"

import pytest

from app.core.config import Settings, get_settings

# The variables above are not enough for a dict setting: pydantic-settings merges the
# dict from the environment into the one from .env, so a developer's
# MULTIVOCO_ROUTE_OVERRIDES still came through. Tests do not read .env at all.
Settings.model_config["env_file"] = None
get_settings.cache_clear()


@pytest.fixture
def settings_env(monkeypatch):
    """Set MULTIVOCO_* variables for one test: settings_env(MOCK="vad,stt")."""

    def apply(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(f"MULTIVOCO_{key}", value)
        get_settings.cache_clear()

    yield apply
    monkeypatch.undo()
    get_settings.cache_clear()
