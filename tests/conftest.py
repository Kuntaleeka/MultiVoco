import os

# Before anything imports app.core.config: tests never use real providers or a real
# database, whatever is in the developer's .env.
os.environ["MULTIVOCO_MOCK"] = "all"
os.environ["MULTIVOCO_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["MULTIVOCO_ROUTE_OVERRIDES"] = "{}"

import pytest

from app.core.config import get_settings


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
