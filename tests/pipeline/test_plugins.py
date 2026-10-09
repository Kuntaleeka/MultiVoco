import sys
import types

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from app.pipeline import plugins


@pytest.fixture
def fake_packages(monkeypatch):
    """Stand-ins for workstream packages, recording the order their hooks run in."""
    calls: list[str] = []

    def make(name: str, with_router: bool = False):
        module = types.ModuleType(name)

        async def startup():
            calls.append(f"start {name}")

        async def shutdown():
            calls.append(f"stop {name}")

        module.startup, module.shutdown = startup, shutdown
        if with_router:
            module.router = APIRouter()
            module.router.add_api_route("/api/fake", lambda: {"ok": True})
        monkeypatch.setitem(sys.modules, name, module)
        return module

    # The database and metrics packages load whenever they exist. Replace the real
    # ones, so these tests do not depend on what those packages do.
    make("app.db")
    make("app.metrics", with_router=True)
    return make, calls


async def test_packages_load_only_when_their_kind_is_not_mocked(fake_packages, settings_env):
    make, calls = fake_packages
    make("app.tts")
    make("app.agent")
    settings_env(MOCK="vad,stt,langid,llm,agent,trace_sink")  # everything but tts

    app = FastAPI()
    loaded = plugins.load_packages(app)
    assert [m.__name__ for m in loaded] == ["app.tts", "app.db", "app.metrics"]
    assert TestClient(app).get("/api/fake").json() == {"ok": True}

    await plugins.run_startup(loaded)
    await plugins.run_shutdown(loaded)
    assert calls == [
        "start app.tts",
        "start app.db",
        "start app.metrics",
        "stop app.metrics",
        "stop app.db",
        "stop app.tts",
    ]


async def test_a_package_with_one_unmocked_kind_is_loaded(fake_packages, settings_env):
    make, _ = fake_packages
    make("app.agent")
    settings_env(MOCK="vad,stt,langid,tts,agent,trace_sink")  # llm is real
    loaded = [m.__name__ for m in plugins.load_packages(FastAPI())]
    assert loaded == ["app.agent", "app.db", "app.metrics"]


def test_no_provider_package_is_imported_when_everything_is_mocked(fake_packages):
    make, _ = fake_packages
    make("app.tts")
    make("app.agent")
    assert [m.__name__ for m in plugins.load_packages(FastAPI())] == ["app.db", "app.metrics"]


def test_a_package_that_does_not_exist_yet_is_skipped(settings_env, caplog):
    settings_env(MOCK="none")
    loaded = plugins.load_packages(FastAPI())
    assert "app.tts" not in [m.__name__ for m in loaded]
    assert "does not exist yet" in caplog.text


async def test_a_failing_shutdown_hook_does_not_stop_the_others(fake_packages):
    make, calls = fake_packages
    first, second = make("app.tts"), make("app.agent")

    async def broken():
        raise RuntimeError("boom")

    second.shutdown = broken
    await plugins.run_shutdown([first, second])
    assert calls == ["stop app.tts"]
