"""Loads the workstream packages and runs their startup and shutdown hooks.

The convention every package under app/ may follow:

- Importing the package registers its implementations (app/core/registry.py).
- `router`: a FastAPI APIRouter, included in the app.
- `async def startup()`: called once before the app serves requests.
- `async def shutdown()`: called once when the app stops, in reverse load order.

A provider package is imported only when one of its component kinds is not mocked, so
a fresh clone with MULTIVOCO_MOCK=all never imports a provider SDK. The database and
metrics packages are always imported when present, since they need no keys.
"""

import importlib
import logging
from types import ModuleType

from fastapi import FastAPI

from app.core.config import get_settings

log = logging.getLogger(__name__)

# Package -> the component kinds it provides. Empty means "always load if present".
PACKAGES: dict[str, tuple[str, ...]] = {
    "app.audio": ("vad",),
    "app.stt": ("stt",),
    "app.langid": ("langid",),
    "app.agent": ("agent", "llm"),
    "app.tts": ("tts",),
    "app.db": (),
    "app.metrics": (),
}


def load_packages(app: FastAPI) -> list[ModuleType]:
    settings = get_settings()
    loaded: list[ModuleType] = []
    for name, kinds in PACKAGES.items():
        needed = [kind for kind in kinds if not settings.is_mocked(kind)]
        if kinds and not needed:
            continue
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise  # the package exists but one of its own imports is missing
            if needed:
                log.warning("%s is not mocked but %s does not exist yet", needed, name)
            continue
        router = getattr(module, "router", None)
        if router is not None:
            app.include_router(router)
        loaded.append(module)
    return loaded


async def run_startup(modules: list[ModuleType]) -> None:
    for module in modules:
        hook = getattr(module, "startup", None)
        if hook is not None:
            await hook()


async def run_shutdown(modules: list[ModuleType]) -> None:
    for module in reversed(modules):
        hook = getattr(module, "shutdown", None)
        if hook is None:
            continue
        try:
            await hook()
        except Exception:
            log.exception("shutdown hook of %s failed", module.__name__)
