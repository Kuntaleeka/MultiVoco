"""How the pipeline gets its components. Nothing else names a provider.

Each workstream registers its implementations when its package is imported:

    register("stt", "deepgram", lambda lang: DeepgramSTT())      # routed per language
    register("vad", DEFAULT, lambda: SileroVAD())                # one implementation

The routed kinds (stt, llm, tts) are chosen by the routing table in languages.py.
The others have a single implementation registered as DEFAULT. Any kind listed in
MULTIVOCO_MOCK gets the mock from app/core/mocks.py instead.
"""

from collections.abc import Callable
from typing import Any

from app.core.config import COMPONENT_KINDS, get_settings
from app.core.interfaces import (
    LLM,
    STT,
    TTS,
    VAD,
    Agent,
    CallContext,
    LanguageDetector,
    TraceSink,
)
from app.core.languages import Lang, route_for

MOCK = "mock"
DEFAULT = "default"
ROUTED_KINDS = ("stt", "llm", "tts")

_factories: dict[str, dict[str, Callable[..., Any]]] = {kind: {} for kind in COMPONENT_KINDS}


class ProviderNotRegistered(LookupError):
    pass


def register(kind: str, name: str, factory: Callable[..., Any]) -> None:
    _factories[kind][name] = factory


def provider_name(kind: str, lang: Lang | None = None) -> str:
    """Name of the implementation that would be used. Recorded in each turn's trace."""
    if get_settings().is_mocked(kind):
        return MOCK
    if kind in ROUTED_KINDS:
        if lang is None:
            raise ValueError(f"{kind} is routed per language")
        return getattr(route_for(lang), kind)
    return DEFAULT


def _create(kind: str, lang: Lang | None, *args: Any) -> Any:
    name = provider_name(kind, lang)
    if name == MOCK:
        import app.core.mocks  # noqa: F401  (registers the mocks)
    try:
        factory = _factories[kind][name]
    except KeyError:
        raise ProviderNotRegistered(
            f"No {kind} implementation registered as {name!r}. "
            f"Import the package that provides it, or add {kind} to MULTIVOCO_MOCK."
        ) from None
    return factory(*args)


def get_stt(lang: Lang) -> STT:
    return _create("stt", lang, lang)


def get_llm(lang: Lang) -> LLM:
    return _create("llm", lang, lang)


def get_tts(lang: Lang) -> TTS:
    return _create("tts", lang, lang)


def get_vad() -> VAD:
    return _create("vad", None)


def get_language_detector() -> LanguageDetector:
    return _create("langid", None)


def get_agent(ctx: CallContext) -> Agent:
    return _create("agent", None, ctx)


def get_trace_sink() -> TraceSink:
    return _create("trace_sink", None)
