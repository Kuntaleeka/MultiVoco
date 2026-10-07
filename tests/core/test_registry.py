import pytest

from app.core import registry
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
from app.core.languages import Lang


@pytest.mark.parametrize("lang", list(Lang))
def test_mocks_satisfy_every_interface(lang):
    assert isinstance(registry.get_stt(lang), STT)
    assert isinstance(registry.get_llm(lang), LLM)
    assert isinstance(registry.get_tts(lang), TTS)
    assert isinstance(registry.get_vad(), VAD)
    assert isinstance(registry.get_language_detector(), LanguageDetector)
    assert isinstance(registry.get_agent(CallContext("c1", lang)), Agent)
    assert isinstance(registry.get_trace_sink(), TraceSink)


def test_provider_names_follow_the_routing_table(settings_env):
    settings_env(MOCK="none")
    assert registry.provider_name("stt", Lang.EN) == "deepgram"
    assert registry.provider_name("tts", Lang.KN) == "azure"
    assert registry.provider_name("vad") == registry.DEFAULT


def test_kinds_can_be_mocked_one_at_a_time(settings_env):
    settings_env(MOCK="llm, tts")
    assert registry.provider_name("llm", Lang.KN) == registry.MOCK
    assert registry.provider_name("stt", Lang.KN) == "groq_whisper"


def test_unregistered_provider_fails_with_a_clear_error(settings_env):
    settings_env(MOCK="none")
    with pytest.raises(registry.ProviderNotRegistered, match="deepgram"):
        registry.get_stt(Lang.EN)


def test_registered_provider_is_used(settings_env, monkeypatch):
    settings_env(MOCK="none")
    sentinel = object()
    monkeypatch.setitem(registry._factories["tts"], "azure", lambda lang: (sentinel, lang))
    assert registry.get_tts(Lang.BN) == (sentinel, Lang.BN)
