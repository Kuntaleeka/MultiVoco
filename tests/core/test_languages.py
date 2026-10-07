import pytest

from app.core.languages import (
    DEFAULT_ROUTES,
    LANGUAGES,
    Lang,
    Script,
    dominant_script,
    lang_from_script,
    route_for,
    shares_route,
    to_ascii_digits,
)
from app.core.mocks import AGENT_REPLIES, USER_UTTERANCES


def test_every_language_is_fully_described():
    assert set(LANGUAGES) == set(Lang) == set(DEFAULT_ROUTES)
    for info in LANGUAGES.values():
        assert len(info.digits) == 10


@pytest.mark.parametrize("lang", list(Lang))
def test_mock_replies_are_in_the_right_script(lang):
    assert dominant_script(AGENT_REPLIES[lang]) is LANGUAGES[lang].script


@pytest.mark.parametrize("lang", [Lang.HI, Lang.KN, Lang.BN])
def test_script_identifies_language_despite_english_words(lang):
    guess, share = lang_from_script(USER_UTTERANCES[lang])  # each contains "EMI"
    assert guess is lang
    assert share == 1.0


def test_latin_text_is_no_evidence():
    assert lang_from_script("mera next EMI kab due hai") == (None, 0.0)
    assert lang_from_script("12345 ?!") == (None, 0.0)


def test_danda_does_not_count_as_devanagari():
    assert lang_from_script("আমি ভালো আছি।")[0] is Lang.BN
    assert dominant_script("।") is None


def test_mixed_native_scripts_report_the_share():
    guess, share = lang_from_script("ನನ್ನ EMI मे")
    assert guess is Lang.KN
    assert 0.5 < share < 1.0


def test_native_digits_convert_to_ascii():
    for info in LANGUAGES.values():
        assert to_ascii_digits(info.digits) == "0123456789"
    assert to_ascii_digits("₹೧೨,೫೦೦ on ০৫/১১") == "₹12,500 on 05/11"


def test_digits_are_not_script_evidence():
    assert dominant_script("೧೨೩ 456") is None


def test_hindi_and_english_share_a_route():
    assert shares_route(Lang.EN, Lang.HI)
    assert not shares_route(Lang.HI, Lang.KN)


def test_route_overrides_merge_over_defaults(settings_env):
    settings_env(ROUTE_OVERRIDES='{"kn": {"llm": "gemini"}}')
    route = route_for(Lang.KN)
    assert route.llm == "gemini"
    assert route.tts == DEFAULT_ROUTES[Lang.KN].tts
    assert route_for(Lang.BN) == DEFAULT_ROUTES[Lang.BN]


def test_script_enum_covers_every_language():
    assert {info.script for info in LANGUAGES.values()} == set(Script)
