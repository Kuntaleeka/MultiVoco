import pytest
from pydantic import ValidationError

from app.core import protocol
from app.core.languages import Lang


def test_start_defaults_to_auto():
    message = protocol.parse_client_message('{"type": "start"}')
    assert isinstance(message, protocol.Start)
    assert message.lang == "auto"


@pytest.mark.parametrize("lang", list(Lang))
def test_start_accepts_every_language(lang):
    message = protocol.parse_client_message(f'{{"type": "start", "lang": "{lang.value}"}}')
    assert message.lang is lang


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"type": "set_language", "lang": "kn"}', protocol.SetLanguage),
        (
            '{"type": "playback_started", "turn_id": 2, "t_client_ms": 81234.5}',
            protocol.PlaybackStarted,
        ),
        (
            '{"type": "playback_position", "turn_id": 2, "ms_played": 640}',
            protocol.PlaybackPosition,
        ),
        ('{"type": "end"}', protocol.End),
    ],
)
def test_client_messages_parse(raw, expected):
    assert isinstance(protocol.parse_client_message(raw), expected)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"type": "nope"}',
        '{"type": "start", "lang": "fr"}',
        '{"type": "set_language", "lang": "auto"}',
        '{"type": "playback_position", "turn_id": 2}',
    ],
)
def test_bad_client_messages_are_rejected(raw):
    with pytest.raises(ValidationError):
        protocol.parse_client_message(raw)


def test_server_messages_serialise_with_their_type():
    message = protocol.LanguageMessage(
        lang=Lang.KN, source=protocol.LanguageSource.AUDIO, confidence=0.93
    )
    assert message.model_dump(mode="json") == {
        "type": "language",
        "lang": "kn",
        "source": "audio",
        "confidence": 0.93,
    }
    assert protocol.State(value=protocol.SessionState.SPEAKING).model_dump(mode="json") == {
        "type": "state",
        "value": "speaking",
    }
