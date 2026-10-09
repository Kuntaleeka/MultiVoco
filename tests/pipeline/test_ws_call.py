"""The real endpoint, over a real (in-process) WebSocket."""

import json

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.languages import Lang
from app.core.mocks import AGENT_REPLIES, GREETING, frames, silence, tone
from app.main import app


def read_until(ws, predicate) -> list:
    """Collect messages until one JSON message satisfies the predicate."""
    seen = []
    while True:
        message = ws.receive()
        if message.get("text") is not None:
            item = json.loads(message["text"])
            seen.append(item)
            if predicate(item):
                return seen
        elif message.get("bytes") is not None:
            seen.append(message["bytes"])
        else:
            raise AssertionError(f"socket closed early: {message}")


def listening(message) -> bool:
    return message == {"type": "state", "value": "listening"}


def test_a_whole_call_over_the_socket():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_text('{"type": "start", "lang": "kn"}')
        greeting = read_until(ws, listening)
        assert greeting[0]["type"] == "ready"
        assert {
            "type": "language",
            "lang": "kn",
            "source": "manual",
            "confidence": None,
        } in greeting
        assert any(isinstance(item, bytes) for item in greeting)
        assert client.get("/healthz").json()["active_sessions"] == 1

        for frame in frames(tone(600) + silence(400)):
            ws.send_bytes(frame)
        turn = read_until(ws, listening)
        finals = [m for m in turn if isinstance(m, dict) and m.get("final")]
        assert [(m["role"], m["text"]) for m in finals] == [
            ("user", "ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?"),
            ("agent", AGENT_REPLIES[Lang.KN]),
        ]
        assert {"type": "audio_start", "turn_id": 1, "sample_rate": 24000} in turn

        ws.send_text('{"type": "end"}')
        assert ws.receive() == {"type": "websocket.close", "code": 1000, "reason": ""}
    with TestClient(app) as client:
        assert client.get("/healthz").json()["active_sessions"] == 0


def test_a_second_caller_is_told_the_line_is_busy(settings_env):
    settings_env(MAX_CONCURRENT_SESSIONS="1")
    with TestClient(app) as client, client.websocket_connect("/ws/call") as first:
        first.send_text('{"type": "start"}')
        read_until(first, listening)
        with client.websocket_connect("/ws/call") as second:
            assert json.loads(second.receive()["text"])["code"] == "busy"
            assert second.receive()["code"] == 4429
        # The first call is unaffected.
        first.send_text('{"type": "end"}')
        assert first.receive()["code"] == 1000


def test_a_dropped_connection_frees_the_line(settings_env):
    settings_env(MAX_CONCURRENT_SESSIONS="1")
    with TestClient(app) as client:
        with client.websocket_connect("/ws/call") as ws:
            ws.send_text('{"type": "start"}')
            ws.receive()
        with client.websocket_connect("/ws/call") as ws:
            ws.send_text('{"type": "start"}')
            assert json.loads(ws.receive()["text"])["type"] == "ready"


def test_a_bad_message_closes_with_4400():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_text('{"type": "start", "lang": "fr"}')
        assert json.loads(ws.receive()["text"])["code"] == "bad_message"
        try:
            assert ws.receive()["code"] == 4400
        except WebSocketDisconnect as closed:
            assert closed.code == 4400


def test_greeting_is_neutral_until_the_language_is_known():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_text('{"type": "start", "lang": "auto"}')
        greeting = read_until(ws, listening)
        assert not any(isinstance(m, dict) and m["type"] == "language" for m in greeting)
        final = next(m for m in greeting if isinstance(m, dict) and m.get("final"))
        assert final["text"] == GREETING
