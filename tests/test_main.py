from fastapi.testclient import TestClient

from app.core.mocks import tone
from app.main import app


def test_healthz():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ws_echo_returns_binary_and_text_unchanged():
    frame = tone(32)
    with TestClient(app) as client, client.websocket_connect("/ws/echo") as ws:
        ws.send_bytes(frame)
        assert ws.receive_bytes() == frame
        ws.send_text('{"type": "start", "lang": "kn"}')
        assert ws.receive_text() == '{"type": "start", "lang": "kn"}'
