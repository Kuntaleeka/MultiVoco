from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.core.pools import shutdown_pools


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    shutdown_pools()


app = FastAPI(title="MultiVoco", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "region": settings.region, "mock": settings.mock}


@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket) -> None:
    """Sends back whatever it receives. For checking transport through the host's proxy."""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await websocket.send_bytes(message["bytes"])
            elif message.get("text") is not None:
                await websocket.send_text(message["text"])
    except WebSocketDisconnect:
        return
