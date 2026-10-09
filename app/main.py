from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.pools import shutdown_pools
from app.pipeline import plugins
from app.pipeline.routes import active_sessions
from app.pipeline.routes import router as call_router

WEB = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await plugins.run_startup(app.state.packages)
    yield
    await plugins.run_shutdown(app.state.packages)
    shutdown_pools()


app = FastAPI(title="MultiVoco", lifespan=lifespan)
app.include_router(call_router)
app.state.packages = plugins.load_packages(app)


@app.get("/healthz")
async def healthz() -> dict[str, str | int]:
    settings = get_settings()
    return {
        "status": "ok",
        "region": settings.region,
        "mock": settings.mock,
        "active_sessions": active_sessions(),
    }


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


# Static pages go last, so they never shadow an API route.
if (WEB / "dashboard").is_dir():
    app.mount("/dashboard", StaticFiles(directory=WEB / "dashboard", html=True), name="dashboard")
if (WEB / "client").is_dir():
    app.mount("/", StaticFiles(directory=WEB / "client", html=True), name="client")
