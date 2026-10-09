import logging

from fastapi import APIRouter, WebSocket

from app.core import protocol as p
from app.core.config import get_settings
from app.pipeline.session import CallSession

log = logging.getLogger(__name__)
router = APIRouter()

_active: set[CallSession] = set()


def active_sessions() -> int:
    return len(_active)


@router.websocket("/ws/call")
async def ws_call(websocket: WebSocket) -> None:
    await websocket.accept()
    if len(_active) >= get_settings().max_concurrent_sessions:
        error = p.Error(code=p.ErrorCode.BUSY, message="All lines are busy. Try again shortly.")
        await websocket.send_text(error.model_dump_json())
        await websocket.close(p.CLOSE_BUSY)
        return
    session = CallSession(websocket)
    _active.add(session)
    try:
        await session.run()
    except Exception:
        log.exception("session %s crashed", session.id)
    finally:
        _active.discard(session)
        await session.teardown()
