from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.camera_service import camera_service

router = APIRouter()


@router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring and warning broadcasts.
    """
    await websocket.accept()
    camera_service.register_websocket(websocket)
    try:
        while True:
            # Keeps connection open, can receive client heartbeats or messages
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        camera_service.unregister_websocket(websocket)
