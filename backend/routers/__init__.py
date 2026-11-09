from routers.camera import router as camera_router
from routers.violations import router as violations_router
from routers.websocket import router as ws_router

__all__ = ["camera_router", "violations_router", "ws_router"]
