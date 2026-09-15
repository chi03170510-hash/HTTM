import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure standard logging for HTTM
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("httm")

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from database import init_db
from config import UPLOAD_DIR
from routers.camera import router as camera_router
from routers.violations import router as violations_router
from routers.websocket import router as ws_router
from services.camera_service import camera_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting HTTM Backend application...")
    camera_service.loop = asyncio.get_running_loop()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    init_db()
    logger.info("Database initialized.")
    yield
    # Shutdown
    logger.info("Shutting down HTTM Backend application...")
    if camera_service.is_running:
        camera_service.stop()


app = FastAPI(
    title="HTTM Backend API",
    description="Hệ thống theo dõi và cảnh báo nhắm mắt - Backend API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup (support React/Next.js frontend, test client, Swagger, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files: /uploads
uploads_base = os.path.join(backend_dir, "uploads")
os.makedirs(uploads_base, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_base), name="uploads")

# Include routers
app.include_router(camera_router, prefix="/api/camera", tags=["Camera"])
app.include_router(violations_router, prefix="/api", tags=["Violations"])
app.include_router(ws_router, tags=["WebSocket"])


@app.get("/", tags=["Health"])
def read_root():
    return {"message": "HTTM Backend API"}


@app.get("/test", include_in_schema=False)
def serve_test_client():
    return FileResponse(os.path.join(backend_dir, "test_client.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)