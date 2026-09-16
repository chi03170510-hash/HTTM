import time
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.violation import CameraStatusResponse
from services.camera_service import camera_service
from config import CAMERA_SOURCE, MODEL_PATH
from database import SessionLocal

router = APIRouter()


@router.post("/start", response_model=CameraStatusResponse)
async def start_camera():
    """Starts camera monitoring."""
    if camera_service.is_running:
        raise HTTPException(status_code=400, detail="Camera đang chạy")
    try:
        loop = asyncio.get_running_loop()
        camera_service.start(
            camera_source=CAMERA_SOURCE,
            model_path=MODEL_PATH,
            db_session_factory=SessionLocal,
            loop=loop,
        )
        return CameraStatusResponse(is_running=True, message="Đã bắt đầu giám sát")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model AI không tồn tại: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không mở được camera: {str(e)}")


@router.post("/stop", response_model=CameraStatusResponse)
async def stop_camera():
    """Stops camera monitoring."""
    if not camera_service.is_running:
        raise HTTPException(status_code=400, detail="Camera chưa chạy")
    try:
        camera_service.stop()
        return CameraStatusResponse(is_running=False, message="Đã dừng giám sát")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dừng camera: {str(e)}")


@router.get("/status", response_model=CameraStatusResponse)
async def get_camera_status():
    """Returns camera running status."""
    status = camera_service.is_running
    msg = "Camera đang chạy" if status else "Camera chưa chạy"
    return CameraStatusResponse(is_running=status, message=msg)


@router.get("/video_feed")
def video_feed():
    """MJPEG streaming endpoint for frontend."""
    def frame_generator():
        start_wait = time.time()
        while not camera_service.is_running and (time.time() - start_wait) < 3.0:
            time.sleep(0.1)

        while camera_service.is_running:
            frame_bytes = camera_service.get_jpeg_frame()
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.04)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )