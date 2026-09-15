import asyncio
from fastapi import APIRouter, HTTPException
from schemas.violation import CameraStatusResponse
from services.camera_service import camera_service
from config import CAMERA_INDEX, MODEL_PATH
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
            camera_source=CAMERA_INDEX,
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