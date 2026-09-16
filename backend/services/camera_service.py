import os
import sys
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, Any
import cv2

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from config import UPLOAD_DIR, CAMERA_SOURCE, MODEL_PATH
from database import SessionLocal
from models.violation import Violation
from services.eye_detection import EyeDetectionService

logger = logging.getLogger("httm")


class CameraService:
    """
    Singleton service managing the background camera capture and AI processing loop.
    Captures frames, delegates to EyeDetectionService, saves violations to DB/disk,
    and broadcasts events over WebSockets.
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.eye_service: Optional[EyeDetectionService] = None
        self.thread: Optional[threading.Thread] = None
        self.latest_result: Optional[Dict[str, Any]] = None
        self.websocket_connections: Set[Any] = set()
        self.db_session_factory = SessionLocal
        self.active_violations: Dict[int, int] = {}  # {face_id: violation_id}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def start(
        self,
        camera_source: Any = None,
        model_path: Optional[str] = None,
        db_session_factory=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Starts the background camera monitoring loop.
        """
        with self._lock:
            if self.is_running:
                if self.thread and self.thread.is_alive():
                    raise ValueError("Camera đang chạy")

                logger.warning("Phát hiện tiến trình camera cũ đã chết. Đang tự động dọn dẹp...")
                self.is_running = False
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                if self.eye_service is not None:
                    self.eye_service.close()
                    self.eye_service = None
                self.thread = None

            source = camera_source if camera_source is not None else CAMERA_SOURCE
            model_file = model_path if model_path is not None else MODEL_PATH

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.warning(f"Không thể mở camera index {source}. Đang thử webcam mặc định (0)...")
                cap.release()
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    logger.warning("Không thể mở webcam 0. Đang thử webcam 1...")
                    cap.release()
                    cap = cv2.VideoCapture(1)

            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Không thể mở bất kỳ nguồn camera nào. Vui lòng kiểm tra lại cấu hình CAMERA_SOURCE hoặc kết nối thiết bị.")
            try:
                self.cap = cap
                self.eye_service = EyeDetectionService(model_file)
                if db_session_factory is not None:
                    self.db_session_factory = db_session_factory
                else:
                    self.db_session_factory = SessionLocal

                if loop is not None:
                    self.loop = loop
                elif self.loop is None:
                    try:
                        self.loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass

                self.active_violations.clear()
                self.is_running = True

                self.thread = threading.Thread(target=self._camera_loop, daemon=True)
                self.thread.start()
                logger.info("Camera started (source=%s)", source)
            except Exception:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                if self.eye_service is not None:
                    self.eye_service.close()
                    self.eye_service = None
                self.is_running = False
                raise

    def stop(self) -> bool:
        """
        Stops the camera monitoring loop and cleans up resources.
        """
        with self._lock:
            if not self.is_running:
                raise ValueError("Camera chưa chạy")

            self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.eye_service is not None:
                self.eye_service.close()
                self.eye_service = None

            self.thread = None
            self.latest_result = None
            self.active_violations.clear()
            logger.info("Camera stopped")

        return True

    def _camera_loop(self):
        """
        Runs in background thread: processes frames, handles violations,
        and schedules WebSocket messages.
        """
        ws_status_interval = 0.1  # Throttle status to ~10/sec
        last_ws_status_time = 0.0

        try:
            while self.is_running:
                if self.cap is None or not self.cap.isOpened():
                    break

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                try:
                    result = self.eye_service.process_frame(frame)
                except Exception as e:
                    logger.error(f"Error in EyeDetectionService: {e}")
                    time.sleep(0.03)
                    continue

                self.latest_result = result
                now_dt = datetime.utcnow()

                if result.get("faces_detected"):
                    for face in result.get("faces", []):
                        face_id = face["face_id"]
                        is_warning = face["is_warning"]
                        eye_state = face["eye_state"]
                        closed_duration = face["closed_duration"]

                        # Case A: Violation triggered and image not yet saved
                        if is_warning and not self.eye_service.is_warning_saved(face_id):
                            timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
                            filename = f"violation_{timestamp_str}_face{face_id}.jpg"
                            file_path = os.path.join(UPLOAD_DIR, filename)

                            # Encode and save image file
                            try:
                                success, encoded_img = cv2.imencode(".jpg", frame)
                                if success:
                                    with open(file_path, "wb") as f:
                                        f.write(encoded_img.tobytes())
                                else:
                                    logger.error(f"cv2.imencode failed for face {face_id}")
                            except Exception as e:
                                logger.error(f"Failed to save violation image: {e}")

                            # Save violation record to MySQL database
                            started_at = now_dt - timedelta(seconds=closed_duration)
                            violation_id = None
                            db = self.db_session_factory()
                            try:
                                violation = Violation(
                                    face_id=face_id,
                                    started_at=started_at,
                                    ended_at=None,
                                    duration=closed_duration,
                                    image_path=filename,
                                    created_at=now_dt,
                                )
                                db.add(violation)
                                db.commit()
                                db.refresh(violation)
                                violation_id = violation.id
                                self.active_violations[face_id] = violation_id
                                logger.warning(
                                    "Violation detected: id=%s, face_id=%s, duration=%.1fs",
                                    violation_id,
                                    face_id,
                                    closed_duration,
                                )
                            except Exception as e:
                                db.rollback()
                                logger.error(f"Database error saving violation: {e}")
                            finally:
                                db.close()

                            self.eye_service.mark_warning_saved(face_id)

                            # Broadcast warning WebSocket event
                            if violation_id is not None:
                                warning_msg = {
                                    "type": "warning",
                                    "timestamp": now_dt.isoformat(),
                                    "face_id": face_id,
                                    "duration": closed_duration,
                                    "violation_id": violation_id,
                                    "image_url": f"/api/violations/{violation_id}/image",
                                    "message": "Phát hiện nhắm mắt >= 5 giây!",
                                }
                                self._schedule_broadcast(warning_msg)

                        # Case B: Eyes reopened after a warning was recorded
                        elif eye_state == "OPEN" and face_id in self.active_violations:
                            violation_id = self.active_violations.pop(face_id, None)
                            if violation_id:
                                db = self.db_session_factory()
                                try:
                                    viol = db.query(Violation).filter(Violation.id == violation_id).first()
                                    if viol:
                                        viol.ended_at = now_dt
                                        total_duration = round((now_dt - viol.started_at).total_seconds(), 2)
                                        viol.duration = total_duration
                                        db.commit()

                                        end_msg = {
                                            "type": "warning_end",
                                            "timestamp": now_dt.isoformat(),
                                            "face_id": face_id,
                                            "total_duration": total_duration,
                                            "violation_id": violation_id,
                                        }
                                        self._schedule_broadcast(end_msg)
                                except Exception as e:
                                    db.rollback()
                                    logger.error(f"Error updating violation ended_at: {e}")
                                finally:
                                    db.close()

                            self.eye_service.clear_warning_saved(face_id)

                # Throttle status message
                now_time = time.time()
                if now_time - last_ws_status_time >= ws_status_interval:
                    status_msg = {
                        "type": "status",
                        "timestamp": datetime.utcnow().isoformat(),
                        "camera_running": True,
                        "faces_detected": result.get("faces_detected", False),
                        "faces": result.get("faces", []),
                    }
                    self._schedule_broadcast(status_msg)
                    last_ws_status_time = now_time

                # Rate limit loop (~30 FPS)
                time.sleep(1 / 30)

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong camera loop: {e}")
        finally:
            with self._lock:
                self.is_running = False
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                if self.eye_service is not None:
                    self.eye_service.close()
                    self.eye_service = None
                logger.info("Camera loop đã dừng và giải phóng tài nguyên.")

    def _schedule_broadcast(self, message: dict):
        """
        Schedules broadcast coroutine on the asyncio event loop safely from worker thread.
        """
        if self.loop and self.loop.is_running() and self.websocket_connections:
            asyncio.run_coroutine_threadsafe(self.broadcast_websocket(message), self.loop)

    async def broadcast_websocket(self, message: dict):
        """
        Sends message to all connected WebSocket clients.
        """
        if not self.websocket_connections:
            return

        disconnected = set()
        for ws in list(self.websocket_connections):
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self.websocket_connections.discard(ws)

    def register_websocket(self, websocket):
        """Registers a new WebSocket connection."""
        self.websocket_connections.add(websocket)
        logger.info("WebSocket client connected (total: %d)", len(self.websocket_connections))

    def unregister_websocket(self, websocket):
        """Unregisters a WebSocket connection."""
        self.websocket_connections.discard(websocket)
        logger.info("WebSocket client disconnected (total: %d)", len(self.websocket_connections))


# Global singleton instance
camera_service = CameraService()