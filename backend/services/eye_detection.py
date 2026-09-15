import os
import sys
import time
import importlib.util
from typing import Dict, List, Any
import cv2
import mediapipe as mp
import numpy as np

# Add ai/eye_detection directory to sys.path to import ear_utils
AI_MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ai", "eye_detection")
)
if AI_MODULE_DIR not in sys.path:
    sys.path.insert(0, AI_MODULE_DIR)

from ear_utils import get_eye_points, calculate_ear

# Load EAR_THRESHOLD directly from ai/eye_detection/config.py to avoid sys.modules['config'] collision
try:
    ai_config_path = os.path.join(AI_MODULE_DIR, "config.py")
    spec = importlib.util.spec_from_file_location("ai_eye_config", ai_config_path)
    ai_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai_cfg)
    EAR_THRESHOLD = getattr(ai_cfg, "EAR_THRESHOLD", 0.10)
except Exception:
    EAR_THRESHOLD = 0.10


# Constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
WARNING_TIME = 5.0


class EyeDetectionService:
    """
    Wraps MediaPipe FaceLandmarker logic for driver eye monitoring.
    Stateful: tracks closed duration and warning saved flags for each face_id.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found at: {model_path}")

        self.model_path = model_path
        
        # Initialize MediaPipe FaceLandmarker
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=10,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # State tracking
        self.eye_closed_start: Dict[int, float] = {}
        self.warning_saved: Dict[int, bool] = {}
        self.start_time = time.time()
        self.last_timestamp_ms = -1
        self.frame_index = 0

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single BGR frame and returns eye monitoring analysis.
        """
        if frame is None or frame.size == 0:
            return {"faces_detected": False, "faces": []}

        height, width, _ = frame.shape
        now = time.time()

        # Timestamp in ms (must be strictly monotonically increasing for MediaPipe VIDEO mode)
        timestamp_ms = int((now - self.start_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect landmarks
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        faces_data: List[Dict[str, Any]] = []

        if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
            current_face_ids = set()

            for face_id, face_landmarks in enumerate(detection_result.face_landmarks):
                current_face_ids.add(face_id)

                left_eye_points = get_eye_points(face_landmarks, LEFT_EYE, width, height)
                right_eye_points = get_eye_points(face_landmarks, RIGHT_EYE, width, height)

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    eye_state = "CLOSED"
                    if face_id not in self.eye_closed_start:
                        self.eye_closed_start[face_id] = now
                    closed_duration = round(now - self.eye_closed_start[face_id], 2)
                else:
                    eye_state = "OPEN"
                    self.eye_closed_start.pop(face_id, None)
                    closed_duration = 0.0

                is_warning = closed_duration >= WARNING_TIME

                faces_data.append({
                    "face_id": face_id,
                    "eye_state": eye_state,
                    "left_ear": round(float(left_ear), 3),
                    "right_ear": round(float(right_ear), 3),
                    "avg_ear": round(float(avg_ear), 3),
                    "closed_duration": closed_duration,
                    "is_warning": is_warning,
                })

            # Clean up timers for faces no longer detected in this frame
            disappeared_faces = [fid for fid in self.eye_closed_start if fid not in current_face_ids]
            for fid in disappeared_faces:
                self.eye_closed_start.pop(fid, None)

        else:
            # No face detected -> reset all timers
            self.eye_closed_start.clear()

        self.frame_index += 1

        return {
            "faces_detected": len(faces_data) > 0,
            "faces": faces_data,
        }

    def reset(self):
        """Resets all timers and frame counters."""
        self.eye_closed_start.clear()
        self.warning_saved.clear()
        self.start_time = time.time()
        self.last_timestamp_ms = -1
        self.frame_index = 0

    def mark_warning_saved(self, face_id: int):
        """Marks that violation image has been saved for this warning session."""
        self.warning_saved[face_id] = True

    def is_warning_saved(self, face_id: int) -> bool:
        """Returns True if violation has already been saved for this warning session."""
        return self.warning_saved.get(face_id, False)

    def clear_warning_saved(self, face_id: int):
        """Clears warning saved mark when the user opens their eyes."""
        self.warning_saved.pop(face_id, None)

    def close(self):
        """Releases MediaPipe landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
