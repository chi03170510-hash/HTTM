"""HTTM CH-20G: face recognition + behavior alerts (experimental)."""
import logging
import math
import time
from datetime import datetime

import cv2
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from behavior_state_manager import BehaviorStateManager
from behavior_timer import BehaviorTimer
from student_face_recognizer import StudentFaceRecognizer
from warning_database import WarningDatabase
from test_behavior_tracking_ch18 import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, FACE_INTERVAL, YOLO_INTERVAL,
    MAX_RESULT_AGE, MAX_FACE_MISSING, MAX_TRACK_DISTANCE,
    MIN_CONFIRM_FRAMES, MAX_BEHAVIOR_MISSING, WARNING_SECONDS,
    GREEN, YELLOW, WHITE, LOG_DIR, WARNING_DIR, MODEL_PATH,
    CentroidTracker, detect_behaviors, remove_duplicates,
    associate_behaviors, draw_results, draw_text,
)

CONFIRM_FRAMES = 3
IDENTITY_TIMEOUT = 2.0  # experimental; never reuse across different Track IDs
FACE_CONFIDENCE = 0.60
LOG_INTERVAL = 1.0

LOG_DIR.mkdir(parents=True, exist_ok=True)
WARNING_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HTTM-CH20G")


def detect_faces(detector, frame):
    height, width = frame.shape[:2]
    results = []
    for face in detector.get(frame):
        if float(face.det_score) < FACE_CONFIDENCE:
            continue
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            continue
        x1, y1, x2, y2 = map(int, face.bbox)
        box = (max(0, min(x1, width - 1)),
               max(0, min(y1, height - 1)),
               max(0, min(x2, width)), max(0, min(y2, height)))
        if box[2] > box[0] and box[3] > box[1]:
            results.append({"bbox": box, "embedding": embedding})
    return results


def match_faces_to_tracks(tracks, faces):
    """Assign each newly detected face to one visible track by center distance."""
    candidates = []
    for track_id, box in tracks.items():
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for index, face in enumerate(faces):
            f = face["bbox"]
            fx, fy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
            distance = math.hypot(cx - fx, cy - fy)
            if distance <= MAX_TRACK_DISTANCE:
                candidates.append((distance, track_id, index))
    assignments, used_tracks, used_faces = {}, set(), set()
    for _, track_id, index in sorted(candidates):
        if track_id not in used_tracks and index not in used_faces:
            assignments[track_id] = faces[index]
            used_tracks.add(track_id)
            used_faces.add(index)
    return assignments


class StudentIdentityManager:
    def __init__(self, confirm_frames=CONFIRM_FRAMES, timeout=IDENTITY_TIMEOUT):
        self.confirm_frames = confirm_frames
        self.timeout = timeout
        self.records = {}

    def update(self, track_id, result, now):
        record = self.records.setdefault(track_id, {
            "student_id": None, "candidate": None, "hits": 0,
            "last_match": None,
        })
        before = record["student_id"]
        if result["status"] == "MATCH" and result["student_id"] is not None:
            candidate = result["student_id"]
            if candidate == record["candidate"]:
                record["hits"] += 1
            else:
                record["candidate"] = candidate
                record["hits"] = 1
                if record["student_id"] != candidate:
                    record["student_id"] = None
                    record["last_match"] = None
            if record["hits"] >= self.confirm_frames:
                record["student_id"] = candidate
                record["last_match"] = now
            elif record["student_id"] == candidate:
                record["last_match"] = now
        else:
            record["candidate"] = None
            record["hits"] = 0
        if (record["last_match"] is None or
                now - record["last_match"] > self.timeout):
            record["student_id"] = None
        if before != record["student_id"]:
            logger.info("IDENTITY CHANGE | Track=%s | Student=%s",
                        track_id, record["student_id"] or "Unknown")
        return record

    def cleanup(self, visible_tracks):
        # Never carry an identity to a newly created Track ID.
        for track_id in list(self.records):
            if track_id not in visible_tracks:
                del self.records[track_id]

    def student_id(self, track_id, now):
        record = self.records.get(track_id)
        if not record or record["last_match"] is None:
            return None
        if now - record["last_match"] > self.timeout:
            return None
        return record["student_id"]


def save_warning(database, identities, frame, warning, now):
    track_id = warning["track_id"]
    behavior = warning["behavior"]
    elapsed = float(warning["elapsed"])
    student_id = identities.student_id(track_id, now)
    logger.warning("POSSIBLE BEHAVIOR | Track=%s | Student=%s | "
                   "Behavior=%s | Elapsed=%.1fs",
                   track_id, student_id or "Unknown", behavior, elapsed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = WARNING_DIR / f"ch20_track_{track_id}_{timestamp}.jpg"
    saved_path = str(image_path) if cv2.imwrite(str(image_path), frame) else None
    if saved_path:
        logger.info("Warning screenshot saved: %s", saved_path)
    else:
        logger.error("Warning screenshot failed: %s", image_path)
    try:
        warning_id = database.add_warning(
            track_id=track_id, student_id=student_id,
            behavior=behavior, duration=elapsed, image_path=saved_path)
        logger.info("SQLite warning saved | ID=%s | Track=%s | Student=%s",
                    warning_id, track_id, student_id or "Unknown")
    except Exception:
        logger.exception("SQLite warning save failed")


def main():
    cap = None
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing YOLO model: {MODEL_PATH}")
        logger.info("Loading InsightFace...")
        detector = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"])
        detector.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("InsightFace ready")
        logger.info("Loading YOLO...")
        model = YOLO(str(MODEL_PATH))
        logger.info("YOLO classes: %s", model.names)
        recognizer = StudentFaceRecognizer()
        tracker = CentroidTracker()
        identities = StudentIdentityManager()
        state_manager = BehaviorStateManager(
            min_confirm_frames=MIN_CONFIRM_FRAMES,
            max_behavior_missing=MAX_BEHAVIOR_MISSING,
            max_face_missing=MAX_FACE_MISSING)
        timer = BehaviorTimer(warning_seconds=WARNING_SECONDS)
        database = WarningDatabase()

        # Same camera opening approach as working CH-18.
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open DroidCam at index {CAMERA_ID}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("DroidCam connected | CH-20G started")
        print("\nQ = Quit | S = Screenshot\n")

        frame_index = 0
        inference_id = 0
        tracks, behaviors = {}, []
        last_face_time = last_yolo_time = 0.0
        last_log = {}
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Cannot read camera frame")
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_index += 1
            run_face = frame_index == 1 or frame_index % FACE_INTERVAL == 0
            run_yolo = frame_index == 1 or frame_index % YOLO_INTERVAL == 0

            if run_face:
                faces = detect_faces(detector, frame)
                now = time.monotonic()
                tracks = tracker.update([f["bbox"] for f in faces], now)
                last_face_time = now
                identities.cleanup(tracks.keys())
                assignments = match_faces_to_tracks(tracks, faces)
                for track_id, face in assignments.items():
                    result = recognizer.recognize(face["embedding"])
                    identity = identities.update(track_id, result, now)
                    if now - last_log.get(track_id, 0) >= LOG_INTERVAL:
                        logger.info(
                            "FACE RECOGNITION | Track=%s | Status=%s | "
                            "Similarity=%.4f | Candidate=%s | Hits=%s/%s | Confirmed=%s",
                            track_id, result["status"], result["similarity"],
                            result["student_id"], identity["hits"], CONFIRM_FRAMES,
                            identities.student_id(track_id, now) or "Unknown")
                        last_log[track_id] = now
                last_log = {k: v for k, v in last_log.items() if k in tracks}

            if run_yolo:
                behaviors = remove_duplicates(detect_behaviors(model, frame))
                inference_id += 1
                last_yolo_time = time.monotonic()

            now = time.monotonic()
            if not last_face_time or now - last_face_time > MAX_RESULT_AGE:
                tracks = {}
                identities.cleanup(tracks.keys())
            if not last_yolo_time or now - last_yolo_time > MAX_RESULT_AGE:
                behaviors = []

            # CH-18 interfaces: both return exactly two values.
            matches, unmatched = associate_behaviors(tracks, behaviors)
            states = state_manager.update(
                visible_tracks=tracks, matches=matches,
                inference_id=inference_id, now=now)
            timers, warnings = timer.update(states=states, now=now)

            display = frame.copy()
            draw_results(display, tracks, behaviors, matches, states, timers)
            for track_id, bbox in tracks.items():
                student_id = identities.student_id(track_id, now)
                draw_text(display, f"Student: {student_id or 'Unknown'}",
                          bbox[0], max(20, bbox[1] - 32),
                          GREEN if student_id else YELLOW, 0.5)
            for warning in warnings:
                save_warning(database, identities, display, warning, now)

            draw_text(display, f"Tracks: {len(tracks)} | Matches: {len(matches)}",
                      10, 25, WHITE, 0.5)
            cv2.imshow("HTTM - CH20G", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = LOG_DIR / f"ch20_manual_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                if cv2.imwrite(str(path), display):
                    logger.info("Manual screenshot saved: %s", path)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception:
        logger.exception("CH-20G runtime error")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("CH-20G stopped")


if __name__ == "__main__":
    main()
