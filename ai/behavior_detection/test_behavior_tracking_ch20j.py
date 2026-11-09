"""HTTM CH-20J: multi-person identity safety and behavior attribution.

Save as ai/behavior_detection/test_behavior_tracking_ch20j.py.
Depends on the existing CH-18, behavior state/timer, recognition, and warning DB modules.
"""
import logging
import math
import numpy as np
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
    MAX_RESULT_AGE, MIN_CONFIRM_FRAMES, MAX_BEHAVIOR_MISSING,
    WARNING_SECONDS, GREEN, YELLOW, WHITE, LOG_DIR, WARNING_DIR,
    MODEL_PATH, detect_behaviors, remove_duplicates,
    associate_behaviors, draw_results, draw_text,
)

# Experimental thresholds: validate with several known and unknown volunteers.
FACE_CONFIDENCE = 0.60
CONFIRM_FRAMES = 3
IDENTITY_GRACE_SECONDS = 4.0  # Display only during uncertain recognition
WARNING_MAX_MATCH_AGE = 1.5  # Strict: no old identity in SQLite warnings
TRACK_MISSING_SECONDS = 2.5
TRACK_MAX_DISTANCE = 115.0
TRACK_MIN_IOU = 0.10
LOG_INTERVAL = 1.0
# Avoid attributing the same enrolled student to two simultaneously visible faces.
DUPLICATE_IDENTITY_POLICY = "suppress"
# Track association uses facial embeddings as a safety gate when available.
TRACK_EMBEDDING_MIN_SIM = 0.55
TRACK_EMBEDDING_WEIGHT = 1.10
# UNCERTAIN scores are never promoted to a verified MATCH.
DISPLAY_UNCERTAIN_MIN_SCORE = 0.76
DISPLAY_UNCERTAIN_EXTENSION = 2.0

LOG_DIR.mkdir(parents=True, exist_ok=True)
WARNING_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("HTTM-CH20J")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "behavior_tracking_ch20j.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / max(1, area_a + area_b - overlap)


class StableFaceTracker:
    """Greedy one-to-one geometry tracker. Never transfers identity between IDs."""

    def __init__(self):
        self.next_id = 1
        self.records = {}

    def update(self, faces, now):
        for track_id in list(self.records):
            if now - self.records[track_id]["last_seen"] > TRACK_MISSING_SECONDS:
                del self.records[track_id]
                logger.info("Track %d expired", track_id)

        candidates = []
        for track_id, record in self.records.items():
            old_box = record["bbox"]
            ox, oy = center(old_box)
            for face_index, face in enumerate(faces):
                new_box = face["bbox"]
                nx, ny = center(new_box)
                distance = math.hypot(ox - nx, oy - ny)
                overlap = box_iou(old_box, new_box)
                if distance > TRACK_MAX_DISTANCE and overlap < TRACK_MIN_IOU:
                    continue
                similarity = cosine_similarity(record["embedding"], face["embedding"])
                if similarity < TRACK_EMBEDDING_MIN_SIM:
                    continue
                # Favor both geometry and consistent facial appearance.
                cost = (distance / TRACK_MAX_DISTANCE - 0.65 * overlap
                        + TRACK_EMBEDDING_WEIGHT * (1.0 - similarity))
                candidates.append((cost, track_id, face_index))

        used_tracks, used_faces = set(), set()
        assignments = {}
        for _, track_id, face_index in sorted(candidates):
            if track_id in used_tracks or face_index in used_faces:
                continue
            used_tracks.add(track_id)
            used_faces.add(face_index)
            face = faces[face_index]
            self.records[track_id]["bbox"] = face["bbox"]
            self.records[track_id]["embedding"] = face["embedding"].copy()
            self.records[track_id]["last_seen"] = now
            assignments[track_id] = face

        for face_index, face in enumerate(faces):
            if face_index in used_faces:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.records[track_id] = {"bbox": face["bbox"], "embedding": face["embedding"].copy(), "last_seen": now}
            assignments[track_id] = face
            logger.info("Created Track ID %d", track_id)

        # Only tracks actually detected in this frame are visible.
        return {track_id: face["bbox"] for track_id, face in assignments.items()}, assignments


class StudentIdentityManager:
    def __init__(self):
        self.records = {}

    def update(self, track_id, result, now):
        record = self.records.setdefault(track_id, {
            "student_id": None, "candidate": None, "hits": 0,
            "last_match": None, "last_seen": now, "last_support": None,
        })
        previous = record["student_id"]
        record["last_seen"] = now
        status = result.get("status")
        candidate = result.get("student_id")

        if status == "MATCH" and candidate is not None:
            if record["candidate"] == candidate:
                record["hits"] = min(CONFIRM_FRAMES, record["hits"] + 1)
            else:
                record["candidate"] = candidate
                record["hits"] = 1
                # A different positively matched student must not inherit the old ID.
                if record["student_id"] != candidate:
                    record["student_id"] = None
                    record["last_match"] = None
            if record["hits"] >= CONFIRM_FRAMES:
                record["student_id"] = candidate
            if record["student_id"] == candidate:
                record["last_match"] = now
                record["last_support"] = now
        else:
            record["candidate"] = None
            record["hits"] = 0
            # A moderately strong UNCERTAIN score can keep a previously
            # confirmed label visible briefly, but cannot authorize SQLite attribution.
            if (status == "UNCERTAIN" and record["student_id"] is not None
                    and float(result.get("similarity") or 0.0) >= DISPLAY_UNCERTAIN_MIN_SCORE
                    and record["last_match"] is not None
                    and now - record["last_match"] <= IDENTITY_GRACE_SECONDS):
                record["last_support"] = now

        last_support = record["last_support"]
        if (record["last_match"] is None
                or now - record["last_match"] > IDENTITY_GRACE_SECONDS + DISPLAY_UNCERTAIN_EXTENSION
                or last_support is None or now - last_support > IDENTITY_GRACE_SECONDS):
            record["student_id"] = None

        if previous != record["student_id"]:
            logger.info("IDENTITY CHANGE | Track=%s | Student=%s",
                        track_id, record["student_id"] or "Unknown")
        return record

    def cleanup(self, active_track_ids):
        active = set(active_track_ids)
        for track_id in list(self.records):
            if track_id not in active:
                del self.records[track_id]

    def get(self, track_id, now, for_warning=False):
        record = self.records.get(track_id)
        if not record or not record["student_id"] or record["last_match"] is None:
            return None
        age = now - record["last_match"]
        if for_warning:
            return record["student_id"] if age <= WARNING_MAX_MATCH_AGE else None
        support = record.get("last_support")
        if support is None or now - support > IDENTITY_GRACE_SECONDS:
            return None
        return (record["student_id"] if age <=
                IDENTITY_GRACE_SECONDS + DISPLAY_UNCERTAIN_EXTENSION else None)



def ambiguous_student_ids(identities, visible_track_ids, now):
    """Find enrolled IDs claimed by multiple visible tracks; never choose a winner."""
    owners = {}
    for track_id in visible_track_ids:
        student_id = identities.get(track_id, now)
        if student_id is not None:
            owners.setdefault(student_id, []).append(track_id)
    return {student_id: ids for student_id, ids in owners.items() if len(ids) > 1}


def safe_student_id(identities, track_id, now, ambiguous_ids, for_warning=False):
    student_id = identities.get(track_id, now, for_warning=for_warning)
    return None if student_id in ambiguous_ids else student_id


def detect_faces(detector, frame):
    height, width = frame.shape[:2]
    faces = []
    for face in detector.get(frame):
        if float(face.det_score) < FACE_CONFIDENCE:
            continue
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            continue
        x1, y1, x2, y2 = map(int, face.bbox)
        bbox = (max(0, min(x1, width - 1)), max(0, min(y1, height - 1)),
                max(0, min(x2, width)), max(0, min(y2, height)))
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            faces.append({"bbox": bbox, "embedding": embedding})
    return faces


def save_warning(database, identities, frame, warning, now, ambiguous_ids):
    track_id = warning["track_id"]
    behavior = warning["behavior"]
    elapsed = float(warning["elapsed"])
    student_id = safe_student_id(identities, track_id, now, ambiguous_ids, for_warning=True)
    logger.warning("POSSIBLE BEHAVIOR | Track=%s | Student=%s | Behavior=%s | Elapsed=%.1fs",
                   track_id, student_id or "Unknown", behavior, elapsed)
    path = WARNING_DIR / (
        f"ch20j_track_{track_id}_{behavior}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    )
    saved_path = str(path) if cv2.imwrite(str(path), frame) else None
    if saved_path:
        logger.info("Warning screenshot saved: %s", saved_path)
    else:
        logger.error("Warning screenshot failed: %s", path)
    try:
        warning_id = database.add_warning(
            track_id=track_id, student_id=student_id,
            behavior=behavior, duration=elapsed, image_path=saved_path,
        )
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
            providers=["CPUExecutionProvider"],
        )
        detector.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("InsightFace ready")
        logger.info("Loading YOLO...")
        model = YOLO(str(MODEL_PATH))
        logger.info("YOLO classes: %s", model.names)
        recognizer = StudentFaceRecognizer()
        tracker = StableFaceTracker()
        identities = StudentIdentityManager()
        state_manager = BehaviorStateManager(
            min_confirm_frames=MIN_CONFIRM_FRAMES,
            max_behavior_missing=MAX_BEHAVIOR_MISSING,
            max_face_missing=TRACK_MISSING_SECONDS,
        )
        timer = BehaviorTimer(warning_seconds=WARNING_SECONDS)
        database = WarningDatabase()

        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open DroidCam at index {CAMERA_ID}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("DroidCam connected | CH-20J started")
        print("\nQ = Quit | S = Screenshot\n")

        frame_index, inference_id = 0, 0
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
                tracks, assignments = tracker.update(faces, now)
                last_face_time = now
                identities.cleanup(tracker.records.keys())
                for track_id, face in assignments.items():
                    result = recognizer.recognize(face["embedding"])
                    record = identities.update(track_id, result, now)
                    if now - last_log.get(track_id, 0.0) >= LOG_INTERVAL:
                        logger.info(
                            "FACE RECOGNITION | Track=%s | Status=%s | Similarity=%.4f | "
                            "Candidate=%s | Hits=%s/%s | Confirmed=%s",
                            track_id, result["status"], result["similarity"],
                            result["student_id"], record["hits"], CONFIRM_FRAMES,
                            identities.get(track_id, now) or "Unknown",
                        )
                        last_log[track_id] = now
                last_log = {k: v for k, v in last_log.items() if k in tracker.records}

            if run_yolo:
                behaviors = remove_duplicates(detect_behaviors(model, frame))
                inference_id += 1
                last_yolo_time = time.monotonic()

            now = time.monotonic()
            if not last_face_time or now - last_face_time > MAX_RESULT_AGE:
                tracks = {}
            if not last_yolo_time or now - last_yolo_time > MAX_RESULT_AGE:
                behaviors = []

            # CH-18 interfaces: two values each. draw_results expects timers dict.
            matches, unmatched = associate_behaviors(tracks, behaviors)
            states = state_manager.update(
                visible_tracks=tracks, matches=matches,
                inference_id=inference_id, now=now,
            )
            timers, warnings = timer.update(states=states, now=now)
            display = frame.copy()
            draw_results(display, tracks, behaviors, matches, states, timers)
            collisions = ambiguous_student_ids(identities, tracks.keys(), now)
            ambiguous_ids = set(collisions)
            if collisions and now - last_log.get("collision", 0.0) >= LOG_INTERVAL:
                logger.warning("AMBIGUOUS IDENTITY | %s | Suppressing Student ID on display and warnings", collisions)
                last_log["collision"] = now
            for track_id, bbox in tracks.items():
                student_id = safe_student_id(identities, track_id, now, ambiguous_ids)
                draw_text(display, f"Student: {student_id or 'Unknown'}",
                          bbox[0], max(20, bbox[1] - 32),
                          GREEN if student_id else YELLOW, 0.5)
            for warning in warnings:
                save_warning(database, identities, display, warning, now, ambiguous_ids)

            draw_text(display, f"Tracks: {len(tracks)} | Matches: {len(matches)} | Ambiguous: {len(ambiguous_ids)}",
                      10, 25, WHITE, 0.5)
            cv2.imshow("HTTM - CH20J", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = LOG_DIR / f"ch20j_manual_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                if cv2.imwrite(str(path), display):
                    logger.info("Manual screenshot saved: %s", path)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception:
        logger.exception("CH-20J runtime error")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("CH-20J stopped")


if __name__ == "__main__":
    main()
