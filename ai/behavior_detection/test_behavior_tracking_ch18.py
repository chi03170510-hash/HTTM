
import cv2
import time
import math
import logging
from pathlib import Path
from collections import deque
from datetime import datetime

from insightface.app import FaceAnalysis
from ultralytics import YOLO

from behavior_state_manager import BehaviorStateManager
from behavior_timer import BehaviorTimer


# ============================================
# CONFIG
# ============================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FACE_INTERVAL = 3
YOLO_INTERVAL = 3

YOLO_IMG_SIZE = 416
YOLO_CONFIDENCE = 0.35
FACE_CONFIDENCE = 0.60

MAX_TRACK_DISTANCE = 100
MAX_FACE_MISSING = 1.5
MAX_RESULT_AGE = 1.0

MIN_FACE_OVERLAP = 0.35
MIN_MATCH_SCORE = 0.45
MIN_MATCH_MARGIN = 0.15
MAX_NORMALIZED_DISTANCE = 1.8

DUPLICATE_IOU = 0.45
DUPLICATE_CONTAINMENT = 0.80

MIN_CONFIRM_FRAMES = 3
MAX_BEHAVIOR_MISSING = 0.8
WARNING_SECONDS = 5.0

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)

LOG_DIR = BASE_DIR / "logs"
WARNING_DIR = LOG_DIR / "warnings"

LOG_DIR.mkdir(parents=True, exist_ok=True)
WARNING_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "behavior_tracking_ch18.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HTTM-CH18")


# ============================================
# GEOMETRY
# ============================================

def center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def area(box):
    x1, y1, x2, y2 = box
    return max(
        1.0,
        float(max(0, x2 - x1) * max(0, y2 - y1))
    )


def intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))

    return float(width * height)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / max(1.0, union)


def containment(a, b):
    return intersection(a, b) / min(area(a), area(b))


def face_overlap(face_box, behavior_box):
    return (
        intersection(face_box, behavior_box)
        / area(face_box)
    )


def face_center_inside(face_box, behavior_box):
    cx, cy = center(face_box)
    x1, y1, x2, y2 = behavior_box

    return x1 <= cx <= x2 and y1 <= cy <= y2


def normalized_distance(face_box, behavior_box):
    fx, fy = center(face_box)
    bx, by = center(behavior_box)

    x1, y1, x2, y2 = behavior_box

    diagonal = math.hypot(
        max(1, x2 - x1),
        max(1, y2 - y1)
    )

    return (
        math.hypot(fx - bx, fy - by)
        / max(1.0, diagonal)
    )


# ============================================
# CENTROID TRACKER
# ============================================

class CentroidTracker:

    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, now):

        for track_id in list(self.tracks):

            elapsed = (
                now - self.tracks[track_id]["last_seen"]
            )

            if elapsed > MAX_FACE_MISSING:
                del self.tracks[track_id]
                logger.info(
                    "Track %d expired",
                    track_id
                )

        candidates = []

        for track_id, track in self.tracks.items():

            tx, ty = center(track["bbox"])

            for index, bbox in enumerate(detections):

                dx, dy = center(bbox)
                distance = math.hypot(
                    tx - dx,
                    ty - dy
                )

                if distance <= MAX_TRACK_DISTANCE:
                    candidates.append(
                        (distance, track_id, index)
                    )

        candidates.sort(key=lambda item: item[0])

        used_tracks = set()
        used_detections = set()

        for _, track_id, index in candidates:

            if (
                track_id in used_tracks
                or index in used_detections
            ):
                continue

            self.tracks[track_id]["bbox"] = (
                detections[index]
            )
            self.tracks[track_id]["last_seen"] = now

            used_tracks.add(track_id)
            used_detections.add(index)

        for index, bbox in enumerate(detections):

            if index in used_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "bbox": bbox,
                "last_seen": now
            }

            logger.info(
                "Created Track ID %d",
                track_id
            )

        return {
            track_id: track["bbox"]
            for track_id, track in self.tracks.items()
            if track["last_seen"] == now
        }


# ============================================
# LOAD MODELS
# ============================================

def load_models():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    logger.info("Loading InsightFace...")

    face_detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"]
    )

    face_detector.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    logger.info("InsightFace ready")
    logger.info("Loading YOLO...")

    behavior_model = YOLO(str(MODEL_PATH))

    logger.info(
        "YOLO classes: %s",
        behavior_model.names
    )

    return face_detector, behavior_model


# ============================================
# FACE DETECTION
# ============================================

def detect_faces(detector, frame):

    height, width = frame.shape[:2]
    detections = []

    for face in detector.get(frame):

        if float(face.det_score) < FACE_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, face.bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 > x1 and y2 > y1:
            detections.append(
                (x1, y1, x2, y2)
            )

    return detections


# ============================================
# YOLO DETECTION
# ============================================

def detect_behaviors(model, frame):

    results = model.predict(
        source=frame,
        imgsz=YOLO_IMG_SIZE,
        conf=YOLO_CONFIDENCE,
        device="cpu",
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            detections.append({
                "bbox": tuple(
                    map(int, box.xyxy[0].tolist())
                ),
                "label": model.names[class_id],
                "confidence": float(box.conf[0])
            })

    return detections


# ============================================
# FILTER DUPLICATES
# ============================================

def remove_duplicates(detections):

    ordered = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    filtered = []

    for detection in ordered:

        duplicate = False

        for kept in filtered:

            if detection["label"] != kept["label"]:
                continue

            if (
                iou(
                    detection["bbox"],
                    kept["bbox"]
                ) >= DUPLICATE_IOU
                or containment(
                    detection["bbox"],
                    kept["bbox"]
                ) >= DUPLICATE_CONTAINMENT
            ):
                duplicate = True
                break

        if not duplicate:
            filtered.append(detection)

    return filtered


# ============================================
# MATCHING
# ============================================

def match_score(face_box, behavior_box):

    overlap = face_overlap(
        face_box,
        behavior_box
    )

    inside = face_center_inside(
        face_box,
        behavior_box
    )

    distance = normalized_distance(
        face_box,
        behavior_box
    )

    if overlap < MIN_FACE_OVERLAP and not inside:
        return 0.0

    if distance > MAX_NORMALIZED_DISTANCE:
        return 0.0

    distance_score = max(
        0.0,
        1.0 - distance / MAX_NORMALIZED_DISTANCE
    )

    return min(
        1.0,
        0.65 * overlap
        + 0.25 * distance_score
        + 0.10 * float(inside)
    )


def associate_behaviors(tracks, behaviors):

    candidates = []

    for index, behavior in enumerate(behaviors):

        for track_id, face_box in tracks.items():

            score = match_score(
                face_box,
                behavior["bbox"]
            )

            if score >= MIN_MATCH_SCORE:
                candidates.append(
                    (score, track_id, index)
                )

    candidates.sort(reverse=True)

    matches = {}
    used_tracks = set()
    used_behaviors = set()

    for score, track_id, index in candidates:

        if (
            track_id in used_tracks
            or index in used_behaviors
        ):
            continue

        competing_tracks = [
            s for s, t, b in candidates
            if b == index and t != track_id
        ]

        if competing_tracks and (
            score - max(competing_tracks)
            < MIN_MATCH_MARGIN
        ):
            continue

        competing_behaviors = [
            s for s, t, b in candidates
            if t == track_id and b != index
        ]

        if competing_behaviors and (
            score - max(competing_behaviors)
            < MIN_MATCH_MARGIN
        ):
            continue

        matches[track_id] = behaviors[index]

        used_tracks.add(track_id)
        used_behaviors.add(index)

    unmatched = [
        behavior
        for index, behavior in enumerate(behaviors)
        if index not in used_behaviors
    ]

    return matches, unmatched


# ============================================
# DRAW HELPERS
# ============================================

def draw_text(
    frame,
    text,
    x,
    y,
    color,
    scale=0.5
):
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


def draw_results(
    frame,
    tracks,
    behaviors,
    matches,
    states,
    timers
):

    matched_ids = {
        id(behavior)
        for behavior in matches.values()
    }

    # YOLO boxes: purple
    for behavior in behaviors:

        x1, y1, x2, y2 = behavior["bbox"]

        status = (
            "MATCHED"
            if id(behavior) in matched_ids
            else "UNMATCHED"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            PURPLE,
            3
        )

        draw_text(
            frame,
            (
                f"YOLO {status}: "
                f"{behavior['label']} "
                f"{behavior['confidence']:.2f}"
            ),
            x1,
            max(20, y1 - 10),
            PURPLE,
            0.45
        )

    # Face boxes and timer
    for track_id, bbox in tracks.items():

        x1, y1, x2, y2 = bbox

        state = states.get(track_id)

        if state is None:
            status = "NORMAL"
            behavior_label = None
        else:
            status = state.status
            behavior_label = state.behavior

        if status == "CONFIRMED":
            color = GREEN
        elif status == "SUSPECTED":
            color = ORANGE
        elif status == "LOST":
            color = GRAY
        else:
            color = YELLOW

        label = f"Track {track_id} | {status}"

        if behavior_label:
            label += f" | {behavior_label}"

        timer = timers.get(track_id)

        if timer is not None:
            label += (
                f" | {timer.elapsed:.1f}/"
                f"{WARNING_SECONDS:.0f}s"
            )

            if timer.warned:
                label += " | WARNING"
                color = RED

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        draw_text(
            frame,
            label,
            x1,
            max(20, y1 - 10),
            color,
            0.45
        )


# ============================================
# MAIN
# ============================================

def main():

    cap = None

    try:

        face_detector, behavior_model = (
            load_models()
        )

        tracker = CentroidTracker()

        state_manager = BehaviorStateManager(
            min_confirm_frames=MIN_CONFIRM_FRAMES,
            max_behavior_missing=MAX_BEHAVIOR_MISSING,
            max_face_missing=MAX_FACE_MISSING
        )

        behavior_timer = BehaviorTimer(
            warning_seconds=WARNING_SECONDS
        )

        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open DroidCam"
            )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        logger.info("DroidCam connected")
        logger.info("CH-18 started")

        frame_index = 0
        yolo_inference_id = 0

        tracks = {}
        behaviors = []
        states = {}

        face_ai_ms = 0.0
        yolo_ai_ms = 0.0

        last_face_time = 0.0
        last_yolo_time = 0.0

        frame_times = deque(maxlen=30)

        previous_behavior_states = {}

        print("\nQ = Quit | S = Screenshot\n")

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:
                logger.warning(
                    "Cannot read camera frame"
                )
                break

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            frame_index += 1

            run_face = (
                frame_index == 1
                or frame_index % FACE_INTERVAL == 0
            )

            run_yolo = (
                frame_index == 1
                or frame_index % YOLO_INTERVAL == 0
            )

            # InsightFace
            if run_face:

                start = time.perf_counter()

                face_boxes = detect_faces(
                    face_detector,
                    frame
                )

                now = time.perf_counter()

                tracks = tracker.update(
                    face_boxes,
                    now
                )

                last_face_time = now
                face_ai_ms = (now - start) * 1000

            # YOLO
            if run_yolo:

                start = time.perf_counter()

                raw_behaviors = detect_behaviors(
                    behavior_model,
                    frame
                )

                behaviors = remove_duplicates(
                    raw_behaviors
                )

                yolo_inference_id += 1

                now = time.perf_counter()

                last_yolo_time = now
                yolo_ai_ms = (now - start) * 1000

            now = time.perf_counter()

            # Clear stale results
            if (
                last_face_time == 0
                or now - last_face_time > MAX_RESULT_AGE
            ):
                tracks = {}

            if (
                last_yolo_time == 0
                or now - last_yolo_time > MAX_RESULT_AGE
            ):
                behaviors = []

            matches, unmatched = associate_behaviors(
                tracks,
                behaviors
            )

            # CH-17: behavior states
            states = state_manager.update(
                visible_tracks=tracks,
                matches=matches,
                inference_id=yolo_inference_id,
                now=time.monotonic()
            )

            # CH-18: behavior timers
            timers, warnings = behavior_timer.update(
                states=states,
                now=time.monotonic()
            )

            # Log state changes
            current_behavior_states = {
                track_id: (
                    state.status,
                    state.behavior
                )
                for track_id, state in states.items()
            }

            for track_id, current in (
                current_behavior_states.items()
            ):

                previous = (
                    previous_behavior_states.get(
                        track_id
                    )
                )

                if previous != current:

                    logger.info(
                        "Track %d | State=%s | Behavior=%s",
                        track_id,
                        current[0],
                        current[1]
                    )

            previous_behavior_states = (
                current_behavior_states
            )

            # Draw detections and timer
            draw_results(
                frame,
                tracks,
                behaviors,
                matches,
                states,
                timers
            )

            # Warning: log and save image once
            for warning in warnings:

                track_id = warning["track_id"]
                behavior = warning["behavior"]
                elapsed = warning["elapsed"]

                logger.warning(
                    "POSSIBLE BEHAVIOR | "
                    "Track=%d | Behavior=%s | "
                    "Elapsed=%.1fs",
                    track_id,
                    behavior,
                    elapsed
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                screenshot_path = (
                    WARNING_DIR
                    / (
                        f"track_{track_id}_"
                        f"{behavior}_{timestamp}.jpg"
                    )
                )

                if cv2.imwrite(
                    str(screenshot_path),
                    frame
                ):
                    logger.info(
                        "Warning screenshot saved: %s",
                        screenshot_path
                    )

            # FPS
            frame_times.append(
                time.perf_counter()
            )

            if len(frame_times) >= 2:

                elapsed = (
                    frame_times[-1]
                    - frame_times[0]
                )

                fps = (
                    (len(frame_times) - 1) / elapsed
                    if elapsed > 0
                    else 0.0
                )
            else:
                fps = 0.0

            confirmed_count = sum(
                1
                for state in states.values()
                if state.status == "CONFIRMED"
            )

            status_lines = [
                f"FPS: {fps:.1f}",
                f"Tracks: {len(tracks)}",
                f"Behaviors: {len(behaviors)}",
                f"Matched: {len(matches)}",
                f"Confirmed: {confirmed_count}",
                f"Face AI: {face_ai_ms:.0f} ms",
                f"YOLO AI: {yolo_ai_ms:.0f} ms"
            ]

            for index, text in enumerate(
                status_lines
            ):

                draw_text(
                    frame,
                    text,
                    10,
                    25 + index * 24,
                    WHITE,
                    0.52
                )

            draw_text(
                frame,
                "Q: Quit | S: Screenshot",
                10,
                FRAME_HEIGHT - 15,
                WHITE,
                0.5
            )

            cv2.imshow(
                "HTTM - CH18 Behavior Timer",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                screenshot_path = (
                    LOG_DIR
                    / f"ch18_{timestamp}.jpg"
                )

                if cv2.imwrite(
                    str(screenshot_path),
                    frame
                ):
                    logger.info(
                        "Screenshot saved: %s",
                        screenshot_path
                    )

    except Exception:

        logger.exception(
            "CH-18 failed"
        )

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        logger.info(
            "CH-18 stopped"
        )


if __name__ == "__main__":
    main()
