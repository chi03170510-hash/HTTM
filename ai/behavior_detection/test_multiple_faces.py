
import cv2
import time
import math
import logging
from pathlib import Path

from insightface.app import FaceAnalysis
from ultralytics import YOLO


# ==========================================
# CONFIG
# ==========================================

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
MAX_MISSING_SECONDS = 1.5
MAX_RESULT_AGE = 1.0

DUPLICATE_IOU = 0.45
DUPLICATE_CONTAINMENT = 0.80

MIN_FACE_OVERLAP = 0.35
MIN_MATCH_SCORE = 0.45
MIN_MATCH_MARGIN = 0.15
MAX_NORMALIZED_DISTANCE = 1.8

# OpenCV uses BGR, not RGB.
COLOR_FACE_MATCHED = (0, 255, 0)       # Green
COLOR_FACE_UNMATCHED = (0, 255, 255)   # Yellow
COLOR_YOLO = (255, 0, 255)             # Purple
COLOR_TEXT = (255, 255, 255)           # White

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "behavior_tracking_ch16.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_PATH,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HTTM-CH16")


# ==========================================
# GEOMETRY
# ==========================================

def box_center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_area(box):
    x1, y1, x2, y2 = box

    return max(
        1.0,
        float(max(0, x2 - x1) * max(0, y2 - y1))
    )


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    width = max(
        0,
        min(ax2, bx2) - max(ax1, bx1)
    )

    height = max(
        0,
        min(ay2, by2) - max(ay1, by1)
    )

    return float(width * height)


def box_iou(a, b):
    intersection = intersection_area(a, b)

    union = (
        box_area(a)
        + box_area(b)
        - intersection
    )

    return intersection / max(1.0, union)


def containment_ratio(a, b):
    intersection = intersection_area(a, b)

    smaller_area = min(
        box_area(a),
        box_area(b)
    )

    return intersection / max(1.0, smaller_area)


def face_overlap(face_box, behavior_box):
    return (
        intersection_area(face_box, behavior_box)
        / box_area(face_box)
    )


def center_inside(a, b):
    cx, cy = box_center(a)
    x1, y1, x2, y2 = b

    return (
        x1 <= cx <= x2
        and y1 <= cy <= y2
    )


def normalized_distance(face_box, behavior_box):
    fx, fy = box_center(face_box)
    bx, by = box_center(behavior_box)

    x1, y1, x2, y2 = behavior_box

    diagonal = math.hypot(
        max(1, x2 - x1),
        max(1, y2 - y1)
    )

    return (
        math.hypot(fx - bx, fy - by)
        / max(1.0, diagonal)
    )


# ==========================================
# CENTROID TRACKER
# ==========================================

class CentroidTracker:

    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, now):

        # Expire old tracks
        for track_id in list(self.tracks):

            elapsed = (
                now - self.tracks[track_id]["last_seen"]
            )

            if elapsed > MAX_MISSING_SECONDS:

                del self.tracks[track_id]

                logger.info(
                    "Track %d expired",
                    track_id
                )

        candidates = []

        for track_id, track in self.tracks.items():

            tx, ty = box_center(track["bbox"])

            for index, detection in enumerate(detections):

                # detection is a tuple
                dx, dy = box_center(detection)

                distance = math.hypot(
                    tx - dx,
                    ty - dy
                )

                if distance <= MAX_TRACK_DISTANCE:

                    candidates.append(
                        (distance, track_id, index)
                    )

        candidates.sort(
            key=lambda item: item[0]
        )

        used_tracks = set()
        used_detections = set()

        for _, track_id, index in candidates:

            if track_id in used_tracks:
                continue

            if index in used_detections:
                continue

            self.tracks[track_id]["bbox"] = (
                detections[index]
            )

            self.tracks[track_id]["last_seen"] = now

            used_tracks.add(track_id)
            used_detections.add(index)

        # Create tracks for unmatched faces
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

        # Current face detections only
        return {
            track_id: track["bbox"]
            for track_id, track in self.tracks.items()
            if track["last_seen"] == now
        }


# ==========================================
# LOAD MODELS
# ==========================================

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
        "YOLO ready: %s",
        behavior_model.names
    )

    return face_detector, behavior_model


# ==========================================
# FACE DETECTION
# ==========================================

def detect_faces(detector, frame):

    faces = detector.get(frame)

    height, width = frame.shape[:2]

    detections = []

    for face in faces:

        if float(face.det_score) < FACE_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            (x1, y1, x2, y2)
        )

    return detections


# ==========================================
# YOLO DETECTION
# ==========================================

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
            confidence = float(box.conf[0])

            label = model.names[class_id]

            bbox = tuple(
                map(
                    int,
                    box.xyxy[0].tolist()
                )
            )

            detections.append({
                "bbox": bbox,
                "label": label,
                "confidence": confidence
            })

    return detections


# ==========================================
# FILTER DUPLICATE YOLO BOXES
# ==========================================

def remove_duplicate_behaviors(detections):

    ordered = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True
    )

    filtered = []

    for detection in ordered:

        is_duplicate = False

        for kept in filtered:

            if detection["label"] != kept["label"]:
                continue

            iou = box_iou(
                detection["bbox"],
                kept["bbox"]
            )

            containment = containment_ratio(
                detection["bbox"],
                kept["bbox"]
            )

            if (
                iou >= DUPLICATE_IOU
                or containment >= DUPLICATE_CONTAINMENT
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(detection)

    return filtered


# ==========================================
# MATCHING
# ==========================================

def calculate_match_score(face_box, behavior_box):

    overlap = face_overlap(
        face_box,
        behavior_box
    )

    inside = center_inside(
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
        (
            0.65 * overlap
            + 0.25 * distance_score
            + 0.10 * float(inside)
        )
    )


def associate_behaviors(tracks, behaviors):

    candidates = []

    for index, behavior in enumerate(behaviors):

        for track_id, face_box in tracks.items():

            score = calculate_match_score(
                face_box,
                behavior["bbox"]
            )

            if score >= MIN_MATCH_SCORE:

                candidates.append(
                    (score, track_id, index)
                )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    matches = {}
    used_tracks = set()
    used_behaviors = set()

    for score, track_id, index in candidates:

        if track_id in used_tracks:
            continue

        if index in used_behaviors:
            continue

        competing_tracks = [
            s
            for s, t, b in candidates
            if b == index and t != track_id
        ]

        if competing_tracks:
            if score - max(competing_tracks) < MIN_MATCH_MARGIN:
                continue

        competing_behaviors = [
            s
            for s, t, b in candidates
            if t == track_id and b != index
        ]

        if competing_behaviors:
            if score - max(competing_behaviors) < MIN_MATCH_MARGIN:
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


# ==========================================
# DRAW
# ==========================================

def draw_text(frame, text, x, y, color, scale=0.5):

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


def draw_results(frame, tracks, behaviors, matches):

    matched_ids = {
        id(behavior)
        for behavior in matches.values()
    }

    # --------------------------------------
    # YOLO: ALWAYS PURPLE
    # --------------------------------------
    for behavior in behaviors:

        x1, y1, x2, y2 = behavior["bbox"]

        is_matched = (
            id(behavior) in matched_ids
        )

        # Always purple, even if unmatched
        color = COLOR_YOLO

        if is_matched:
            prefix = "YOLO MATCHED"
        else:
            prefix = "YOLO UNMATCHED"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        draw_text(
            frame,
            (
                f"{prefix}: "
                f"{behavior['label']} "
                f"{behavior['confidence']:.2f}"
            ),
            x1,
            max(20, y1 - 10),
            color,
            0.45
        )

    # --------------------------------------
    # INSIGHTFACE TRACKS
    # --------------------------------------
    for track_id, face_box in tracks.items():

        x1, y1, x2, y2 = face_box

        behavior = matches.get(track_id)

        if behavior is None:

            color = COLOR_FACE_UNMATCHED
            label = f"Track {track_id} | NO MATCH"

        else:

            color = COLOR_FACE_MATCHED

            label = (
                f"Track {track_id} | "
                f"{behavior['label']} "
                f"{behavior['confidence']:.2f}"
            )

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


# ==========================================
# MAIN
# ==========================================

def main():

    cap = None

    try:

        face_detector, behavior_model = load_models()

        tracker = CentroidTracker()

        logger.info(
            "Connecting DroidCam CAMERA_ID=%d",
            CAMERA_ID
        )

        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError("Cannot open DroidCam")

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

        frame_index = 0

        tracks = {}
        behaviors = []
        matches = {}
        unmatched = []

        face_ai_ms = 0.0
        yolo_ai_ms = 0.0

        last_face_time = 0.0
        last_yolo_time = 0.0

        previous_time = time.perf_counter()
        fps = 0.0

        previous_match_state = None

        logger.info("CH-16 started")

        print("\nCONTROLS:")
        print("Q: Quit")
        print("S: Save screenshot")

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:
                logger.warning("Cannot read camera frame")
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

            # Face AI
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

                face_ai_ms = (
                    now - start
                ) * 1000

            # YOLO AI
            if run_yolo:

                start = time.perf_counter()

                raw_behaviors = detect_behaviors(
                    behavior_model,
                    frame
                )

                behaviors = remove_duplicate_behaviors(
                    raw_behaviors
                )

                now = time.perf_counter()

                last_yolo_time = now

                yolo_ai_ms = (
                    now - start
                ) * 1000

            # Clear stale display results
            now = time.perf_counter()

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

            # Match
            matches, unmatched = associate_behaviors(
                tracks,
                behaviors
            )

            # Log only when state changes
            match_state = (
                len(tracks),
                len(behaviors),
                tuple(
                    sorted(
                        (track_id, behavior["label"])
                        for track_id, behavior in matches.items()
                    )
                ),
                len(unmatched)
            )

            if match_state != previous_match_state:

                logger.info(
                    "Tracks=%d | Behaviors=%d | "
                    "Matched=%d | Unmatched=%d",
                    len(tracks),
                    len(behaviors),
                    len(matches),
                    len(unmatched)
                )

                previous_match_state = match_state

            # Draw
            draw_results(
                frame,
                tracks,
                behaviors,
                matches
            )

            # FPS
            now = time.perf_counter()
            delta = now - previous_time

            if delta > 0:
                fps = (
                    0.8 * fps
                    + 0.2 * (1.0 / delta)
                )

            previous_time = now

            # Status
            status_lines = [
                f"FPS: {fps:.1f}",
                f"Tracks: {len(tracks)}",
                f"Behaviors: {len(behaviors)}",
                f"Matched: {len(matches)}",
                f"Unmatched: {len(unmatched)}",
                f"Face AI: {face_ai_ms:.0f} ms",
                f"YOLO AI: {yolo_ai_ms:.0f} ms"
            ]

            for index, text in enumerate(status_lines):

                draw_text(
                    frame,
                    text,
                    10,
                    25 + index * 24,
                    COLOR_TEXT,
                    0.52
                )

            draw_text(
                frame,
                "Q: Quit | S: Screenshot",
                10,
                FRAME_HEIGHT - 15,
                COLOR_TEXT,
                0.5
            )

            cv2.imshow(
                "HTTM - CH16 Purple YOLO Boxes",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):

                screenshot_path = (
                    LOG_DIR
                    / f"ch16_{int(time.time())}.jpg"
                )

                success = cv2.imwrite(
                    str(screenshot_path),
                    frame
                )

                if success:
                    logger.info(
                        "Screenshot saved: %s",
                        screenshot_path
                    )
                else:
                    logger.error(
                        "Screenshot save failed"
                    )

    except Exception:
        logger.exception("CH-16 failed")

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        logger.info("CH-16 stopped")


if __name__ == "__main__":
    main()
