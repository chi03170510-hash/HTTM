
import cv2
import time
import math
import logging
from pathlib import Path
from insightface.app import FaceAnalysis

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

DETECTION_SIZE = (640, 640)
MIN_DETECTION_SCORE = 0.60
AI_FRAME_INTERVAL = 3

MAX_TRACK_DISTANCE = 100
MAX_MISSING_SECONDS = 1.5

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "face_tracking.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HTTM-Face-Tracking")


# ==========================================
# SIMPLE CENTROID TRACKER
# ==========================================
class CentroidTracker:
    def __init__(self):
        self.next_track_id = 1
        self.tracks = {}

    @staticmethod
    def center(bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

    def update(self, detections, now):
        # Remove tracks that have been missing too long.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] > MAX_MISSING_SECONDS
        ]

        for track_id in expired:
            del self.tracks[track_id]
            logger.info("Track %d expired", track_id)

        # Build candidate matches.
        candidates = []

        for track_id, track in self.tracks.items():
            tx, ty = self.center(track["bbox"])

            for detection_index, detection in enumerate(detections):
                dx, dy = self.center(detection["bbox"])

                distance = math.hypot(tx - dx, ty - dy)

                if distance <= MAX_TRACK_DISTANCE:
                    candidates.append(
                        (distance, track_id, detection_index)
                    )

        # Greedy one-to-one matching by distance.
        candidates.sort(key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()

        for distance, track_id, detection_index in candidates:
            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            detection = detections[detection_index]

            self.tracks[track_id]["bbox"] = detection["bbox"]
            self.tracks[track_id]["score"] = detection["score"]
            self.tracks[track_id]["last_seen"] = now

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        # Create new tracks for unmatched detections.
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "score": detection["score"],
                "last_seen": now
            }

            logger.info("Created Track ID %d", track_id)

        # Return only tracks observed in this update.
        return {
            track_id: self.tracks[track_id].copy()
            for track_id in matched_tracks
            if track_id in self.tracks
        } | {
            track_id: track.copy()
            for track_id, track in self.tracks.items()
            if track["last_seen"] == now
        }


# ==========================================
# FACE DETECTOR
# ==========================================
def initialize_detector():
    logger.info("Loading InsightFace...")

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    logger.info("InsightFace ready")
    return app


def detect_faces(app, frame):
    faces = app.get(frame)
    detections = []

    height, width = frame.shape[:2]

    for face in faces:
        score = float(face.det_score)

        if score < MIN_DETECTION_SCORE:
            continue

        x1, y1, x2, y2 = map(int, face.bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score
        })

    return detections


# ==========================================
# DRAW TRACKS
# ==========================================
def draw_tracks(frame, tracks):
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = track["bbox"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Track ID: {track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


# ==========================================
# MAIN
# ==========================================
def main():
    cap = None

    try:
        detector = initialize_detector()
        tracker = CentroidTracker()

        logger.info("Opening DroidCam: %d", CAMERA_ID)

        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError("Cannot open DroidCam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        logger.info("DroidCam connected")

        frame_index = 0
        visible_tracks = {}

        fps = 0.0
        ai_time_ms = 0.0
        previous_time = time.perf_counter()

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Failed to read camera frame")
                break

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            frame_index += 1
            now = time.perf_counter()

            if (
                frame_index == 1
                or frame_index % AI_FRAME_INTERVAL == 0
            ):
                ai_start = time.perf_counter()

                detections = detect_faces(detector, frame)

                ai_time_ms = (
                    time.perf_counter() - ai_start
                ) * 1000

                visible_tracks = tracker.update(
                    detections,
                    time.perf_counter()
                )

            draw_tracks(frame, visible_tracks)

            elapsed = now - previous_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = (
                    current_fps if fps == 0
                    else 0.9 * fps + 0.1 * current_fps
                )

            previous_time = now

            cv2.putText(
                frame,
                f"Active tracks: {len(visible_tracks)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | AI: {ai_time_ms:.1f} ms",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "HTTM - Face Tracking CH-11",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception:
        logger.exception("Tracking failed")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()
        logger.info("Program stopped")


if __name__ == "__main__":
    main()
