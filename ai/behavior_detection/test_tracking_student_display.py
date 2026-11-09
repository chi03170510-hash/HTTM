
import cv2
import time
import math
import sqlite3
import logging
from pathlib import Path
from insightface.app import FaceAnalysis

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "students_faces.db"

CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

AI_FRAME_INTERVAL = 3
MIN_DETECTION_SCORE = 0.60
MAX_TRACK_DISTANCE = 100
MAX_MISSING_SECONDS = 1.5

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "tracking_student_display.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HTTM-CH13")


# ==========================================
# SQLITE
# ==========================================
def load_students():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True
    )

    try:
        rows = conn.execute(
            """
            SELECT student_id, student_name
            FROM students
            ORDER BY student_id
            """
        ).fetchall()
    finally:
        conn.close()

    return dict(rows)


# ==========================================
# TRACKER
# ==========================================
class CentroidTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def center(bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

    def update(self, detections, now):
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] > MAX_MISSING_SECONDS
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]
            logger.info("Track %d expired", track_id)

        candidates = []

        for track_id, track in self.tracks.items():
            tx, ty = self.center(track["bbox"])

            for index, detection in enumerate(detections):
                dx, dy = self.center(detection["bbox"])
                distance = math.hypot(tx - dx, ty - dy)

                if distance <= MAX_TRACK_DISTANCE:
                    candidates.append(
                        (distance, track_id, index)
                    )

        candidates.sort(key=lambda item: item[0])

        used_tracks = set()
        used_detections = set()

        for _, track_id, index in candidates:
            if track_id in used_tracks:
                continue

            if index in used_detections:
                continue

            self.tracks[track_id].update({
                "bbox": detections[index]["bbox"],
                "last_seen": now
            })

            used_tracks.add(track_id)
            used_detections.add(index)

        for index, detection in enumerate(detections):
            if index in used_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "last_seen": now
            }

            logger.info(
                "Created Track ID %d",
                track_id
            )

        return {
            track_id: track.copy()
            for track_id, track in self.tracks.items()
            if track["last_seen"] == now
        }


# ==========================================
# FACE DETECTION
# ==========================================
def initialize_detector():
    logger.info("Loading InsightFace...")

    detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"]
    )

    detector.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    logger.info("InsightFace ready")
    return detector


def detect_faces(detector, frame):
    faces = detector.get(frame)
    detections = []

    height, width = frame.shape[:2]

    for face in faces:
        if float(face.det_score) < MIN_DETECTION_SCORE:
            continue

        x1, y1, x2, y2 = map(int, face.bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": (x1, y1, x2, y2)
        })

    return detections


# ==========================================
# STUDENT MAPPER
# ==========================================
class StudentMapper:
    def __init__(self, students):
        self.students = students
        self.mapping = {}

    def assign(self, track_id, student_id, active_tracks):
        if track_id not in active_tracks:
            logger.warning(
                "Track %d is not currently visible",
                track_id
            )
            return False

        if student_id not in self.students:
            logger.warning(
                "Student %s not found",
                student_id
            )
            return False

        for other_track, assigned_id in self.mapping.items():
            if (
                other_track != track_id
                and assigned_id == student_id
            ):
                logger.warning(
                    "Student %s already assigned to Track %d",
                    student_id,
                    other_track
                )
                return False

        self.mapping[track_id] = student_id

        logger.info(
            "Assigned Track %d -> Student %s",
            track_id,
            student_id
        )
        return True

    def cleanup(self, active_tracks):
        for track_id in list(self.mapping):
            if track_id not in active_tracks:
                del self.mapping[track_id]
                logger.info(
                    "Removed mapping for Track %d",
                    track_id
                )


# ==========================================
# DRAW
# ==========================================
def draw_tracks(frame, tracks, mapper, selected_id):
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = track["bbox"]

        student_id = mapper.mapping.get(track_id)

        if student_id:
            label = f"Track {track_id} | Student {student_id}"
            color = (0, 255, 0)
        else:
            label = f"Track {track_id} | UNASSIGNED"
            color = (0, 255, 255)

        if track_id == selected_id:
            color = (255, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )


# ==========================================
# MAIN
# ==========================================
def main():
    cap = None

    try:
        students = load_students()

        logger.info(
            "Students loaded: %d",
            len(students)
        )

        for student_id, name in students.items():
            logger.info(
                "Student %s: %s",
                student_id,
                name
            )

        detector = initialize_detector()
        tracker = CentroidTracker()
        mapper = StudentMapper(students)

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

        logger.info("DroidCam connected")

        frame_index = 0
        visible_tracks = {}
        selected_id = None

        print("\nCONTROLS:")
        print("N: Select next visible Track ID")
        print("A: Assign Student 001 to selected Track")
        print("D: Remove selected Track assignment")
        print("U: Clear all assignments")
        print("Q: Quit")

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Camera frame unavailable")
                break

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            frame_index += 1

            if (
                frame_index == 1
                or frame_index % AI_FRAME_INTERVAL == 0
            ):
                now = time.perf_counter()

                detections = detect_faces(
                    detector,
                    frame
                )

                visible_tracks = tracker.update(
                    detections,
                    now
                )

                mapper.cleanup(tracker.tracks)

                if selected_id not in tracker.tracks:
                    selected_id = None

            draw_tracks(
                frame,
                visible_tracks,
                mapper,
                selected_id
            )

            selected_text = (
                f"Selected Track: {selected_id}"
                if selected_id is not None
                else "Selected Track: NONE"
            )

            cv2.putText(
                frame,
                selected_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "N: Select | A: Assign 001 | D: Remove",
                (10, FRAME_HEIGHT - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "U: Clear all | Q: Quit",
                (10, FRAME_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "HTTM - CH13 Student Display",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("n"):
                ids = sorted(visible_tracks.keys())

                if not ids:
                    selected_id = None
                    logger.warning(
                        "No visible tracks to select"
                    )
                elif selected_id not in ids:
                    selected_id = ids[0]
                else:
                    index = ids.index(selected_id)
                    selected_id = ids[
                        (index + 1) % len(ids)
                    ]

                logger.info(
                    "Selected Track: %s",
                    selected_id
                )

            elif key == ord("a"):
                if selected_id is None:
                    logger.warning(
                        "Select a track first using N"
                    )
                else:
                    mapper.assign(
                        selected_id,
                        "001",
                        visible_tracks
                    )

            elif key == ord("d"):
                if selected_id in mapper.mapping:
                    del mapper.mapping[selected_id]
                    logger.info(
                        "Removed assignment for Track %d",
                        selected_id
                    )

            elif key == ord("u"):
                mapper.mapping.clear()
                logger.info("Cleared all assignments")

    except Exception:
        logger.exception("CH-13 failed")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()
        logger.info("Program stopped")


if __name__ == "__main__":
    main()
