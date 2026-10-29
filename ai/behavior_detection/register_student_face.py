
import cv2
import sqlite3
import numpy as np
import time
import logging
import re

from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

DETECTION_SIZE = (640, 640)
MIN_DETECTION_SCORE = 0.65

REQUIRED_SAMPLES = 10
SAMPLE_INTERVAL = 0.5

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "students_faces.db"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "face_registration.log"

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_PATH,
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("HTTM-Registration")


# ==========================================
# DATABASE
# ==========================================
def initialize_database():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id)
                REFERENCES students(student_id)
        )
    """)

    conn.commit()
    return conn


def save_student(conn, student_id, student_name, embeddings):
    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    # Atomic registration: student and embeddings
    # are committed together.
    with conn:
        conn.execute("""
            INSERT INTO students (
                student_id,
                student_name,
                created_at
            )
            VALUES (?, ?, ?)
        """, (
            student_id,
            student_name,
            timestamp
        ))

        for embedding in embeddings:
            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            if embedding.shape != (512,):
                raise ValueError(
                    "Invalid embedding shape"
                )

            conn.execute("""
                INSERT INTO face_embeddings (
                    student_id,
                    embedding,
                    created_at
                )
                VALUES (?, ?, ?)
            """, (
                student_id,
                embedding.tobytes(),
                timestamp
            ))

    logger.info(
        "Registered student %s with %d samples",
        student_id,
        len(embeddings)
    )


# ==========================================
# INSIGHTFACE
# ==========================================
def initialize_insightface():
    logger.info("Loading InsightFace...")

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=[
            "detection",
            "recognition"
        ],
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    logger.info("InsightFace ready")
    return app


# ==========================================
# CAMERA
# ==========================================
def connect_camera():
    logger.info("Opening DroidCam: %s", CAMERA_ID)

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

    logger.info("DroidCam connected")
    return cap


# ==========================================
# INPUT VALIDATION
# ==========================================
def get_student_info(conn):
    while True:
        student_id = input(
            "Enter Student ID: "
        ).strip()

        if not re.fullmatch(
            r"[A-Za-z0-9_-]{1,32}",
            student_id
        ):
            print(
                "Student ID must contain only "
                "letters, numbers, _ or -"
            )
            continue

        exists = conn.execute(
            "SELECT 1 FROM students WHERE student_id=?",
            (student_id,)
        ).fetchone()

        if exists:
            print(
                "Student ID already registered. "
                "Use another ID."
            )
            continue

        break

    while True:
        student_name = input(
            "Enter Student Name: "
        ).strip()

        if 1 <= len(student_name) <= 100:
            break

        print("Name must be 1-100 characters")

    print("\nStudent information:")
    print("ID:", student_id)
    print("Name:", student_name)

    consent = input(
        "Student agrees to biometric enrollment? "
        "(yes/no): "
    ).strip().lower()

    if consent != "yes":
        print("Registration cancelled")
        return None, None

    return student_id, student_name


# ==========================================
# DRAW TEXT
# ==========================================
def draw_text(frame, text, y, color):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# ==========================================
# MAIN
# ==========================================
def main():
    conn = None
    cap = None

    try:
        conn = initialize_database()

        student_id, student_name = get_student_info(
            conn
        )

        if student_id is None:
            return

        app = initialize_insightface()
        cap = connect_camera()

        embeddings = []
        collecting = False
        last_sample_time = 0

        print("\nControls:")
        print("S = Start collecting samples")
        print("Q = Quit")
        print("Move head slightly between samples")

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.error("Cannot read frame")
                break

            height, width = frame.shape[:2]

            if width > FRAME_WIDTH:
                scale = FRAME_WIDTH / width
                frame = cv2.resize(
                    frame,
                    (
                        FRAME_WIDTH,
                        int(height * scale)
                    )
                )

            # Detect faces
            try:
                faces = app.get(frame)
            except Exception:
                logger.exception(
                    "Face detection failed"
                )
                faces = []

            valid_faces = [
                face for face in faces
                if float(face.det_score)
                >= MIN_DETECTION_SCORE
            ]

            # Draw bounding boxes
            for face in valid_faces:
                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # Collect embeddings
            if collecting:
                now = time.monotonic()

                if (
                    len(faces) == 1
                    and len(valid_faces) == 1
                    and now - last_sample_time
                    >= SAMPLE_INTERVAL
                ):
                    embedding = (
                        valid_faces[0].normed_embedding
                    )

                    if embedding is not None:
                        embedding = np.asarray(
                            embedding,
                            dtype=np.float32
                        )

                        if (
                            embedding.shape == (512,)
                            and np.all(
                                np.isfinite(embedding)
                            )
                        ):
                            embeddings.append(
                                embedding.copy()
                            )

                            last_sample_time = now

                            logger.info(
                                "Collected sample %d/%d",
                                len(embeddings),
                                REQUIRED_SAMPLES
                            )

                if len(embeddings) >= REQUIRED_SAMPLES:
                    save_student(
                        conn,
                        student_id,
                        student_name,
                        embeddings
                    )

                    print(
                        "\nRegistration successful!"
                    )

                    print(
                        "Database:",
                        DB_PATH
                    )
                    break

            # Display status
            draw_text(
                frame,
                f"Student ID: {student_id}",
                30,
                (255, 255, 0)
            )

            draw_text(
                frame,
                f"Faces: {len(faces)}",
                60,
                (255, 255, 0)
            )

            draw_text(
                frame,
                (
                    f"Samples: {len(embeddings)}"
                    f"/{REQUIRED_SAMPLES}"
                ),
                90,
                (255, 255, 0)
            )

            if collecting:
                status = "COLLECTING"
                color = (0, 255, 0)

                if len(faces) != 1:
                    status = "NEED EXACTLY ONE FACE"
                    color = (0, 0, 255)
                elif len(valid_faces) != 1:
                    status = "LOW FACE CONFIDENCE"
                    color = (0, 0, 255)
            else:
                status = "PRESS S TO START"
                color = (0, 255, 255)

            draw_text(
                frame,
                status,
                120,
                color
            )

            cv2.imshow(
                "HTTM - Student Face Registration",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and not collecting:
                collecting = True
                last_sample_time = 0
                logger.info(
                    "Started registration for %s",
                    student_id
                )

            elif key == ord("q"):
                logger.info(
                    "Registration cancelled"
                )
                break

    except KeyboardInterrupt:
        logger.info("Stopped by keyboard")

    except Exception:
        logger.exception(
            "Registration error"
        )

    finally:
        if cap is not None:
            cap.release()

        if conn is not None:
            conn.close()

        cv2.destroyAllWindows()
        logger.info("Program stopped")


if __name__ == "__main__":
    main()
