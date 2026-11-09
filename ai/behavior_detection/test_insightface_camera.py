
import cv2
import time
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

PROCESS_EVERY_N_FRAMES = 3
DETECTION_SIZE = (640, 640)
MIN_DETECTION_SCORE = 0.5

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "insightface_log.txt"

# ==========================================
# LOGGING
# ==========================================
logger = logging.getLogger("HTTM-InsightFace")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================
# INITIALIZE INSIGHTFACE
# ==========================================
def initialize_insightface():
    logger.info("Loading InsightFace model...")

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    logger.info("InsightFace initialized successfully")
    return app


# ==========================================
# CONNECT DROIDCAM
# ==========================================
def connect_camera():
    logger.info(
        "Connecting to DroidCam CAMERA_ID=%s",
        CAMERA_ID
    )

    # Do not force DirectShow backend
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        logger.error(
            "Cannot open DroidCam CAMERA_ID=%s",
            CAMERA_ID
        )
        return None

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    logger.info("DroidCam connected successfully")
    return cap


# ==========================================
# DRAW FACE BOUNDING BOX
# ==========================================
def draw_faces(frame, faces):
    height, width = frame.shape[:2]

    for index, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"Face {index + 1} "
            f"({float(face.det_score):.2f})"
        )

        text_y = max(20, y1 - 10)

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame


# ==========================================
# MAIN
# ==========================================
def main():
    logger.info("=" * 45)
    logger.info("HTTM - INSIGHTFACE CAMERA TEST")
    logger.info("=" * 45)

    cap = None

    try:
        app = initialize_insightface()

        cap = connect_camera()

        if cap is None:
            logger.error("Camera initialization failed")
            return

        frame_count = 0
        last_faces = []
        last_ai_time = 0.0

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Cannot read camera frame")
                break

            # ==================================
            # RESIZE FRAME
            # ==================================
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

            frame_count += 1

            # ==================================
            # RUN INSIGHTFACE EVERY N FRAMES
            # ==================================
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                start_time = time.perf_counter()

                try:
                    detected_faces = app.get(frame)

                    last_faces = [
                        face
                        for face in detected_faces
                        if float(face.det_score)
                        >= MIN_DETECTION_SCORE
                    ]

                    last_ai_time = (
                        time.perf_counter() - start_time
                    ) * 1000

                    logger.info(
                        "Frame=%d | Faces=%d | AI=%.1f ms",
                        frame_count,
                        len(last_faces),
                        last_ai_time
                    )

                    # ==================================
                    # FACE EMBEDDING
                    # ==================================
                    for index, face in enumerate(last_faces):
                        embedding = face.normed_embedding

                        if embedding is None:
                            logger.warning(
                                "Face %d has no embedding",
                                index + 1
                            )
                            continue

                        logger.info(
                            "Face %d | Embedding=%s | "
                            "Score=%.3f",
                            index + 1,
                            embedding.shape,
                            float(face.det_score)
                        )

                except Exception:
                    logger.exception(
                        "InsightFace inference error"
                    )
                    last_faces = []

            # ==================================
            # DRAW BOUNDING BOX
            # ==================================
            frame = draw_faces(frame, last_faces)

            # ==================================
            # DISPLAY STATUS
            # ==================================
            cv2.putText(
                frame,
                f"Faces: {len(last_faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"AI: {last_ai_time:.1f} ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to exit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "HTTM - DroidCam InsightFace",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Exit requested")
                break

    except KeyboardInterrupt:
        logger.info("Stopped by keyboard")

    except Exception:
        logger.exception("Unexpected application error")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()
        logger.info("Program stopped")


if __name__ == "__main__":
    main()
