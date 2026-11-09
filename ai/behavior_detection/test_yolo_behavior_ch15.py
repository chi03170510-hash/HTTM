
import cv2
import time
import logging
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

CAMERA_ID = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

IMG_SIZE = 416
CONFIDENCE = 0.35

# Run AI once every N frames
PROCESS_EVERY_N_FRAMES = 3

# Behavior labels
BEHAVIOR_COLORS = {
    "sleeping": (0, 0, 255),
    "using_phone": (0, 165, 255)
}

# ==========================================
# LOGGING
# ==========================================
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "yolo_behavior_ch15.log"

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

logger = logging.getLogger("HTTM-CH15")


# ==========================================
# LOAD MODEL
# ==========================================
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    logger.info("Loading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    logger.info("YOLO model loaded")
    logger.info("Classes: %s", model.names)

    return model


# ==========================================
# DETECT BEHAVIORS
# ==========================================
def detect_behaviors(model, frame):
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
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

            behavior = model.names.get(
                class_id,
                f"class_{class_id}"
            )

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            detections.append({
                "behavior": behavior,
                "confidence": confidence,
                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            })

    return detections


# ==========================================
# DRAW DETECTIONS
# ==========================================
def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        behavior = detection["behavior"]
        confidence = detection["confidence"]

        color = BEHAVIOR_COLORS.get(
            behavior,
            (0, 255, 0)
        )

        label = (
            f"{behavior} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label_y = max(25, y1 - 10)

        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )


# ==========================================
# MAIN
# ==========================================
def main():
    cap = None

    try:
        model = load_model()

        logger.info(
            "Connecting DroidCam: CAMERA_ID=%d",
            CAMERA_ID
        )

        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError(
                "Cannot connect to DroidCam"
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

        frame_count = 0

        last_detections = []
        last_ai_time_ms = 0.0

        previous_time = time.perf_counter()
        fps = 0.0

        logger.info("CH-15 started")

        print("\nCONTROLS:")
        print("Q: Quit")
        print("S: Save screenshot")

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning(
                    "Cannot read camera frame"
                )
                break

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            frame_count += 1

            # ==================================
            # RUN YOLO EVERY N FRAMES
            # ==================================
            if (
                frame_count == 1
                or frame_count
                % PROCESS_EVERY_N_FRAMES == 0
            ):
                ai_start = time.perf_counter()

                last_detections = detect_behaviors(
                    model,
                    frame
                )

                last_ai_time_ms = (
                    time.perf_counter() - ai_start
                ) * 1000

            # ==================================
            # DRAW LAST DETECTIONS
            # ==================================
            draw_detections(
                frame,
                last_detections
            )

            # ==================================
            # FPS
            # ==================================
            current_time = time.perf_counter()

            delta = current_time - previous_time

            if delta > 0:
                instant_fps = 1.0 / delta

                fps = (
                    0.8 * fps
                    + 0.2 * instant_fps
                )

            previous_time = current_time

            # ==================================
            # DISPLAY INFORMATION
            # ==================================
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"AI: {last_ai_time_ms:.0f} ms",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Detections: {len(last_detections)}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q: Quit | S: Screenshot",
                (15, FRAME_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "HTTM - CH15 YOLO Behavior Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                screenshot_path = (
                    LOG_DIR
                    / f"ch15_{int(time.time())}.jpg"
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
                        "Failed to save screenshot"
                    )

    except Exception:
        logger.exception("CH-15 failed")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        logger.info("CH-15 stopped")


if __name__ == "__main__":
    main()
