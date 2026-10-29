
import cv2
import time
import logging
from pathlib import Path
from collections import deque
from datetime import datetime

from behavior_state_manager import BehaviorStateManager
from behavior_timer import BehaviorTimer
from warning_database import WarningDatabase

from test_behavior_tracking_ch18 import (
    CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FACE_INTERVAL,
    YOLO_INTERVAL,
    MAX_RESULT_AGE,
    MAX_FACE_MISSING,
    MIN_CONFIRM_FRAMES,
    MAX_BEHAVIOR_MISSING,
    WARNING_SECONDS,
    WHITE,
    LOG_DIR,
    WARNING_DIR,
    CentroidTracker,
    load_models,
    detect_faces,
    detect_behaviors,
    remove_duplicates,
    associate_behaviors,
    draw_results,
    draw_text
)


# ============================================
# CH-19 CONFIG
# ============================================

LOG_DIR.mkdir(parents=True, exist_ok=True)
WARNING_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "behavior_tracking_ch19.log"

logger = logging.getLogger("HTTM-CH19")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================
# SAVE WARNING
# ============================================

def save_warning(
    frame,
    warning,
    warning_database
):
    track_id = warning["track_id"]
    behavior = warning["behavior"]
    elapsed = warning["elapsed"]

    logger.warning(
        "POSSIBLE BEHAVIOR | "
        "Track=%d | Behavior=%s | Elapsed=%.1fs",
        track_id,
        behavior,
        elapsed
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_path = (
        WARNING_DIR
        / f"track_{track_id}_{behavior}_{timestamp}.jpg"
    )

    image_saved = cv2.imwrite(
        str(image_path),
        frame
    )

    if image_saved:
        logger.info(
            "Warning screenshot saved: %s",
            image_path
        )
    else:
        logger.error(
            "Cannot save warning screenshot: %s",
            image_path
        )

    try:
        warning_id = warning_database.add_warning(
            track_id=track_id,
            behavior=behavior,
            duration=elapsed,
            image_path=image_path if image_saved else None,
            student_id=None
        )

        logger.info(
            "SQLite warning saved | "
            "ID=%d | Track=%d | Behavior=%s",
            warning_id,
            track_id,
            behavior
        )

    except Exception:
        logger.exception(
            "Failed to save warning to SQLite"
        )


# ============================================
# MAIN
# ============================================

def main():

    cap = None

    try:

        # Load InsightFace and YOLO
        face_detector, behavior_model = load_models()

        # CH-16: Track IDs
        tracker = CentroidTracker()

        # CH-17: Behavior states
        state_manager = BehaviorStateManager(
            min_confirm_frames=MIN_CONFIRM_FRAMES,
            max_behavior_missing=MAX_BEHAVIOR_MISSING,
            max_face_missing=MAX_FACE_MISSING
        )

        # CH-18: Five-second timer
        behavior_timer = BehaviorTimer(
            warning_seconds=WARNING_SECONDS
        )

        # CH-19: SQLite database
        warning_database = WarningDatabase()

        logger.info(
            "SQLite database initialized"
        )

        # Open DroidCam
        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera ID {CAMERA_ID}"
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
        logger.info("CH-19 started")

        frame_index = 0
        yolo_inference_id = 0

        tracks = {}
        behaviors = {}
        states = {}

        # Store detections as a list
        behaviors = []

        last_face_time = 0.0
        last_yolo_time = 0.0

        face_ai_ms = 0.0
        yolo_ai_ms = 0.0

        frame_times = deque(maxlen=30)
        previous_behavior_states = {}

        print(
            "\nQ = Quit | S = Screenshot\n"
        )

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

            # ====================================
            # FACE DETECTION
            # ====================================

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

            # ====================================
            # YOLO DETECTION
            # ====================================

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
                yolo_ai_ms = (
                    now - start
                ) * 1000

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

            # ====================================
            # CH-16: MATCH YOLO TO TRACK ID
            # ====================================

            matches, unmatched = associate_behaviors(
                tracks,
                behaviors
            )

            # ====================================
            # CH-17: UPDATE STATES
            # ====================================

            current_time = time.monotonic()

            states = state_manager.update(
                visible_tracks=tracks,
                matches=matches,
                inference_id=yolo_inference_id,
                now=current_time
            )

            # ====================================
            # CH-18: UPDATE TIMERS
            # ====================================

            timers, warnings = behavior_timer.update(
                states=states,
                now=current_time
            )

            # Log state transitions
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

            # ====================================
            # DRAW RESULTS
            # ====================================

            draw_results(
                frame,
                tracks,
                behaviors,
                matches,
                states,
                timers
            )

            # ====================================
            # CH-19: SAVE WARNING TO SQLITE
            # ====================================

            for warning in warnings:

                save_warning(
                    frame=frame,
                    warning=warning,
                    warning_database=warning_database
                )

            # ====================================
            # FPS
            # ====================================

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
                "HTTM - CH19 SQLite",
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
                    0.5
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
                "HTTM - CH19 SQLite Warning",
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
                    / f"ch19_{timestamp}.jpg"
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
            "CH-19 failed"
        )

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        logger.info(
            "CH-19 stopped"
        )


if __name__ == "__main__":
    main()
