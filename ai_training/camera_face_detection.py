import cv2
import time
import threading
from pathlib import Path
import mediapipe as mp

CAMERA_INDEX = 2
TARGET_SIZE = 640
DISPLAY_FPS_LIMIT = 30
AI_INTERVAL = 0.30
MAX_FACES = 10

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_landmarker.task"
)

latest_frame = None
latest_frame_id = 0
latest_result = []
camera_fps = 0.0
ai_fps = 0.0
ai_count = 0

frame_lock = threading.Lock()
result_lock = threading.Lock()
stop_event = threading.Event()


def resize_with_padding(frame):
    h, w = frame.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h))

    top = (TARGET_SIZE - new_h) // 2
    bottom = TARGET_SIZE - new_h - top
    left = (TARGET_SIZE - new_w) // 2
    right = TARGET_SIZE - new_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )


def camera_worker():
    global latest_frame, latest_frame_id, camera_fps

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open iVCam")
        stop_event.set()
        return

    previous_capture = None

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Cannot read frame")
                stop_event.set()
                break

            processed = resize_with_padding(frame)
            now = time.perf_counter()

            if previous_capture is not None:
                instant = 1.0 / max(
                    now - previous_capture, 1e-6
                )
                camera_fps = (
                    instant if camera_fps == 0
                    else 0.9 * camera_fps + 0.1 * instant
                )

            previous_capture = now

            with frame_lock:
                latest_frame = processed
                latest_frame_id += 1

    finally:
        cap.release()


def ai_worker():
    global latest_result, ai_fps, ai_count

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=MAX_FACES
    )

    last_started = 0.0
    last_completed = None
    last_processed_id = -1

    try:
        with FaceLandmarker.create_from_options(options) as detector:
            print("FaceLandmarker loaded successfully")

            while not stop_event.is_set():
                now = time.perf_counter()

                if now - last_started < AI_INTERVAL:
                    stop_event.wait(0.005)
                    continue

                with frame_lock:
                    if (
                        latest_frame is None
                        or latest_frame_id == last_processed_id
                    ):
                        frame = None
                    else:
                        frame = latest_frame.copy()
                        frame_id = latest_frame_id

                if frame is None:
                    stop_event.wait(0.005)
                    continue

                last_started = time.perf_counter()
                last_processed_id = frame_id

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect(mp_image)

                boxes = []
                h, w = frame.shape[:2]

                for landmarks in result.face_landmarks:
                    xs = [p.x for p in landmarks]
                    ys = [p.y for p in landmarks]

                    x1 = max(0, int(min(xs) * w))
                    y1 = max(0, int(min(ys) * h))
                    x2 = min(w - 1, int(max(xs) * w))
                    y2 = min(h - 1, int(max(ys) * h))

                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))

                completed = time.perf_counter()

                if last_completed is not None:
                    instant = 1.0 / max(
                        completed - last_completed, 1e-6
                    )
                    ai_fps = (
                        instant if ai_fps == 0
                        else 0.8 * ai_fps + 0.2 * instant
                    )

                last_completed = completed

                with result_lock:
                    latest_result = boxes
                    ai_count += 1

    except Exception as exc:
        print("Face AI ERROR:", repr(exc))
        stop_event.set()


def draw_text(frame, text, y, color):
    cv2.putText(
        frame, text, (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, color, 2, cv2.LINE_AA
    )


def main():
    if not MODEL_PATH.is_file():
        print("Model not found:", MODEL_PATH)
        return

    camera_thread = threading.Thread(
        target=camera_worker, daemon=True
    )
    ai_thread = threading.Thread(
        target=ai_worker, daemon=True
    )

    camera_thread.start()
    ai_thread.start()

    display_fps = 0.0
    previous_display = None
    next_display = time.perf_counter()
    display_interval = 1.0 / DISPLAY_FPS_LIMIT

    print("HTTM - CH05 Face Detection")
    print("Press Q to exit")

    try:
        while not stop_event.is_set():
            now = time.perf_counter()

            if now < next_display:
                stop_event.wait(
                    min(next_display - now, 0.01)
                )
                continue

            next_display = now + display_interval

            with frame_lock:
                frame = (
                    None if latest_frame is None
                    else latest_frame.copy()
                )
                current_camera_fps = camera_fps

            if frame is None:
                stop_event.wait(0.01)
                continue

            with result_lock:
                boxes = list(latest_result)
                current_ai_fps = ai_fps
                current_ai_count = ai_count

            for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Face {i}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            display_now = time.perf_counter()

            if previous_display is not None:
                instant = 1.0 / max(
                    display_now - previous_display, 1e-6
                )
                display_fps = (
                    instant if display_fps == 0
                    else 0.9 * display_fps + 0.1 * instant
                )

            previous_display = display_now

            draw_text(
                frame,
                f"Camera FPS: {current_camera_fps:.1f}",
                35, (0, 255, 0)
            )
            draw_text(
                frame,
                f"Video FPS: {display_fps:.1f}",
                65, (0, 255, 0)
            )
            draw_text(
                frame,
                f"Face AI FPS: {current_ai_fps:.1f}",
                95, (0, 255, 255)
            )
            draw_text(
                frame,
                f"Faces: {len(boxes)} | AI count: {current_ai_count}",
                125, (255, 255, 255)
            )

            cv2.imshow(
                "HTTM - CH05 Face Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        camera_thread.join(timeout=3)
        ai_thread.join(timeout=3)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
