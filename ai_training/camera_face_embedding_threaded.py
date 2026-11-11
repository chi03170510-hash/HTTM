import cv2
import time
import threading
import numpy as np
from insightface.app import FaceAnalysis

CAMERA_INDEX = 2
TARGET_SIZE = 640
AI_INTERVAL = 0.30
DISPLAY_FPS = 30

latest_frame = None
latest_frame_id = 0
latest_boxes = []
latest_ai_time = 0.0
latest_latency = 0.0

camera_fps = 0.0
ai_fps = 0.0
ai_count = 0

frame_lock = threading.Lock()
result_lock = threading.Lock()
stop_event = threading.Event()


def resize_with_padding(frame):
    h, w = frame.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    resized = cv2.resize(frame, (nw, nh))

    top = (TARGET_SIZE - nh) // 2
    bottom = TARGET_SIZE - nh - top
    left = (TARGET_SIZE - nw) // 2
    right = TARGET_SIZE - nw - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


def camera_worker():
    global latest_frame, latest_frame_id, camera_fps

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        stop_event.set()
        return

    previous = None

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()

            if not ok:
                print("ERROR: Cannot read frame")
                stop_event.set()
                break

            frame = resize_with_padding(frame)
            now = time.perf_counter()

            if previous is not None:
                instant = 1.0 / max(now - previous, 1e-6)
                camera_fps = (
                    instant if camera_fps == 0
                    else 0.9 * camera_fps + 0.1 * instant
                )

            previous = now

            with frame_lock:
                latest_frame = frame
                latest_frame_id += 1

    finally:
        cap.release()


def ai_worker():
    global latest_boxes, latest_ai_time
    global latest_latency, ai_fps, ai_count

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))

        print("InsightFace ready")

        last_start = 0.0
        last_complete = None
        last_id = -1

        while not stop_event.is_set():
            now = time.perf_counter()

            if now - last_start < AI_INTERVAL:
                stop_event.wait(0.005)
                continue

            with frame_lock:
                if (
                    latest_frame is None
                    or latest_frame_id == last_id
                ):
                    frame = None
                else:
                    frame = latest_frame.copy()
                    frame_id = latest_frame_id

            if frame is None:
                stop_event.wait(0.005)
                continue

            last_id = frame_id
            last_start = time.perf_counter()

            faces = app.get(frame)

            boxes = []

            for face in faces:
                embedding = face.embedding

                if embedding is None:
                    continue

                if (
                    embedding.shape != (512,)
                    or not np.all(np.isfinite(embedding))
                ):
                    continue

                x1, y1, x2, y2 = face.bbox.astype(int)
                boxes.append((x1, y1, x2, y2))

            completed = time.perf_counter()
            latency = (completed - last_start) * 1000

            if last_complete is not None:
                instant = 1.0 / max(
                    completed - last_complete, 1e-6
                )
                ai_fps = (
                    instant if ai_fps == 0
                    else 0.8 * ai_fps + 0.2 * instant
                )

            last_complete = completed

            with result_lock:
                latest_boxes = boxes
                latest_ai_time = completed
                latest_latency = latency
                ai_count += 1

    except Exception as exc:
        print("AI ERROR:", repr(exc))
        stop_event.set()


def draw_text(frame, text, y, color):
    cv2.putText(
        frame, text, (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, color, 2, cv2.LINE_AA
    )


def main():
    camera_thread = threading.Thread(
        target=camera_worker, daemon=True
    )
    ai_thread = threading.Thread(
        target=ai_worker, daemon=True
    )

    camera_thread.start()
    ai_thread.start()

    video_fps = 0.0
    previous_display = None
    display_interval = 1.0 / DISPLAY_FPS
    next_display = time.perf_counter()

    print("HTTM CH06 - Threaded Face Embedding")
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
                boxes = list(latest_boxes)
                result_time = latest_ai_time
                latency = latest_latency
                current_ai_fps = ai_fps
                current_ai_count = ai_count

            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    frame, "Embedding OK",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            display_now = time.perf_counter()

            if previous_display is not None:
                instant = 1.0 / max(
                    display_now - previous_display, 1e-6
                )
                video_fps = (
                    instant if video_fps == 0
                    else 0.9 * video_fps + 0.1 * instant
                )

            previous_display = display_now

            result_age = (
                (display_now - result_time) * 1000
                if result_time > 0 else 0
            )

            draw_text(
                frame,
                f"Camera FPS: {current_camera_fps:.1f}",
                30, (0, 255, 0)
            )
            draw_text(
                frame,
                f"Video FPS: {video_fps:.1f}",
                60, (0, 255, 0)
            )
            draw_text(
                frame,
                f"AI FPS: {current_ai_fps:.1f}",
                90, (0, 255, 255)
            )
            draw_text(
                frame,
                f"AI latency: {latency:.0f} ms",
                120, (0, 255, 255)
            )
            draw_text(
                frame,
                f"Result age: {result_age:.0f} ms",
                150, (255, 255, 255)
            )
            draw_text(
                frame,
                f"Faces: {len(boxes)} | AI count: {current_ai_count}",
                180, (255, 255, 255)
            )

            cv2.imshow(
                "HTTM CH06 - Threaded Embedding",
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
