import cv2
import time
import threading

CAMERA_INDEX = 2
TARGET_SIZE = 640
DISPLAY_FPS_LIMIT = 30
AI_INTERVAL = 0.30  # Khoang 3.3 FPS
SIMULATED_AI_SECONDS = 0.10

latest_frame = None
latest_frame_id = 0
latest_capture_time = 0.0
latest_result = None

frame_lock = threading.Lock()
result_lock = threading.Lock()
stop_event = threading.Event()

camera_fps = 0.0
ai_fps = 0.0
ai_count = 0


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
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )


def camera_worker():
    global latest_frame, latest_frame_id
    global latest_capture_time, camera_fps

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
                print("ERROR: Cannot read camera frame")
                stop_event.set()
                break

            processed = resize_with_padding(frame)
            now = time.perf_counter()

            if previous_capture is not None:
                delta = now - previous_capture
                instant_fps = 1.0 / max(delta, 1e-6)

                if camera_fps == 0:
                    camera_fps = instant_fps
                else:
                    camera_fps = (
                        0.9 * camera_fps + 0.1 * instant_fps
                    )

            previous_capture = now

            with frame_lock:
                latest_frame = processed
                latest_frame_id += 1
                latest_capture_time = now

    finally:
        cap.release()


def ai_worker():
    global latest_result, ai_fps, ai_count

    last_started = 0.0
    last_completed = None
    last_processed_id = -1

    while not stop_event.is_set():
        now = time.perf_counter()

        if now - last_started < AI_INTERVAL:
            stop_event.wait(0.005)
            continue

        with frame_lock:
            if latest_frame is None:
                frame = None
            elif latest_frame_id == last_processed_id:
                frame = None
            else:
                frame = latest_frame.copy()
                frame_id = latest_frame_id
                capture_time = latest_capture_time

        if frame is None:
            stop_event.wait(0.005)
            continue

        last_started = time.perf_counter()
        last_processed_id = frame_id

        # AI GIA LAP - CHUA PHAI HYBRID V7
        if stop_event.wait(SIMULATED_AI_SECONDS):
            break

        h, w = frame.shape[:2]

        bbox = (
            w // 4,
            h // 4,
            3 * w // 4,
            3 * h // 4
        )

        completed = time.perf_counter()
        latency_ms = (completed - capture_time) * 1000

        if last_completed is not None:
            instant_fps = 1.0 / max(
                completed - last_completed, 1e-6
            )

            if ai_fps == 0:
                ai_fps = instant_fps
            else:
                ai_fps = 0.8 * ai_fps + 0.2 * instant_fps

        last_completed = completed

        with result_lock:
            ai_count += 1
            latest_result = {
                "bbox": bbox,
                "frame_id": frame_id,
                "latency_ms": latency_ms,
                "completed_at": completed
            }


def draw_text(frame, text, y, color):
    cv2.putText(
        frame,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


def main():
    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    ai_thread = threading.Thread(
        target=ai_worker,
        daemon=True
    )

    camera_thread.start()
    ai_thread.start()

    display_fps = 0.0
    previous_display = None
    display_interval = 1.0 / DISPLAY_FPS_LIMIT
    next_display = time.perf_counter()

    print("HTTM - CH04 - Two Worker Architecture")
    print("Camera index:", CAMERA_INDEX)
    print("Target display FPS:", DISPLAY_FPS_LIMIT)
    print("Target AI FPS:", round(1 / AI_INTERVAL, 2))
    print("AI mode: SIMULATED")
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
                result = (
                    None if latest_result is None
                    else latest_result.copy()
                )
                current_ai_fps = ai_fps
                current_ai_count = ai_count

            if result is not None:
                x1, y1, x2, y2 = result["bbox"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                draw_text(
                    frame,
                    "AI DEMO - NOT REAL DETECTION",
                    160,
                    (0, 255, 255)
                )

                draw_text(
                    frame,
                    f"AI latency: {result['latency_ms']:.0f} ms",
                    190,
                    (255, 255, 255)
                )

                age_ms = (
                    time.perf_counter()
                    - result["completed_at"]
                ) * 1000

                draw_text(
                    frame,
                    f"Result age: {age_ms:.0f} ms",
                    220,
                    (255, 255, 255)
                )

            display_now = time.perf_counter()

            if previous_display is not None:
                instant_fps = 1.0 / max(
                    display_now - previous_display,
                    1e-6
                )

                if display_fps == 0:
                    display_fps = instant_fps
                else:
                    display_fps = (
                        0.9 * display_fps
                        + 0.1 * instant_fps
                    )

            previous_display = display_now

            draw_text(
                frame,
                f"Camera FPS: {current_camera_fps:.1f}",
                35,
                (0, 255, 0)
            )

            draw_text(
                frame,
                f"Video FPS: {display_fps:.1f}",
                65,
                (0, 255, 0)
            )

            draw_text(
                frame,
                f"AI FPS: {current_ai_fps:.1f}",
                95,
                (0, 255, 255)
            )

            draw_text(
                frame,
                f"AI processed: {current_ai_count}",
                125,
                (255, 255, 255)
            )

            cv2.imshow(
                "HTTM - CH04 Threaded",
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
