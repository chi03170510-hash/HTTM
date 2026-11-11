import cv2
import time
import threading

CAMERA_INDEX = 2
TARGET_SIZE = 640
AI_INTERVAL = 0.30

latest_frame = None
latest_result = None
frame_lock = threading.Lock()
result_lock = threading.Lock()
stop_event = threading.Event()

ai_fps = 0.0
ai_count = 0


def resize_with_padding(frame):
    h, w = frame.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)

    new_w = round(w * scale)
    new_h = round(h * scale)

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
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open iVCam")
        stop_event.set()
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Cannot read frame")
                stop_event.set()
                break

            processed = resize_with_padding(frame)

            with frame_lock:
                global latest_frame
                latest_frame = processed

    finally:
        cap.release()


def ai_worker():
    global latest_result, ai_fps, ai_count

    last_run = 0.0
    last_completed = None

    while not stop_event.is_set():
        now = time.perf_counter()

        if now - last_run < AI_INTERVAL:
            stop_event.wait(0.005)
            continue

        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            stop_event.wait(0.01)
            continue

        last_run = time.perf_counter()

        # Simulate AI inference taking 100 ms.
        # Replace this with Hybrid V7 later.
        time.sleep(0.10)

        h, w = frame.shape[:2]
        bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

        completed = time.perf_counter()
        if last_completed is not None:
            ai_fps = 1.0 / max(completed - last_completed, 1e-6)
        last_completed = completed

        with result_lock:
            latest_result = bbox
            ai_count += 1


def main():
    camera_thread = threading.Thread(
        target=camera_worker, daemon=True
    )
    ai_thread = threading.Thread(
        target=ai_worker, daemon=True
    )

    camera_thread.start()
    ai_thread.start()

    previous_display = time.perf_counter()
    display_fps = 0.0

    print("HTTM - Threaded Camera Test")
    print("Press Q to exit")
    print("AI is simulated, not Hybrid V7")

    try:
        while not stop_event.is_set():
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            with result_lock:
                bbox = latest_result
                current_ai_fps = ai_fps
                current_ai_count = ai_count

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 255), 2
                )
                cv2.putText(
                    frame, "AI DEMO",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )

            now = time.perf_counter()
            instant_fps = 1.0 / max(
                now - previous_display, 1e-6
            )
            previous_display = now

            # Smooth the displayed FPS value.
            display_fps = (
                0.9 * display_fps + 0.1 * instant_fps
            )

            cv2.putText(
                frame, f"Video FPS: {display_fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"AI FPS: {current_ai_fps:.1f}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            cv2.putText(
                frame, f"AI processed: {current_ai_count}",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("HTTM - CH04 Threaded", frame)

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
