import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

from camera_coordinator import Worker, ROOT, INTEGRATION, CAMERA_INDEX

sys.path.insert(0, str(ROOT / "ai_training" / "recognition"))
from face_tracker import FaceTracker

MAX_AI_FRAMES = 20


def ai_loop(frame_queue, result_queue, stop_event):
    worker = None
    tracker = FaceTracker(iou_threshold=0.3, max_age=2.0)

    try:
        worker = Worker(
            ROOT / ".venv" / "Scripts" / "python.exe",
            INTEGRATION / "face_id_stream_worker.py",
        )

        with tempfile.TemporaryDirectory(prefix="httm_tracking_") as folder:
            image_path = Path(folder) / "frame.jpg"

            for frame_id in range(1, MAX_AI_FRAMES + 1):
                if stop_event.is_set():
                    break

                try:
                    captured_at, frame = frame_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError("Cannot save frame")

                started = time.perf_counter()
                result = worker.process_frame(image_path, frame_id)

                detections = [
                    (
                        tuple(face["face_bbox"]),
                        face["student_id"],
                        face["score"],
                    )
                    for face in result["faces"]
                ]

                tracks = tracker.update(
                    detections,
                    time.perf_counter(),
                )

                result_queue.put({
                    "frame_id": frame_id,
                    "tracks": tracks,
                    "elapsed": time.perf_counter() - started,
                    "captured_at": captured_at,
                })

    except Exception as exc:
        result_queue.put({"error": str(exc)})

    finally:
        if worker is not None:
            worker.close()

        result_queue.put({"finished": True})


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=ai_loop,
        args=(frame_queue, result_queue, stop_event),
        daemon=True,
    )

    completed = 0
    error = None
    finished = False

    try:
        thread.start()

        while True:
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError("Cannot read camera")

            captured_at = time.perf_counter()

            if not stop_event.is_set():
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    frame_queue.put_nowait((captured_at, frame.copy()))
                except queue.Full:
                    pass

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break

                if "error" in result:
                    error = result["error"]
                    stop_event.set()

                elif "finished" in result:
                    finished = True

                else:
                    completed += 1

                    print(
                        f"Frame {result['frame_id']}: "
                        f"tracks={result['tracks']}, "
                        f"AI={result['elapsed']:.2f}s, "
                        f"age={time.perf_counter() - result['captured_at']:.2f}s",
                        flush=True,
                    )

                    if completed >= MAX_AI_FRAMES:
                        stop_event.set()

            cv2.putText(
                frame,
                f"FaceTracker: {completed}/{MAX_AI_FRAMES} | Q: Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("HTTM FaceTracker Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            if error is not None or finished:
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

        if thread.is_alive():
            thread.join(timeout=105)

        if thread.is_alive():
            print("WARNING: AI thread did not stop within timeout.")

    if error is not None:
        raise RuntimeError(error)

    print(f"Completed: {completed}/{MAX_AI_FRAMES}")


if __name__ == "__main__":
    main()
