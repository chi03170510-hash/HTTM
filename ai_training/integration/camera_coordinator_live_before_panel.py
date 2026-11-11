import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2

from camera_coordinator import (
    Worker,
    ROOT,
    INTEGRATION,
    CAMERA_INDEX,
    WORKER_TIMEOUT,
)
from person_face_association_v4 import associate_v4


MAX_AI_FRAMES = 10
RESULT_TTL = 0.6
WINDOW_NAME = "HTTM Camera Live"


def ai_loop(frame_queue, result_queue, stop_event):
    face_worker = None
    hybrid_worker = None

    try:
        face_worker = Worker(
            ROOT / ".venv/Scripts/python.exe",
            INTEGRATION / "face_id_stream_worker.py",
        )

        hybrid_worker = Worker(
            ROOT / ".venv-hybrid/Scripts/python.exe",
            INTEGRATION / "hybrid_v7_stream_worker.py",
        )

        with tempfile.TemporaryDirectory(
            prefix="httm_live_"
        ) as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"

            for frame_id in range(1, MAX_AI_FRAMES + 1):
                if stop_event.is_set():
                    break

                try:
                    captured_at, frame = frame_queue.get(
                        timeout=1
                    )
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError("Cannot save temporary frame")

                started = time.perf_counter()

                face_started = time.perf_counter()
                face_result = face_worker.process_frame(
                    image_path, frame_id
                )
                face_time = time.perf_counter() - face_started

                hybrid_started = time.perf_counter()
                hybrid_result = hybrid_worker.process_frame(
                    image_path, frame_id
                )
                hybrid_time = time.perf_counter() - hybrid_started

                face_tracks = [
                    (
                        f"frame_{frame_id}_face_{i}",
                        face["face_bbox"],
                        face["student_id"],
                        face["score"],
                    )
                    for i, face in enumerate(face_result["faces"])
                ]

                matches = associate_v4(
                    face_tracks,
                    hybrid_result["people"],
                )

                elapsed = time.perf_counter() - started

                result = {
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "completed_at": time.perf_counter(),
                    "face_count": face_result["face_count"],
                    "person_count": hybrid_result["person_count"],
                    "matches": matches,
                    "face_time": face_time,
                    "hybrid_time": hybrid_time,
                    "elapsed": elapsed,
                }

                result_queue.put(result)

    except Exception as exc:
        result_queue.put({"error": str(exc)})

    finally:
        if face_worker is not None:
            face_worker.close()

        if hybrid_worker is not None:
            hybrid_worker.close()

        result_queue.put({"finished": True})


def main():
    for path in (
        ROOT / ".venv/Scripts/python.exe",
        ROOT / ".venv-hybrid/Scripts/python.exe",
        INTEGRATION / "face_id_stream_worker.py",
        INTEGRATION / "hybrid_v7_stream_worker.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=ai_loop,
        args=(frame_queue, result_queue, stop_event),
        daemon=True,
    )

    latest_result = None
    completed = 0
    error = None
    finished = False

    try:
        worker.start()

        while True:
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError("Camera read failed")

            captured_at = time.perf_counter()

            if not stop_event.is_set():
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    frame_queue.put_nowait(
                        (captured_at, frame.copy())
                    )
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
                    latest_result = result
                    completed += 1

                    print(
                        f"Frame {result['frame_id']}: "
                        f"faces={result['face_count']}, "
                        f"people={result['person_count']}, "
                        f"matched={len(result['matches'])}, "
                        f"face={result['face_time']:.2f}s, "
                        f"hybrid={result['hybrid_time']:.2f}s, "
                        f"total={result['elapsed']:.2f}s, "
                        f"result_age="
                        f"{result['completed_at'] - result['captured_at']:.2f}s",
                        flush=True,
                    )

                    if completed >= MAX_AI_FRAMES:
                        stop_event.set()

            if latest_result is not None:
                age = (
                    time.perf_counter()
                    - latest_result["captured_at"]
                )

                if age <= RESULT_TTL:
                    for match in latest_result["matches"]:
                        x1, y1, x2, y2 = map(
                            int, match["person_bbox"]
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 255),
                            2,
                        )

                        label = (
                            f"{match['student_id']} "
                            f"({match['association_status']})"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

            cv2.putText(
                frame,
                f"AI results: {completed}/{MAX_AI_FRAMES} | Q: Quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            if error is not None or finished:
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

        if worker.is_alive():
            print("Waiting for AI worker to close...")
            worker.join(timeout=WORKER_TIMEOUT + 15)

        if worker.is_alive():
            print(
                "WARNING: AI thread did not exit within timeout."
            )

    if error is not None:
        raise RuntimeError(error)

    print(f"Completed AI results: {completed}")


if __name__ == "__main__":
    main()
