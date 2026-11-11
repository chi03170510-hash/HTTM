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
import sys

sys.path.insert(0, str(ROOT / "ai_training" / "recognition"))
from face_tracker import FaceTracker


MAX_AI_FRAMES = 10
RESULT_TTL = 0.6
WINDOW_NAME = "HTTM Camera Live"


def ai_loop(frame_queue, result_queue, stop_event):
    face_worker = None
    hybrid_worker = None
    tracker = FaceTracker(iou_threshold=0.3, max_age=2.0)

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

                detections = [
                    (
                        tuple(face["face_bbox"]),
                        face["student_id"],
                        face["score"],
                    )
                    for face in face_result["faces"]
                ]

                tracked_faces = tracker.update(
                    detections,
                    time.perf_counter(),
                )

                face_tracks = [
                    (
                        track_id,
                        bbox,
                        next(
                            (
                                face["student_id"]
                                for face in face_result["faces"]
                                if tuple(face["face_bbox"]) == bbox
                            ),
                            stable_label,
                        ),
                        score,
                    )
                    for track_id, bbox, stable_label, score
                    in tracked_faces
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
                    "tracked_faces": tracked_faces,
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



def draw_ai_panel(frame, result):
    """Draw historical AI results beside the live camera."""
    import numpy as np

    height, width = frame.shape[:2]
    panel_width = 360

    panel = np.zeros(
        (max(height, 480), panel_width, 3),
        dtype=np.uint8,
    )
    panel[:] = (35, 35, 35)

    def line(text, y, color=(230, 230, 230), scale=0.48):
        cv2.putText(
            panel,
            str(text)[:48],
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )

    line("HTTM - AI MONITORING", 30, (0, 220, 255), 0.62)
    line("Historical AI result - not live", 55)

    if result is None:
        line("Waiting for AI...", 100)
    else:
        age = time.perf_counter() - result["captured_at"]

        line(f"Frame: {result['frame_id']}", 90)
        line(f"Result age: {age:.1f}s", 115)
        line(f"Processing: {result['elapsed']:.2f}s", 140)
        line(f"Faces: {result['face_count']}", 165)
        line(f"People: {result['person_count']}", 190)

        tracked = result.get("tracked_faces", [])
        line(f"Face tracks: {len(tracked)}", 215)

        for track_id, bbox, stable_label, score in tracked[:2]:
            line(
                f"T{track_id} | Stable: {stable_label}",
                240 + (track_id - 1) * 22,
                (0, 255, 255),
                0.43,
            )

        y = 300

        if not result["matches"]:
            line("No associated student", y)
        else:
            for match in result["matches"][:2]:
                line(
                    f"Student: {match.get('student_id', 'Unknown')}",
                    y,
                    (0, 255, 255),
                )
                y += 25

                score = match.get("face_score")
                if isinstance(score, (int, float)):
                    line(f"Face similarity: {score:.3f}", y)
                    y += 25

                line(
                    f"Person: {match.get('person_ref', '?')}",
                    y,
                )
                y += 25

                line(
                    f"Association: {match.get('association_status', '?')}",
                    y,
                    scale=0.40,
                )
                y += 30

                behaviors = match.get("behaviors", {})

                for name, value in behaviors.items():
                    state = (
                        value.get("state", "unknown")
                        if isinstance(value, dict)
                        else "unknown"
                    )
                    line(f"{name}: {state}", y, scale=0.42)
                    y += 22

                y += 15

                if y >= panel.shape[0] - 20:
                    break

    if height < panel.shape[0]:
        frame = cv2.copyMakeBorder(
            frame,
            0,
            panel.shape[0] - height,
            0,
            0,
            cv2.BORDER_CONSTANT,
        )
    elif height > panel.shape[0]:
        panel = cv2.copyMakeBorder(
            panel,
            0,
            height - panel.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
        )

    return np.hstack((frame, panel))

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

            frame = draw_ai_panel(frame, latest_result)

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


