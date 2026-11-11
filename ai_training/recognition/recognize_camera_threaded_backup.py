import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR / "enrollment"))

from enrollment_utils import DATA_DIR, load_face_model


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.frame_id = 0
        self.frame_time = 0.0
        self.results = []
        self.result_time = 0.0
        self.ai_latency = 0.0
        self.camera_count = 0
        self.ai_count = 0
        self.error = None


def load_database():
    database = {}

    for path in sorted(DATA_DIR.glob("SV*/embeddings.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                student_id = str(data["student_id"])
                embeddings = np.asarray(
                    data["embeddings"], dtype=np.float32
                )

            if student_id != path.parent.name:
                raise ValueError("Student ID mismatch")

            if (
                embeddings.ndim != 2
                or embeddings.shape != (len(embeddings), 512)
                or len(embeddings) == 0
                or not np.all(np.isfinite(embeddings))
            ):
                raise ValueError("Invalid embeddings")

            norms = np.linalg.norm(embeddings, axis=1)
            if np.any(norms <= 1e-8):
                raise ValueError("Zero-norm embedding")

            database[student_id] = embeddings / norms[:, None]

        except (OSError, ValueError, KeyError) as exc:
            print(f"[SKIP] {path}: {exc}")

    return database


def recognize(embedding, database, threshold):
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (512,) or not np.all(np.isfinite(embedding)):
        return "Unknown", 0.0

    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        return "Unknown", 0.0

    embedding = embedding / norm

    scores = {
        sid: float(np.max(refs @ embedding))
        for sid, refs in database.items()
    }

    if not scores:
        return "Unknown", 0.0

    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]

    if best_score < threshold:
        return "Unknown", best_score

    return best_id, best_score


def camera_worker(state, camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        state.error = "Cannot open camera"
        state.stop.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while not state.stop.is_set():
            ret, frame = cap.read()

            if not ret:
                state.error = "Cannot read camera frame"
                state.stop.set()
                break

            with state.lock:
                state.frame = frame
                state.frame_id += 1
                state.frame_time = time.perf_counter()
                state.camera_count += 1
    finally:
        cap.release()


def ai_worker(state, database, threshold):
    try:
        app = load_face_model()
    except Exception as exc:
        state.error = f"Cannot load InsightFace: {exc}"
        state.stop.set()
        return

    last_frame_id = -1

    while not state.stop.is_set():
        with state.lock:
            if state.frame is None or state.frame_id == last_frame_id:
                frame = None
            else:
                frame = state.frame.copy()
                last_frame_id = state.frame_id
                source_time = state.frame_time

        if frame is None:
            state.stop.wait(0.01)
            continue

        start = time.perf_counter()

        try:
            faces = app.get(frame)
            results = []

            for face in faces:
                bbox = tuple(
                    int(value) for value in face.bbox[:4]
                )

                student_id, score = recognize(
                    face.embedding, database, threshold
                )

                results.append((bbox, student_id, score))

        except Exception as exc:
            state.error = f"AI inference error: {exc}"
            state.stop.set()
            break

        elapsed = (time.perf_counter() - start) * 1000

        with state.lock:
            state.results = results
            state.result_time = source_time
            state.ai_latency = elapsed
            state.ai_count += 1


def draw_results(frame, results):
    for bbox, student_id, score in results:
        x1, y1, x2, y2 = bbox

        color = (
            (0, 255, 0)
            if student_id != "Unknown"
            else (0, 165, 255)
        )

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), color, 2
        )

        cv2.putText(
            frame,
            f"{student_id} ({score:.3f})",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def main():
    parser = argparse.ArgumentParser(
        description="HTTM CH-08.3 - Threaded face recognition"
    )
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    if not -1.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between -1 and 1")

    database = load_database()

    if not database:
        print("[ERROR] No valid enrollments")
        return

    for sid, refs in database.items():
        print(f"[LOADED] {sid}: {len(refs)} samples")

    state = SharedState()

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(state, args.camera),
        daemon=True,
    )

    ai_thread = threading.Thread(
        target=ai_worker,
        args=(state, database, args.threshold),
        daemon=True,
    )

    camera_thread.start()
    ai_thread.start()

    previous_time = time.perf_counter()
    previous_camera_count = 0
    previous_ai_count = 0
    video_count = 0

    camera_fps = 0.0
    video_fps = 0.0
    ai_fps = 0.0

    print("[START] Threaded recognition")
    print("Press Q to quit")

    try:
        while not state.stop.is_set():
            with state.lock:
                frame = (
                    state.frame.copy()
                    if state.frame is not None
                    else None
                )
                results = list(state.results)
                result_time = state.result_time
                ai_latency = state.ai_latency
                camera_count = state.camera_count
                ai_count = state.ai_count

            if frame is None:
                time.sleep(0.02)
                continue

            display = frame.copy()

            result_age_ms = (
                (time.perf_counter() - result_time) * 1000
                if result_time > 0
                else 0.0
            )

            # Do not display stale boxes indefinitely.
            if result_time > 0 and result_age_ms <= 1000:
                draw_results(display, results)

            video_count += 1
            now = time.perf_counter()
            interval = now - previous_time

            if interval >= 1.0:
                camera_fps = (
                    camera_count - previous_camera_count
                ) / interval

                ai_fps = (
                    ai_count - previous_ai_count
                ) / interval

                video_fps = video_count / interval

                previous_camera_count = camera_count
                previous_ai_count = ai_count
                previous_time = now
                video_count = 0

            metrics = [
                f"Camera FPS: {camera_fps:.1f}",
                f"Video FPS: {video_fps:.1f}",
                f"AI FPS: {ai_fps:.1f}",
                f"AI latency: {ai_latency:.0f} ms",
                f"Result age: {result_age_ms:.0f} ms",
                f"Faces: {len(results)}",
            ]

            for index, line in enumerate(metrics):
                cv2.putText(
                    display,
                    line,
                    (15, 30 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("HTTM - Threaded Face Recognition", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        state.stop.set()
        camera_thread.join(timeout=3)
        ai_thread.join(timeout=3)
        cv2.destroyAllWindows()

        if state.error:
            print(f"[ERROR] {state.error}")

        print("Camera closed")


if __name__ == "__main__":
    main()
