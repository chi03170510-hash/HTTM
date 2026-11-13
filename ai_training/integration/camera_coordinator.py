import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

INTEGRATION = Path(__file__).resolve().parent
ROOT = INTEGRATION.parents[1]

sys.path.insert(0, str(INTEGRATION))
from person_face_association_v4 import associate_v4

CAMERA_INDEX = 0
MAX_FRAMES = 10
WORKER_TIMEOUT = 90


class Worker:
    def __init__(self, python_path, script_path):
        self.process = subprocess.Popen(
            [str(python_path), "-u", str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.responses = queue.Queue()
        self.reader = threading.Thread(
            target=self._read_output,
            daemon=True,
        )
        self.reader.start()

    def _read_output(self):
        try:
            for line in self.process.stdout:
                self.responses.put(("line", line))
        finally:
            self.responses.put(("eof", None))

    def process_frame(self, image_path, frame_id):
        request = {
            "frame_id": frame_id,
            "image": str(image_path),
        }

        if self.process.poll() is not None:
            raise RuntimeError("Worker exited")

        self.process.stdin.write(
            json.dumps(request) + "\n"
        )
        self.process.stdin.flush()

        try:
            kind, line = self.responses.get(
                timeout=WORKER_TIMEOUT
            )
        except queue.Empty:
            raise TimeoutError(
                f"Worker timeout on frame {frame_id}"
            )

        if kind != "line":
            raise RuntimeError("Worker closed output")

        result = json.loads(line)

        if "error" in result:
            raise RuntimeError(result["error"])

        if result.get("frame_id") != frame_id:
            raise RuntimeError("Frame ID mismatch")

        if Path(result["image"]).resolve() != image_path.resolve():
            raise RuntimeError("Image mismatch")

        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

        if self.process.stdout:
            self.process.stdout.close()


def main():
    face_python = ROOT / ".venv/Scripts/python.exe"
    hybrid_python = ROOT / ".venv-hybrid/Scripts/python.exe"

    face_script = INTEGRATION / "face_id_stream_worker.py"
    hybrid_script = INTEGRATION / "hybrid_v7_stream_worker.py"

    for path in (
        face_python,
        hybrid_python,
        face_script,
        hybrid_script,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Cannot open iVCam camera")

    face_worker = None
    hybrid_worker = None

    try:
        face_worker = Worker(face_python, face_script)
        hybrid_worker = Worker(hybrid_python, hybrid_script)

        with tempfile.TemporaryDirectory(
            prefix="httm_camera_"
        ) as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"

            for frame_id in range(1, MAX_FRAMES + 1):
                ok, frame = cap.read()

                if not ok:
                    raise RuntimeError("Camera read failed")

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError("Cannot write frame")

                start = time.perf_counter()

                def _run_timed(worker):
                    t0 = time.perf_counter()
                    res = worker.process_frame(image_path, frame_id)
                    return res, time.perf_counter() - t0

                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_face = pool.submit(_run_timed, face_worker)
                    fut_hybrid = pool.submit(_run_timed, hybrid_worker)
                    face_result, face_time = fut_face.result()
                    hybrid_result, hybrid_time = fut_hybrid.result()

                face_tracks = [
                    (
                        f"frame_{frame_id}_face_{i}",
                        item["face_bbox"],
                        item["student_id"],
                        item["score"],
                    )
                    for i, item in enumerate(
                        face_result["faces"]
                    )
                ]

                matches = associate_v4(
                    face_tracks,
                    hybrid_result["people"],
                    adaptive_close_up=True,
                )

                for match in matches:
                    box = match["person_bbox"]

                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
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
                        (0, 255, 0),
                        2,
                    )

                elapsed = time.perf_counter() - start

                cv2.putText(
                    frame,
                    f"Frame {frame_id}/{MAX_FRAMES} "
                    f"| {elapsed:.1f}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("HTTM Camera Integration", frame)

                print(
                    f"Frame {frame_id}: "
                    f"faces={face_result['face_count']}, "
                    f"people={hybrid_result['person_count']}, "
                    f"matched={len(matches)}, "
                    f"face={face_time:.2f}s, "
                    f"hybrid={hybrid_time:.2f}s, "
                    f"total={elapsed:.2f}s",
                    flush=True,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if face_worker is not None:
            face_worker.close()

        if hybrid_worker is not None:
            hybrid_worker.close()


if __name__ == "__main__":
    main()

