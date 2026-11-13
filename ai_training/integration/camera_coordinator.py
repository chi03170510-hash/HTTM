import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

INTEGRATION = Path(__file__).resolve().parent
ROOT = INTEGRATION.parents[1]

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


def _worker_python():
    hybrid_python = ROOT / ".venv-hybrid/Scripts/python.exe"
    return hybrid_python if hybrid_python.is_file() else Path(sys.executable)


def main():
    ai_python = _worker_python()
    ai_script = INTEGRATION / "smart_classroom_stream_worker.py"

    for path in (ai_python, ai_script):
        if not path.is_file():
            raise FileNotFoundError(path)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    ai_worker = None

    try:
        ai_worker = Worker(ai_python, ai_script)

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
                ai_result = ai_worker.process_frame(image_path, frame_id)
                elapsed = time.perf_counter() - start

                objects = ai_result.get("objects") or []
                behaviors = ai_result.get("behaviors") or []
                occupancy = int(ai_result.get("occupancy", 0))

                for obj in objects:
                    x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                    color = (0, 255, 0) if str(obj["class_name"]).lower() == "person" else (255, 165, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{obj['class_name']} {obj['confidence']:.2f}",
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                for beh in behaviors:
                    x1, y1, x2, y2 = [int(v) for v in beh["bbox"]]
                    color = (255, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"BEH: {beh['class_name']} {beh['confidence']:.2f}",
                        (x1, max(25, y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                cv2.putText(
                    frame,
                    f"Frame {frame_id}/{MAX_FRAMES} | Occupancy: {occupancy} | {elapsed:.2f}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("HTTM Camera Integration", frame)

                print(
                    f"Frame {frame_id}: "
                    f"occupancy={occupancy}, "
                    f"objects={len(objects)}, "
                    f"behaviors={len(behaviors)}, "
                    f"inference={ai_result.get('inference_time_ms', 0.0):.1f}ms, "
                    f"total={elapsed:.2f}s",
                    flush=True,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if ai_worker is not None:
            ai_worker.close()


if __name__ == "__main__":
    main()
