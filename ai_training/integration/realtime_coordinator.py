import json
import subprocess
import sys
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parent
ROOT = INTEGRATION.parents[1]

sys.path.insert(0, str(INTEGRATION))
from person_face_association_v4 import associate_v4


class StreamWorker:
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

    def process_frame(self, image_path, frame_id):
        if self.process.poll() is not None:
            raise RuntimeError("Worker exited unexpectedly")

        request = {
            "frame_id": frame_id,
            "image": str(image_path),
        }

        self.process.stdin.write(
            json.dumps(request, ensure_ascii=False) + "\n"
        )
        self.process.stdin.flush()

        line = self.process.stdout.readline()

        if not line:
            raise RuntimeError("Worker returned no response")

        result = json.loads(line)

        if "error" in result:
            raise RuntimeError(result["error"])

        if result.get("frame_id") != frame_id:
            raise RuntimeError(
                f"Frame mismatch: expected {frame_id}, "
                f"got {result.get('frame_id')}"
            )

        if Path(result["image"]).resolve() != image_path.resolve():
            raise RuntimeError("Worker image mismatch")

        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
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

    images = [
        ROOT / "ai_training/test_images/classroom_test_01.jpg",
        ROOT / "ai_training/test_images/classroom_test_02.jpg",
    ]

    for path in [face_python, hybrid_python, face_script, hybrid_script, *images]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = ROOT / "ai_training/runs/ch09_coordinator"
    output_dir.mkdir(parents=True, exist_ok=True)

    face_worker = None
    hybrid_worker = None

    try:
        face_worker = StreamWorker(face_python, face_script)
        hybrid_worker = StreamWorker(hybrid_python, hybrid_script)

        for frame_id, image_path in enumerate(images, start=1):
            face_result = face_worker.process_frame(image_path, frame_id)
            hybrid_result = hybrid_worker.process_frame(image_path, frame_id)

            face_tracks = [
                (
                    f"frame_{frame_id}_face_{index}",
                    face["face_bbox"],
                    face["student_id"],
                    face["score"],
                )
                for index, face in enumerate(face_result["faces"])
            ]

            matches = associate_v4(
                face_tracks,
                hybrid_result["people"],
            )

            result = {
                "frame_id": frame_id,
                "image": str(image_path.resolve()),
                "face_count": face_result["face_count"],
                "person_count": hybrid_result["person_count"],
                "matched_count": len(matches),
                "matches": matches,
                "mode": "IMAGE_ONLY",
            }

            output_path = output_dir / f"frame_{frame_id:04d}.json"
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(
                f"Frame {frame_id}: "
                f"faces={result['face_count']}, "
                f"people={result['person_count']}, "
                f"matched={result['matched_count']}"
            )
            print(f"Saved: {output_path}")

    finally:
        if face_worker is not None:
            face_worker.close()
        if hybrid_worker is not None:
            hybrid_worker.close()


if __name__ == "__main__":
    main()
