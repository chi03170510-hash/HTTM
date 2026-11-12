import json
import sys
from pathlib import Path

import cv2
from bbox_visual_metadata import enabled as visual_enabled, write_source
from face_inference_device import load_live_face_model, face_runtime

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR / "recognition"))

from recognize_camera_tracking import (
    load_database,
    load_face_model,
    recognize,
)


def main():
    # Keep stdout reserved for JSON responses.
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        visualization_enabled = visual_enabled()
        print("Loading Face ID database and model...", file=sys.stderr)
        database = load_database()
        model = load_live_face_model(load_face_model)
        runtime = face_runtime(model)
        print("Face ID model loaded once.", file=sys.stderr)

        for line in sys.stdin:
            if not line.strip():
                continue

            frame_id = None

            try:
                request = json.loads(line)
                frame_id = request["frame_id"]
                image_path = Path(request["image"])
                threshold = float(request.get("threshold", 0.45))

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                frame = cv2.imread(str(image_path))

                if frame is None:
                    raise RuntimeError("Cannot read image")

                faces = []

                for face in model.get(frame):
                    bbox = [int(v) for v in face.bbox[:4]]

                    student_id, score = recognize(
                        face.embedding,
                        database,
                        threshold,
                    )

                    faces.append({
                        "track_id": None,
                        "face_bbox": bbox,
                        "student_id": student_id,
                        "score": float(score),
                    })

                if visualization_enabled:
                    write_source(image_path, frame_id, frame.shape, "face", faces=faces)

                result = {
                    "frame_id": frame_id,
                    "image": str(image_path.resolve()),
                    "face_count": len(faces),
                    "inference_runtime": runtime,
                    "faces": faces,
                }

            except Exception as exc:
                result = {
                    "frame_id": frame_id,
                    "error": str(exc),
                }

            protocol_stdout.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            protocol_stdout.flush()

    finally:
        sys.stdout = protocol_stdout


if __name__ == "__main__":
    main()
