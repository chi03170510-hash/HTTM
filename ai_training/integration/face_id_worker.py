import argparse
import json
import sys
from pathlib import Path

import cv2

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR / "recognition"))

from recognize_camera_tracking import (
    load_database,
    load_face_model,
    recognize,
)


def process_image(image_path, frame_id, threshold):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    database = load_database()
    model = load_face_model()

    faces = []

    for face in model.get(image):
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

    return {
        "frame_id": frame_id,
        "image": str(image_path),
        "face_count": len(faces),
        "faces": faces,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    result = process_image(
        args.image,
        args.frame_id,
        args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Face ID worker: OK")
    print("Frame ID:", result["frame_id"])
    print("Faces:", result["face_count"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
