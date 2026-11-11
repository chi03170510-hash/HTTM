import argparse
from pathlib import Path

import cv2
import numpy as np

from enrollment_utils import (
    IMAGE_EXTENSIONS,
    extract_embedding,
    load_face_model,
    save_embeddings,
    validate_student_id,
)


def main():
    parser = argparse.ArgumentParser(
        description="HTTM CH07 - Face enrollment from images"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Example: SV001"
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing consented face images"
    )

    args = parser.parse_args()

    student_id = validate_student_id(args.student_id)
    image_dir = Path(args.images).resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No supported images found")

    print(f"Student ID: {student_id}")
    print(f"Images found: {len(image_paths)}")
    print("Loading InsightFace...")

    app = load_face_model()
    embeddings = []

    for image_path in image_paths:
        image_bytes = np.fromfile(
            str(image_path), dtype=np.uint8
        )
        image = cv2.imdecode(
            image_bytes, cv2.IMREAD_COLOR
        )

        try:
            embedding = extract_embedding(app, image)
            embeddings.append(embedding)
            print(f"[OK] {image_path.name}")

        except ValueError as exc:
            print(f"[SKIP] {image_path.name}: {exc}")

    if not embeddings:
        print("No valid face samples. Nothing saved.")
        return

    output_path = save_embeddings(
        student_id, embeddings
    )

    print()
    print("Enrollment completed")
    print(f"Student ID: {student_id}")
    print(f"Valid samples: {len(embeddings)}")
    print(f"Embedding shape: ({len(embeddings)}, 512)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
