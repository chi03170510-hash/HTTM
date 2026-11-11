import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR / "enrollment"))

from enrollment_utils import DATA_DIR, load_face_model


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

            if embeddings.ndim != 2 or embeddings.shape[1] != 512:
                raise ValueError("Invalid embedding shape")

            if len(embeddings) == 0:
                raise ValueError("Empty enrollment")

            if not np.all(np.isfinite(embeddings)):
                raise ValueError("Non-finite embedding")

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
        return "Unknown", 0.0, {}

    norm = float(np.linalg.norm(embedding))

    if norm <= 1e-8:
        return "Unknown", 0.0, {}

    embedding = embedding / norm

    scores = {
        student_id: float(np.max(references @ embedding))
        for student_id, references in database.items()
    }

    if not scores:
        return "Unknown", 0.0, scores

    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]

    if best_score < threshold:
        return "Unknown", best_score, scores

    return best_id, best_score, scores


def main():
    parser = argparse.ArgumentParser(
        description="HTTM CH-08.2 - Multi-face recognition"
    )
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    if not -1.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between -1 and 1")

    database = load_database()

    if not database:
        print("[ERROR] No valid enrollments found")
        return

    print("Loaded students:")
    for student_id, embeddings in database.items():
        print(f"  {student_id}: {len(embeddings)} samples")

    print("Loading InsightFace...")
    app = load_face_model()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[OK] Camera opened")
    print("SPACE: Recognize all visible faces")
    print("Q: Quit")

    results = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Cannot read frame")
                break

            display = frame.copy()

            for bbox, student_id, score in results:
                x1, y1, x2, y2 = bbox

                color = (
                    (0, 255, 0)
                    if student_id != "Unknown"
                    else (0, 165, 255)
                )

                cv2.rectangle(
                    display, (x1, y1), (x2, y2), color, 2
                )

                label = f"{student_id} ({score:.3f})"

                cv2.putText(
                    display,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.putText(
                display,
                "SPACE: Recognize | Q: Quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("HTTM - Multi Face Recognition", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                start = time.perf_counter()

                faces = app.get(frame)
                results = []

                print(f"\nDetected faces: {len(faces)}")

                for index, face in enumerate(faces, start=1):
                    bbox = tuple(
                        int(value) for value in face.bbox[:4]
                    )

                    embedding = face.embedding

                    student_id, score, scores = recognize(
                        embedding, database, args.threshold
                    )

                    results.append((bbox, student_id, score))

                    print(
                        f"Face {index}: {student_id} "
                        f"| score={score:.4f}"
                    )

                    for sid, similarity in sorted(scores.items()):
                        print(f"  {sid}: {similarity:.4f}")

                elapsed = (time.perf_counter() - start) * 1000
                print(f"Inference latency: {elapsed:.1f} ms")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed")


if __name__ == "__main__":
    main()
