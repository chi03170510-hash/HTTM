import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR / "enrollment"))

from enrollment_utils import DATA_DIR, load_face_model, extract_embedding


def load_database():
    database = {}

    for path in sorted(DATA_DIR.glob("SV*/embeddings.npz")):
        with np.load(path, allow_pickle=False) as data:
            student_id = str(data["student_id"])
            embeddings = data["embeddings"].astype(np.float32)

        if student_id != path.parent.name:
            print(f"[SKIP] ID mismatch: {path}")
            continue

        if (
            embeddings.ndim != 2
            or embeddings.shape[0] == 0
            or embeddings.shape[1] != 512
            or not np.all(np.isfinite(embeddings))
        ):
            print(f"[SKIP] Invalid embeddings: {path}")
            continue

        norms = np.linalg.norm(embeddings, axis=1)

        if np.any(norms <= 1e-8):
            print(f"[SKIP] Zero-norm embeddings: {path}")
            continue

        database[student_id] = embeddings / norms[:, None]

    return database


def recognize(embedding, database, threshold):
    scores = {}

    for student_id, references in database.items():
        similarities = references @ embedding
        scores[student_id] = float(np.max(similarities))

    if not scores:
        return "Unknown", 0.0, scores

    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]

    if best_score < threshold:
        return "Unknown", best_score, scores

    return best_id, best_score, scores


def main():
    parser = argparse.ArgumentParser()
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

    app = load_face_model()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("SPACE: Recognize | Q: Quit")

    result_text = "Press SPACE to recognize"

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            display = frame.copy()

            cv2.putText(
                display,
                result_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("HTTM - Face Recognition", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                try:
                    embedding = extract_embedding(app, frame)
                    student_id, score, scores = recognize(
                        embedding, database, args.threshold
                    )

                    result_text = f"{student_id} | score={score:.3f}"

                    print(f"\nResult: {result_text}")
                    for sid, similarity in sorted(scores.items()):
                        print(f"  {sid}: {similarity:.4f}")

                except ValueError as exc:
                    result_text = "No valid single face"
                    print(f"[SKIP] {exc}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
