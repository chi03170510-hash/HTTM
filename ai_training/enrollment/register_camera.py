import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from enrollment_utils import (
    DATA_DIR,
    extract_embedding,
    load_face_model,
    save_embeddings,
    validate_student_id,
)


def save_camera_samples(student_id, samples):
    if not samples:
        print("[SKIP] No samples collected")
        return False

    output_path = DATA_DIR / student_id / "embeddings.npz"

    if not output_path.exists():
        try:
            saved_path = save_embeddings(student_id, samples)
            print(f"[SAVED] {len(samples)} samples")
            print(f"Path: {saved_path}")
            return True
        except FileExistsError:
            print("[ERROR] Enrollment already exists")
            return False

    # Existing enrollment: read and validate old data.
    try:
        with np.load(output_path, allow_pickle=False) as data:
            existing_id = str(data["student_id"])
            existing = data["embeddings"].copy()
    except Exception as exc:
        print(f"[ERROR] Cannot read existing enrollment: {exc}")
        return False

    if existing_id != student_id:
        print("[ERROR] Student ID mismatch")
        return False

    if (
        existing.ndim != 2
        or existing.shape[1] != 512
        or not np.all(np.isfinite(existing))
    ):
        print("[ERROR] Invalid existing embeddings")
        return False

    new_samples = np.asarray(samples, dtype=np.float32)
    combined = np.vstack([existing, new_samples])

    # Create a backup before replacing the old file.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.with_name(
        f"embeddings_backup_{timestamp}.npz"
    )

    try:
        with open(output_path, "rb") as source:
            with open(backup_path, "xb") as backup:
                backup.write(source.read())

        # Write to a temporary file, then replace atomically.
        temp_path = output_path.with_name(
            "embeddings_pending.npz"
        )

        with open(temp_path, "xb") as temp_file:
            np.savez_compressed(
                temp_file,
                student_id=np.array(student_id),
                embeddings=combined,
            )

        os.replace(temp_path, output_path)

    except Exception as exc:
        print(f"[ERROR] Cannot update enrollment: {exc}")
        return False

    print(f"[UPDATED] Previous samples: {len(existing)}")
    print(f"[UPDATED] Added samples: {len(new_samples)}")
    print(f"[UPDATED] Total samples: {len(combined)}")
    print(f"Backup: {backup_path}")
    print(f"Path: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="HTTM CH-07.3 - Camera enrollment"
    )
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    student_id = validate_student_id(args.student_id)

    print(f"Student ID: {student_id}")
    print("Loading InsightFace...")

    app = load_face_model()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    samples = []

    print("[OK] Camera opened")
    print("SPACE: Capture embedding")
    print("S: Save enrollment")
    print("Q: Quit")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Cannot read camera frame")
                break

            display = frame.copy()

            cv2.putText(
                display,
                f"Student: {student_id}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                f"Samples: {len(samples)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "SPACE: Capture | S: Save | Q: Quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("HTTM - Face Enrollment", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                start = time.perf_counter()

                try:
                    embedding = extract_embedding(app, frame)
                    samples.append(embedding)

                    latency_ms = (
                        time.perf_counter() - start
                    ) * 1000

                    print(
                        f"[CAPTURED] Sample {len(samples)} "
                        f"| shape={embedding.shape} "
                        f"| {latency_ms:.1f} ms"
                    )

                except ValueError as exc:
                    print(f"[SKIP] {exc}")

            if key == ord("s"):
                if save_camera_samples(student_id, samples):
                    samples.clear()
                    print("[OK] Samples saved. Press Q to quit.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed")


if __name__ == "__main__":
    main()
