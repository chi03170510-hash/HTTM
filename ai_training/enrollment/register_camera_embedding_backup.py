import argparse
import time

import cv2

from enrollment_utils import (
    extract_embedding,
    load_face_model,
    validate_student_id,
)


def main():
    parser = argparse.ArgumentParser(
        description="HTTM CH-07.3 - Camera face embedding test"
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

    print("[OK] Camera opened")
    print("SPACE: Extract one embedding")
    print("Q: Quit")

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
            "SPACE: Capture | Q: Quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                print(
                    f"[OK] Embedding shape: {embedding.shape}, "
                    f"latency: {latency_ms:.1f} ms"
                )

            except ValueError as exc:
                print(f"[SKIP] {exc}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed")


if __name__ == "__main__":
    main()
