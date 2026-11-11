import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="HTTM CH-07.3 - Camera enrollment"
    )
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    print(f"Student ID: {args.student_id}")
    print(f"Camera index: {args.camera}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[OK] Camera opened")
    print("Press SPACE to test capture")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Cannot read camera frame")
            break

        cv2.putText(
            frame,
            f"Student: {args.student_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("HTTM - Face Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            print("[TEST] SPACE pressed - camera capture OK")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed")


if __name__ == "__main__":
    main()
