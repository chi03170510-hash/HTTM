import cv2
import time

CAMERA_INDEX = 2
TARGET_SIZE = 640

def resize_with_padding(frame, target_size=640):
    h, w = frame.shape[:2]

    scale = min(target_size / w, target_size / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    return cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open iVCam")
        return

    print("HTTM Camera Preview")
    print(f"Camera index: {CAMERA_INDEX}")
    print("Press Q to exit")

    previous_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Cannot read frame")
                break

            processed = resize_with_padding(frame, TARGET_SIZE)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            cv2.putText(
                processed,
                f"FPS: {fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("HTTM - Camera Preview", processed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
