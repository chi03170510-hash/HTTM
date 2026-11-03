
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========== CONFIG ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = "yolo11n-pose.pt"

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

IMG_SIZE = 320
CONFIDENCE = 0.4
KEYPOINT_CONF = 0.45
PROCESS_INTERVAL = 0.15

# COCO keypoints:
# 5 left shoulder, 6 right shoulder
# 7 left elbow, 8 right elbow
# 9 left wrist, 10 right wrist
# 11 left hip, 12 right hip
# 13 left knee, 14 right knee

def point_valid(kpts, idx):
    return kpts[idx, 2] >= KEYPOINT_CONF


def hand_raised(kpts, shoulder_idx, elbow_idx, wrist_idx):
    if not all(point_valid(kpts, i) for i in
               [shoulder_idx, elbow_idx, wrist_idx]):
        return False

    shoulder_y = kpts[shoulder_idx, 1]
    elbow_y = kpts[elbow_idx, 1]
    wrist_y = kpts[wrist_idx, 1]

    # OpenCV: smaller Y means higher position
    return wrist_y < shoulder_y - 20 and wrist_y < elbow_y


def classify_posture(kpts):
    required = [11, 12, 13, 14]

    if not all(point_valid(kpts, i) for i in required):
        return "UNKNOWN"

    hip_y = (kpts[11, 1] + kpts[12, 1]) / 2
    knee_y = (kpts[13, 1] + kpts[14, 1]) / 2

    shoulder_valid = point_valid(kpts, 5) and point_valid(kpts, 6)

    if not shoulder_valid:
        return "UNKNOWN"

    shoulder_y = (kpts[5, 1] + kpts[6, 1]) / 2
    torso_height = abs(hip_y - shoulder_y)

    if torso_height < 10:
        return "UNKNOWN"

    hip_knee_distance = abs(knee_y - hip_y)
    ratio = hip_knee_distance / torso_height

    if ratio < 0.75:
        return "SITTING (EST.)"

    if ratio > 1.15:
        return "STANDING (EST.)"

    return "UNKNOWN"


def main():
    print("[INFO] Loading YOLO11n-Pose...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    last_inference = 0.0
    last_result = None
    fps = 0.0
    previous_time = time.perf_counter()

    print("[INFO] Webcam started. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame.")
                break

            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now

            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.85 * fps + 0.15 * instant_fps

            if now - last_inference >= PROCESS_INTERVAL:
                results = model.predict(
                    frame,
                    imgsz=IMG_SIZE,
                    conf=CONFIDENCE,
                    device="cpu",
                    verbose=False
                )
                last_result = results[0]
                last_inference = time.perf_counter()

            person_count = 0

            if last_result is not None:
                # Draw keypoints and skeleton
                annotated = last_result.plot()
                frame = annotated

                if last_result.keypoints is not None:
                    xy = last_result.keypoints.xy.cpu().numpy()

                    if last_result.keypoints.conf is not None:
                        kp_conf = last_result.keypoints.conf.cpu().numpy()
                    else:
                        kp_conf = np.ones(xy.shape[:2], dtype=np.float32)

                    person_count = len(xy)

                    for i in range(person_count):
                        kpts = np.column_stack((xy[i], kp_conf[i]))

                        left = hand_raised(kpts, 5, 7, 9)
                        right = hand_raised(kpts, 6, 8, 10)
                        posture = classify_posture(kpts)

                        if left and right:
                            action = "BOTH HANDS RAISED"
                        elif left:
                            action = "LEFT HAND RAISED"
                        elif right:
                            action = "RIGHT HAND RAISED"
                        else:
                            action = "NO HAND RAISED"

                        if last_result.boxes is not None:
                            box = last_result.boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = map(int, box)

                            cv2.putText(
                                frame,
                                action,
                                (x1, max(25, y1 - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                frame,
                                posture,
                                (x1, max(45, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                2
                            )

            cv2.rectangle(frame, (0, 0), (310, 65), (30, 30, 30), -1)

            cv2.putText(
                frame,
                f"Display FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("HTTM - YOLO11n Pose Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released.")


if __name__ == "__main__":
    main()
