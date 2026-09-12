import cv2
import mediapipe as mp
import csv
import os

from ear_utils import get_eye_points, calculate_ear


VIDEO_PATH = "data/test_eye.mp4"
MODEL_PATH = "models/face_landmarker.task"

CSV_PATH = "data/ear_test.csv"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Không mở được video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

print("Calibration video opened successfully.")
print("FPS:", fps)
print("Press Q to quit.")


frame_index = 0
current_state = "UNKNOWN"
is_recording = False

csv_file = open(
    CSV_PATH,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "state",
    "left_ear",
    "right_ear",
    "avg_ear"
])

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Đã đọc hết video.")
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (frame_index / fps) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            height, width, _ = frame.shape

            left_eye_points = get_eye_points(
                face_landmarks,
                LEFT_EYE,
                width,
                height
            )

            right_eye_points = get_eye_points(
                face_landmarks,
                RIGHT_EYE,
                width,
                height
            )

            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            avg_ear = (
                left_ear + right_ear
            ) / 2.0

            if is_recording and current_state != "UNKNOWN":
                csv_writer.writerow([
                    timestamp_ms,
                    current_state,
                    left_ear,
                    right_ear,
                    avg_ear
                ])

            cv2.putText(
                frame,
                f"Left EAR: {left_ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right EAR: {right_ear:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Avg EAR: {avg_ear:.3f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:
            cv2.putText(
                frame,
                "NO FACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"State: {current_state}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        record_text = "RECORDING: ON" if is_recording else "RECORDING: OFF"

        cv2.putText(
            frame,
            record_text,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "EAR Calibration",
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("o"):
            current_state = "OPEN"
            print("State changed to OPEN")

        elif key == ord("c"):
            current_state = "CLOSED"
            print("State changed to CLOSED")
        
        elif key == ord("s"):
            is_recording = not is_recording

            if is_recording:
                print("Recording ON")
            else:
                print("Recording OFF")

        elif key == ord("q"):
            break

        frame_index += 1

csv_file.close()
cap.release()
cv2.destroyAllWindows()