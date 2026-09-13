import cv2
import mediapipe as mp
import time

from ear_utils import get_eye_points, calculate_ear
from config import EAR_THRESHOLD

SOURCE = "camera"   # "camera" hoặc "video"

CAMERA_INDEX = 0
VIDEO_PATH = "data/test_eye.mp4"
MODEL_PATH = "models/face_landmarker.task"

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
    num_faces=10
)

if SOURCE == "camera":
    cap = cv2.VideoCapture(CAMERA_INDEX)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Không mở được nguồn camera/video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

print("Source opened successfully.")
print("Source:", SOURCE)

if SOURCE == "video":
    print("FPS:", fps)

print("Press Q to quit.")


frame_index = 0
start_time = time.time()


with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            if SOURCE == "video":
                print("Đã đọc hết video.")
            else:
                print("Không đọc được frame từ camera.")
            break

        # OpenCV sử dụng BGR
        # MediaPipe cần RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Chuyển NumPy array thành MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp của frame, đơn vị millisecond
        if SOURCE == "camera":
            timestamp_ms = int((time.time() - start_time) * 1000)
        else:
            timestamp_ms = int((frame_index / fps) * 1000)

        # Phát hiện landmark
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:

            height, width, _ = frame.shape

            for face_id, face_landmarks in enumerate(result.face_landmarks):

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

                avg_ear = (left_ear + right_ear) / 2.0

                eye_closed = avg_ear < EAR_THRESHOLD

                if eye_closed:
                    eye_state = "CLOSED"
                else:
                    eye_state = "OPEN"

                # Vẽ landmark mắt trái
                for index in LEFT_EYE:
                    landmark = face_landmarks[index]

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                # Vẽ landmark mắt phải
                for index in RIGHT_EYE:
                    landmark = face_landmarks[index]

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 0, 255),
                        -1
                    )

                # Hiển thị trạng thái từng khuôn mặt
                text_y = 40 + face_id * 40

                cv2.putText(
                    frame,
                    f"Face {face_id}: {eye_state} | EAR: {avg_ear:.3f}",
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            print("No face detected.")

        cv2.imshow("Face Landmarker Test", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        frame_index += 1


cap.release()
cv2.destroyAllWindows()