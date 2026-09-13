import cv2
import mediapipe as mp
import time

from ear_utils import get_eye_points, calculate_ear
from config import EAR_THRESHOLD


# =========================================================
# CẤU HÌNH
# =========================================================

SOURCE = "camera"   # "camera" hoặc "video"

CAMERA_INDEX = 0
VIDEO_PATH = "data/test_eye.mp4"
MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Nhắm mắt liên tục >= 5 giây thì cảnh báo
WARNING_TIME = 5.0


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,

    # Giữ nguyên code mới của Thiên
    num_faces=10
)


# =========================================================
# MỞ CAMERA / VIDEO
# =========================================================

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
print("EAR_THRESHOLD:", EAR_THRESHOLD)
print("Press Q to quit.")


# =========================================================
# BIẾN CHƯƠNG TRÌNH
# =========================================================

frame_index = 0
start_time = time.time()

eye_closed_start = {}


# =========================================================
# CHẠY FACE LANDMARKER
# =========================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:

            if SOURCE == "video":
                print("Đã đọc hết video.")
            else:
                print("Không đọc được frame từ camera.")

            break


        # =====================================================
        # OpenCV dùng BGR
        # MediaPipe dùng RGB
        # =====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # =====================================================
        # TIMESTAMP
        # =====================================================

        if SOURCE == "camera":

            timestamp_ms = int(
                (time.time() - start_time) * 1000
            )

        else:

            timestamp_ms = int(
                (frame_index / fps) * 1000
            )


        # =====================================================
        # PHÁT HIỆN LANDMARK
        # =====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # =====================================================
        # CÓ KHUÔN MẶT
        # =====================================================

        if result.face_landmarks:

            height, width, _ = frame.shape


            # Duyệt từng khuôn mặt
            for face_id, face_landmarks in enumerate(
                result.face_landmarks
            ):


                # =============================================
                # LẤY LANDMARK MẮT
                # =============================================

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


                # =============================================
                # TÍNH EAR
                # =============================================

                left_ear = calculate_ear(
                    left_eye_points
                )

                right_ear = calculate_ear(
                    right_eye_points
                )


                avg_ear = (
                    left_ear + right_ear
                ) / 2.0


                # =============================================
                # XÁC ĐỊNH OPEN / CLOSED
                # =============================================

                eye_closed = (
                    avg_ear < EAR_THRESHOLD
                )


                # =============================================
                # TIMER 5 GIÂY
                # =============================================

                if eye_closed:

                    eye_state = "CLOSED"


                    # Nếu khuôn mặt này vừa bắt đầu nhắm mắt
                    if face_id not in eye_closed_start:

                        eye_closed_start[face_id] = (
                            time.time()
                        )


                    # Tính số giây đã nhắm mắt
                    closed_duration = (
                        time.time()
                        - eye_closed_start[face_id]
                    )


                else:

                    eye_state = "OPEN"

                    # Mở mắt -> reset timer của khuôn mặt này
                    eye_closed_start.pop(
                        face_id,
                        None
                    )

                    closed_duration = 0.0


                # =============================================
                # VẼ LANDMARK MẮT TRÁI
                # =============================================

                for index in LEFT_EYE:

                    landmark = face_landmarks[index]

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )


                # =============================================
                # VẼ LANDMARK MẮT PHẢI
                # =============================================

                for index in RIGHT_EYE:

                    landmark = face_landmarks[index]

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 0, 255),
                        -1
                    )


                # =============================================
                # MÀU OPEN / CLOSED
                # =============================================

                if eye_closed:

                    state_color = (0, 0, 255)

                else:

                    state_color = (0, 255, 0)


                # =============================================
                # HIỂN THỊ EAR + OPEN/CLOSED
                # =============================================

                text_y = 40 + face_id * 100


                cv2.putText(
                    frame,
                    (
                        f"Face {face_id}: "
                        f"{eye_state} | "
                        f"EAR: {avg_ear:.3f}"
                    ),
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    state_color,
                    2
                )


                # =============================================
                # HIỂN THỊ CLOSED TIME
                # =============================================

                cv2.putText(
                    frame,
                    f"Closed Time: {closed_duration:.1f}s",
                    (20, text_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )


                # =============================================
                # CẢNH BÁO SAU 5 GIÂY
                # =============================================

                if closed_duration >= WARNING_TIME:


                    # -----------------------------------------
                    # Viền đỏ quanh camera
                    # -----------------------------------------

                    cv2.rectangle(
                        frame,
                        (5, 5),
                        (width - 5, height - 5),
                        (0, 0, 255),
                        12
                    )


                    # -----------------------------------------
                    # Khung đỏ ở giữa
                    # -----------------------------------------

                    warning_box_y1 = (
                        height // 2 - 80
                    )

                    warning_box_y2 = (
                        height // 2 + 80
                    )


                    cv2.rectangle(
                        frame,
                        (40, warning_box_y1),
                        (
                            width - 40,
                            warning_box_y2
                        ),
                        (0, 0, 180),
                        -1
                    )


                    # -----------------------------------------
                    # WARNING lớn
                    # -----------------------------------------

                    cv2.putText(
                        frame,
                        "WARNING!",
                        (
                            70,
                            height // 2
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (255, 255, 255),
                        5
                    )


                    # -----------------------------------------
                    # Nội dung cảnh báo
                    # -----------------------------------------

                    cv2.putText(
                        frame,
                        "EYES CLOSED >= 5 SECONDS",
                        (
                            70,
                            height // 2 + 50
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


        # =====================================================
        # KHÔNG PHÁT HIỆN KHUÔN MẶT
        # =====================================================

        else:

            # Không thấy mặt -> reset toàn bộ timer
            eye_closed_start.clear()


            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


        # =====================================================
        # HIỂN THỊ CAMERA
        # =====================================================

        cv2.imshow(
            "Driver Monitoring System",
            frame
        )


        # =====================================================
        # NHẤN Q ĐỂ THOÁT
        # =====================================================

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break


        frame_index += 1


# =========================================================
# KẾT THÚC
# =========================================================

cap.release()
cv2.destroyAllWindows()