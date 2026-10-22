
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. CAU HINH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_behavior_v1_4.pt"

# DroidCam iPhone
CAMERA_ID = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PROCESS_EVERY_N_FRAMES = 3
CONFIDENCE = 0.35

# Canh bao khi hanh vi > 5 giay
WARNING_SECONDS = 5.0

# Cho phep mat detection ngan (giay)
# Neu mat lau hon thi reset timer
MAX_MISSING_SECONDS = 0.5

# ==========================================
# 2. LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}"
    )

print("Dang load model YOLO...")

model = YOLO(str(MODEL_PATH))

print("Load model thanh cong!")
print("Danh sach hanh vi:", model.names)

# ==========================================
# 3. KET NOI CAMERA
# ==========================================

print("Dang ket noi DroidCam...")

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_MSMF)

if not cap.isOpened():
    raise RuntimeError(
        "Khong ket noi duoc DroidCam Camera 1!"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Camera da san sang!")
print("Nhan Q de thoat.")

# ==========================================
# 4. TRANG THAI HANH VI
# ==========================================

# Timer theo tung loai hanh vi
# Chi phu hop demo mot nguoi
behavior_states = {
    "sleeping": {
        "start": None,
        "last_seen": None
    },
    "using_phone": {
        "start": None,
        "last_seen": None
    }
}

frame_count = 0
last_results = None
last_detection_count = 0

previous_time = time.perf_counter()

# ==========================================
# 5. VONG LAP CAMERA
# ==========================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Khong doc duoc frame!")
            break

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        frame_count += 1
        now = time.perf_counter()

        # ==================================
        # 6. CHAY YOLO MOI N FRAME
        # ==================================

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=640,
                verbose=False
            )

            last_results = results[0]
            last_detection_count = len(last_results.boxes)

            detected_behaviors = set()

            for box in last_results.boxes:

                class_id = int(box.cls[0])
                behavior = model.names[class_id]

                if behavior in behavior_states:
                    detected_behaviors.add(behavior)

            # Cap nhat timer
            for behavior, state in behavior_states.items():

                if behavior in detected_behaviors:

                    if state["start"] is None:
                        state["start"] = now

                    state["last_seen"] = now

                else:
                    if state["last_seen"] is not None:
                        missing_time = (
                            now - state["last_seen"]
                        )

                        if missing_time > MAX_MISSING_SECONDS:
                            state["start"] = None
                            state["last_seen"] = None

        # ==================================
        # 7. VE BOUNDING BOX
        # ==================================

        if last_results is not None:
            display_frame = last_results.plot(
                img=frame.copy()
            )
        else:
            display_frame = frame.copy()

        # ==================================
        # 8. HIEN THI TIMER + WARNING
        # ==================================

        y_position = 110

        for behavior, state in behavior_states.items():

            if state["start"] is None:
                continue

            if (
                state["last_seen"] is None
                or now - state["last_seen"] > MAX_MISSING_SECONDS
            ):
                continue

            duration = now - state["start"]

            if duration > WARNING_SECONDS:

                color = (0, 0, 255)
                status = "WARNING"

            else:

                color = (0, 255, 255)
                status = "DETECTING"

            label = (
                f"{behavior}: {duration:.1f}s "
                f"- {status}"
            )

            cv2.putText(
                display_frame,
                label,
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            y_position += 35

        # ==================================
        # 9. FPS
        # ==================================

        fps = 1.0 / max(
            now - previous_time,
            0.0001
        )

        previous_time = now

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Detections: {last_detection_count}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # ==================================
        # 10. HIEN THI CAMERA
        # ==================================

        cv2.imshow(
            "HTTM - AI Behavior Warning Demo",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Da dung chuong trinh!")
