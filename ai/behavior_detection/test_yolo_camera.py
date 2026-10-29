
import cv2
import time
import threading
from pathlib import Path
from ultralytics import YOLO

# ========== CAU HINH ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PROCESS_INTERVAL = 0.15  # Khoang cach giua cac lan AI
IMG_SIZE = 416
CONFIDENCE = 0.20

WARNING_SECONDS = 5.0
MAX_MISSING_SECONDS = 0.8

# ========== CAMERA DOC FRAME MOI NHAT ==========
class CameraStream:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)

        if not self.cap.isOpened():
            raise RuntimeError("Khong mo duoc DroidCam!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        self.thread = threading.Thread(
            target=self.update, daemon=True
        )
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ========== LOAD MODEL ==========
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay: {MODEL_PATH}")

cv2.setNumThreads(2)

print("Dang load YOLO...")
model = YOLO(str(MODEL_PATH))
print("Model:", model.names)

camera = CameraStream(CAMERA_ID)

behavior_states = {
    "sleeping": {"start": None, "last_seen": None},
    "using_phone": {"start": None, "last_seen": None}
}

last_boxes = []
last_detection_count = 0
last_inference_time = 0
previous_time = time.perf_counter()
fps_smooth = 0.0

try:
    while True:
        frame = camera.read()

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(
            frame, (FRAME_WIDTH, FRAME_HEIGHT)
        )

        now = time.perf_counter()

        # ========== CHAY AI ==========
        if now - last_inference_time >= PROCESS_INTERVAL:
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]
            last_boxes = []

            detected_behaviors = set()

            for box in result.boxes:
                class_id = int(box.cls[0])
                behavior = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                last_boxes.append(
                    (x1, y1, x2, y2, behavior, confidence)
                )

                if behavior in behavior_states:
                    detected_behaviors.add(behavior)

            last_detection_count = len(last_boxes)

            # Thoi diem ket thuc inference
            detection_time = time.perf_counter()
            last_inference_time = detection_time

            for behavior, state in behavior_states.items():
                if behavior in detected_behaviors:
                    if state["start"] is None:
                        state["start"] = detection_time
                    state["last_seen"] = detection_time

                elif state["last_seen"] is not None:
                    if (detection_time - state["last_seen"]
                            > MAX_MISSING_SECONDS):
                        state["start"] = None
                        state["last_seen"] = None

        # ========== VE BOX ==========
        display_frame = frame.copy()

        for x1, y1, x2, y2, behavior, confidence in last_boxes:
            cv2.rectangle(
                display_frame,
                (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                display_frame,
                f"{behavior} {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        # ========== TIMER + WARNING ==========
        current_time = time.perf_counter()
        y_position = 110

        for behavior, state in behavior_states.items():
            if state["start"] is None:
                continue

            if (state["last_seen"] is None
                    or current_time - state["last_seen"]
                    > MAX_MISSING_SECONDS):
                continue

            duration = current_time - state["start"]

            if duration >= WARNING_SECONDS:
                status = "WARNING"
                color = (0, 0, 255)
            else:
                status = "DETECTING"
                color = (0, 255, 255)

            cv2.putText(
                display_frame,
                f"{behavior}: {duration:.1f}s - {status}",
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )
            y_position += 35

        # ========== FPS ==========
        delta = max(current_time - previous_time, 0.0001)
        instant_fps = 1.0 / delta
        fps_smooth = (
            0.9 * fps_smooth + 0.1 * instant_fps
            if fps_smooth > 0 else instant_fps
        )
        previous_time = current_time

        cv2.putText(
            display_frame,
            f"FPS: {fps_smooth:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            display_frame,
            f"Detections: {last_detection_count}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 255), 2
        )

        cv2.imshow(
            "HTTM - Optimized AI Camera",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.stop()
    cv2.destroyAllWindows()

print("Da dung chuong trinh!")
