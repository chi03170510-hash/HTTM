
import cv2
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ==========================================
# 1. CAU HINH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

IMG_SIZE = 416
CONFIDENCE = 0.35
PROCESS_INTERVAL = 0.15

WARNING_SECONDS = 5.0
MAX_MISSING_SECONDS = 0.8
MIN_CONFIRM_FRAMES = 3

LOG_DIR = BASE_DIR / "logs"
WARNING_DIR = LOG_DIR / "warnings"
LOG_PATH = LOG_DIR / "detection_log.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)
WARNING_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIORS = ("sleeping", "using_phone")


# ==========================================
# 2. GHI LOG CSV
# ==========================================

log_lock = threading.Lock()

if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
    with open(
        LOG_PATH, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "event",
            "behavior",
            "confidence",
            "duration_seconds",
            "x1", "y1", "x2", "y2",
            "image_path"
        ])


def write_log(
    event,
    behavior,
    confidence=0.0,
    duration=0.0,
    box=None,
    image_path=""
):
    if box is None:
        box = ("", "", "", "")

    with log_lock:
        with open(
            LOG_PATH, "a",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                event,
                behavior,
                round(confidence, 3),
                round(duration, 2),
                *box,
                image_path
            ])


# ==========================================
# 3. CAMERA THREAD
# ==========================================

class CameraStream:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(
            camera_id, cv2.CAP_MSMF
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Khong mo duoc DroidCam!"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id

            return self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ==========================================
# 4. TRANG THAI CHUNG
# ==========================================

shared_lock = threading.Lock()
stop_event = threading.Event()

shared = {
    "boxes": [],
    "detections": 0,
    "ai_fps": 0.0,
    "inference_ms": 0.0,
    "states": {
        behavior: {
            "start": None,
            "last_seen": None,
            "hits": 0,
            "first_seen": None,
            "confirmed": False,
            "warning_sent": False,
            "confidence": 0.0
        }
        for behavior in BEHAVIORS
    }
}


def reset_state(state):
    state["start"] = None
    state["last_seen"] = None
    state["hits"] = 0
    state["first_seen"] = None
    state["confirmed"] = False
    state["warning_sent"] = False
    state["confidence"] = 0.0


# ==========================================
# 5. LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}"
    )

cv2.setNumThreads(2)

print("Dang load YOLO...")
print("Model path:", MODEL_PATH.resolve())

model = YOLO(str(MODEL_PATH))

print("Load model thanh cong!")
print("Danh sach hanh vi:", model.names)


# ==========================================
# 6. KET NOI CAMERA
# ==========================================

print("Dang ket noi DroidCam...")
camera = CameraStream(CAMERA_ID)

print("Camera da san sang!")
print("Nhan Q de thoat.")


# ==========================================
# 7. AI WORKER
# ==========================================

def ai_worker():
    last_processed_id = -1
    last_inference_end = 0.0

    while not stop_event.is_set():

        frame, frame_id = camera.read()

        if frame is None:
            time.sleep(0.01)
            continue

        if frame_id == last_processed_id:
            time.sleep(0.005)
            continue

        now = time.perf_counter()

        if now - last_inference_end < PROCESS_INTERVAL:
            time.sleep(0.01)
            continue

        last_processed_id = frame_id

        inference_start = time.perf_counter()

        try:
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE,
                verbose=False
            )
        except Exception as error:
            print("[AI ERROR]", error)
            stop_event.set()
            break

        inference_end = time.perf_counter()
        last_inference_end = inference_end

        inference_ms = (
            inference_end - inference_start
        ) * 1000.0

        result = results[0]

        boxes = []
        detected = {}

        for box in result.boxes:

            class_id = int(box.cls[0])
            behavior = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            coordinates = (x1, y1, x2, y2)

            boxes.append((
                x1, y1, x2, y2,
                behavior, confidence
            ))

            if behavior in BEHAVIORS:
                if (
                    behavior not in detected
                    or confidence > detected[behavior][0]
                ):
                    detected[behavior] = (
                        confidence, coordinates
                    )

            write_log(
                "DETECTION",
                behavior,
                confidence,
                box=coordinates
            )

        # Cap nhat timer theo tung loai hanh vi
        # Phu hop demo mot nguoi
        with shared_lock:

            shared["boxes"] = boxes
            shared["detections"] = len(boxes)
            shared["inference_ms"] = inference_ms
            shared["ai_fps"] = (
                1000.0 / max(inference_ms, 1.0)
            )

            for behavior, state in shared["states"].items():

                if behavior in detected:

                    confidence, coordinates = detected[behavior]

                    if state["last_seen"] is not None:
                        gap = (
                            inference_end - state["last_seen"]
                        )

                        if gap > MAX_MISSING_SECONDS:
                            reset_state(state)

                    if state["first_seen"] is None:
                        state["first_seen"] = inference_end

                    state["hits"] += 1
                    state["last_seen"] = inference_end
                    state["confidence"] = confidence

                    if (
                        not state["confirmed"]
                        and state["hits"] >= MIN_CONFIRM_FRAMES
                    ):
                        state["confirmed"] = True
                        state["start"] = state["first_seen"]

                        write_log(
                            "CONFIRMED",
                            behavior,
                            confidence,
                            box=coordinates
                        )

                else:
                    if state["last_seen"] is not None:

                        missing_time = (
                            inference_end - state["last_seen"]
                        )

                        if missing_time > MAX_MISSING_SECONDS:

                            if state["confirmed"]:
                                duration = (
                                    state["last_seen"]
                                    - state["start"]
                                )

                                write_log(
                                    "ENDED",
                                    behavior,
                                    state["confidence"],
                                    duration
                                )

                            reset_state(state)


worker = threading.Thread(
    target=ai_worker,
    daemon=True
)

worker.start()


# ==========================================
# 8. DISPLAY LOOP
# ==========================================

previous_time = time.perf_counter()
display_fps = 0.0

try:
    while not stop_event.is_set():

        frame, frame_id = camera.read()

        if frame is None:
            time.sleep(0.01)
            continue

        now = time.perf_counter()
        display_frame = frame.copy()

        # Lay snapshot trang thai
        with shared_lock:

            boxes = list(shared["boxes"])
            detection_count = shared["detections"]
            ai_fps = shared["ai_fps"]
            inference_ms = shared["inference_ms"]

            states_snapshot = {
                behavior: dict(state)
                for behavior, state
                in shared["states"].items()
            }

        # ==================================
        # 9. VE BOUNDING BOX
        # ==================================

        for (
            x1, y1, x2, y2,
            behavior, confidence
        ) in boxes:

            state = states_snapshot.get(behavior)

            if (
                state is not None
                and state["confirmed"]
                and state["start"] is not None
                and now - state["last_seen"]
                <= MAX_MISSING_SECONDS
                and now - state["start"]
                >= WARNING_SECONDS
            ):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f"{behavior} {confidence:.2f}"
            )

            cv2.putText(
                display_frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # ==================================
        # 10. TIMER + WARNING
        # ==================================

        y_position = 125

        for behavior, state in states_snapshot.items():

            if not state["confirmed"]:
                continue

            if state["start"] is None:
                continue

            if state["last_seen"] is None:
                continue

            if (
                now - state["last_seen"]
                > MAX_MISSING_SECONDS
            ):
                continue

            duration = now - state["start"]

            if duration >= WARNING_SECONDS:
                status = "WARNING"
                color = (0, 0, 255)

                # Chi luu mot canh bao moi dot hanh vi
                should_save = False

                with shared_lock:
                    live_state = shared["states"][behavior]

                    if (
                        live_state["confirmed"]
                        and live_state["start"]
                        == state["start"]
                        and not live_state["warning_sent"]
                    ):
                        live_state["warning_sent"] = True
                        should_save = True

                if should_save:

                    filename = (
                        f"{behavior}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )

                    image_path = WARNING_DIR / filename

                    saved = cv2.imwrite(
                        str(image_path),
                        display_frame
                    )

                    saved_path = (
                        str(image_path)
                        if saved else ""
                    )

                    write_log(
                        "WARNING",
                        behavior,
                        state["confidence"],
                        duration,
                        image_path=saved_path
                    )

                    print(
                        f"[WARNING] {behavior}: "
                        f"{duration:.1f}s"
                    )

            else:
                status = "DETECTING"
                color = (0, 255, 255)

            cv2.putText(
                display_frame,
                (
                    f"{behavior}: "
                    f"{duration:.1f}s - {status}"
                ),
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            y_position += 35

        # ==================================
        # 11. FPS
        # ==================================

        delta = max(
            now - previous_time,
            0.0001
        )

        instant_fps = 1.0 / delta

        if display_fps == 0:
            display_fps = instant_fps
        else:
            display_fps = (
                0.9 * display_fps
                + 0.1 * instant_fps
            )

        previous_time = now

        # Nen den de de doc thong tin
        cv2.rectangle(
            display_frame,
            (0, 0),
            (350, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display_frame,
            f"Display FPS: {display_fps:.1f}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            (
                f"AI FPS: {ai_fps:.1f} "
                f"({inference_ms:.0f} ms)"
            ),
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Detections: {detection_count}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "HTTM - Optimized AI Camera",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_event.set()
    worker.join(timeout=3)
    camera.stop()
    cv2.destroyAllWindows()

print("Da dung chuong trinh!")
print("Log:", LOG_PATH)
print("Anh canh bao:", WARNING_DIR)
