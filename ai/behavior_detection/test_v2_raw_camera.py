
import cv2
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "experimental" / "three_behavior_v2_best.pt"

model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH)
print("Classes:", model.names)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Khong mo duoc webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Nhan Q de thoat")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.10,
        iou=0.70,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("HTTM - V2 RAW CAMERA TEST", annotated)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        behavior = model.names[class_id]

        print(f"{behavior}: {confidence:.3f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
