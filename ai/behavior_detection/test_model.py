
from pathlib import Path
from ultralytics import YOLO

# Lay duong dan model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_behavior_v1_4.pt"

# Kiem tra file
if not MODEL_PATH.exists():
    print("Khong tim thay model:", MODEL_PATH)
    exit()

# Load model
model = YOLO(str(MODEL_PATH))

print("LOAD MODEL THANH CONG!")
print("Danh sach hanh vi:", model.names)
print("Loai model:", model.task)
