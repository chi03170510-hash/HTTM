
import cv2
from pathlib import Path

# ==========================================
# CAU HINH
# ==========================================

CAMERA_INDEX = 1
MAX_SIZE = 640
PROCESS_EVERY_N_FRAMES = 5

# Duong dan file nhan dien khuon mat
BASE_DIR = Path(__file__).resolve().parent

FACE_MODEL_PATH = (
    BASE_DIR / "haarcascade_frontalface_default.xml"
)

# ==========================================
# KHOI TAO FACE DETECTOR
# ==========================================

if not FACE_MODEL_PATH.is_file():
    print("LOI: Khong tim thay file XML!")
    print("Duong dan:", FACE_MODEL_PATH)
    raise SystemExit(1)

face_cascade = cv2.CascadeClassifier(
    str(FACE_MODEL_PATH)
)

if face_cascade.empty():
    print("LOI: Khong tai duoc face detector!")
    raise SystemExit(1)

print("Da tai face detector thanh cong!")

# ==========================================
# KET NOI CAMERA IPHONE
# ==========================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_MSMF
)

if not cap.isOpened():
    print("LOI: Khong mo duoc camera iPhone!")
    raise SystemExit(1)

print("Da ket noi camera iPhone!")
print("Nhan Q de thoat.")

# ==========================================
# BIEN DEM
# ==========================================

frame_count = 0
ai_count = 0
faces = []

# ==========================================
# VONG LAP CAMERA
# ==========================================

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Khong doc duoc frame!")
            break

        # ----------------------------------
        # RESIZE FRAME
        # ----------------------------------

        height, width = frame.shape[:2]

        scale = min(
            MAX_SIZE / width,
            MAX_SIZE / height,
            1.0
        )

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        frame_count += 1

        # ----------------------------------
        # PHAT HIEN KHUON MAT MOI 5 FRAME
        # ----------------------------------

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:

            ai_count += 1

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

        # ----------------------------------
        # VE BOUNDING BOX
        # ----------------------------------

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ----------------------------------
        # HIEN THI THONG TIN
        # ----------------------------------

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Total frames: {frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"AI processed: {ai_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "HTTM - Face Detection CH-05",
            frame
        )

        # Nhan Q de thoat
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Da dong camera.")
    print("Tong frames:", frame_count)
    print("So lan xu ly AI:", ai_count)
