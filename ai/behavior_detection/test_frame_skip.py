
import cv2

CAMERA_INDEX = 1
MAX_SIZE = 640
PROCESS_EVERY_N_FRAMES = 5

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Khong mo duoc camera iPhone!")
    raise SystemExit(1)

frame_count = 0
ai_count = 0

print("Camera da ket noi.")
print("Nhan Q de thoat.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame!")
        break

    # Resize frame
    height, width = frame.shape[:2]
    scale = min(MAX_SIZE / width, MAX_SIZE / height, 1.0)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    frame_count += 1

    # Chi xu ly AI moi 5 frame
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        ai_count += 1

        # Day la vi tri se goi model AI sau nay
        print(f"AI processing frame: {frame_count}")

    # Hien thi thong tin
    cv2.putText(
        resized_frame,
        f"Total frames: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        resized_frame,
        f"AI processed: {ai_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("HTTM - Frame Skipping", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tong frame:", frame_count)
print("So lan xu ly AI:", ai_count)
