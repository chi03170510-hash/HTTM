
import cv2

# Camera iPhone qua DroidCam
CAMERA_INDEX = 1
MAX_SIZE = 640

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Khong mo duoc camera iPhone!")
    raise SystemExit(1)

print("Camera da ket noi.")
print("Nhan Q de thoat.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame!")
        break

    # Lay kich thuoc frame goc
    height, width = frame.shape[:2]

    # Tinh ti le resize, khong lam meo hinh
    scale = min(MAX_SIZE / width, MAX_SIZE / height, 1.0)

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize frame
    resized_frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Hien thi thong tin
    cv2.putText(
        resized_frame,
        f"Size: {new_width}x{new_height}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("HTTM - Original Frame", frame)
    cv2.imshow("HTTM - Resized Frame", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Da dong camera.")
