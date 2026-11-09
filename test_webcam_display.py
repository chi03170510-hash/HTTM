
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Khong mo duoc webcam")
    raise SystemExit(1)

print("Webcam da mo thanh cong")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame")
        break

    cv2.imshow("HTTM - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
