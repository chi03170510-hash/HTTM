
import cv2

CAMERA_ID = 0

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Khong the mo webcam laptop!")
    exit()

print("Webcam laptop da ket noi!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame!")
        break

    cv2.imshow("HTTM - Laptop Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
