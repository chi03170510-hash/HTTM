
import cv2

# Thu camera so 1 bang Windows Media Foundation
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Khong mo duoc Camera 1!")
    raise SystemExit(1)

print("Dang kiem tra Camera 1...")
print("Nhan Q de thoat.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc hinh anh!")
        break

    cv2.imshow("HTTM - Kiem tra Camera 1", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
