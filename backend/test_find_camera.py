import cv2
import time

CAMERA_INDEX = 0

print("=" * 60)
print(f"Test webcam vật lý, camera index: {CAMERA_INDEX}")
print("=" * 60)

print("\n[Thử] Camera index 0...")
cap = cv2.VideoCapture(CAMERA_INDEX)
print(f"  isOpened: {cap.isOpened()}")

if cap.isOpened():
    print("\n✅ Mở thành công! Đọc thử 10 frame...")
    success_count = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
            print(f"  Frame {i+1}: OK - Shape: {frame.shape}")
        else:
            print(f"  Frame {i+1}: FAIL")
        time.sleep(0.05)
    cap.release()
    print(f"\n🎉 Đọc được {success_count}/10 frame!")
else:
    print("\n❌ Không mở được stream!")
    print("\nChecklist:")
    print("  1. Webcam vật lý đã được kết nối chưa?")
    print("  2. Ứng dụng khác có đang chiếm camera không?")
    print("  3. Windows đã cấp quyền camera cho Python chưa?")