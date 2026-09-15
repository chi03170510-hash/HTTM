import cv2
import time

URL = "http://192.168.1.3:4747/video"

print("=" * 60)
print(f"Test đọc HTTP stream: {URL}")
print("=" * 60)

# Thử với FFmpeg backend
print("\n[Thử 1] CAP_FFMPEG...")
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
print(f"  isOpened: {cap.isOpened()}")

if not cap.isOpened():
    print("\n[Thử 2] Backend mặc định...")
    cap = cv2.VideoCapture(URL)
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
    print("  1. DroidCam Client đã TẮT hẳn chưa? (kiểm tra system tray)")
    print("  2. Tab Chrome đã đóng chưa?")
    print("  3. URL http://192.168.1.3:4747/video có mở được trên Chrome không?")
    print("  4. Điện thoại và PC cùng WiFi?")