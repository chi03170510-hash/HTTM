from pathlib import Path
from ultralytics import YOLO
import cv2
import time


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
INPUT_DIR = BASE_DIR / "test_videos"
OUTPUT_DIR = BASE_DIR / "video_results"


# ============================================================
# CONFIG
# ============================================================

CONF_THRESHOLD = 0.25
IMG_SIZE = 640

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TEST YOLO - VIDEO STUDENT BEHAVIOR")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        print("[ERROR] Không tìm thấy model:")
        print(MODEL_PATH)
        return

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------

    video_files = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print("[ERROR] Không tìm thấy video trong:")
        print(INPUT_DIR)
        return

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading model:")
    print(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    print()
    print("Classes:")
    print(model.names)

    print()
    print(f"Found {len(video_files)} video(s)")

    # --------------------------------------------------------
    # PROCESS EACH VIDEO
    # --------------------------------------------------------

    for video_path in video_files:

        print()
        print("=" * 60)
        print(f"Video: {video_path.name}")
        print("=" * 60)

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            print("[ERROR] Không mở được video")
            continue

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # Nếu video không đọc được FPS
        if fps <= 0:
            fps = 25.0

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )

        print(f"Resolution : {width}x{height}")
        print(f"FPS        : {fps:.2f}")
        print(f"Frames     : {total_frames}")
        print(f"Duration   : {duration:.2f}s")

        # ----------------------------------------------------
        # OUTPUT VIDEO
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            f"result_{video_path.stem}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        frame_number = 0

        class_counts = {
            "sleeping": 0,
            "using_phone": 0,
            "talking": 0
        }

        start_time = time.time()

        # ----------------------------------------------------
        # READ VIDEO
        # ----------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ------------------------------------------------
            # YOLO PREDICTION
            # ------------------------------------------------

            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )

            result = results[0]

            # ------------------------------------------------
            # COUNT DETECTIONS
            # ------------------------------------------------

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                class_name = (
                    model.names[class_id]
                )

                if class_name in class_counts:
                    class_counts[class_name] += 1

            # ------------------------------------------------
            # DRAW BOXES
            # ------------------------------------------------

            annotated_frame = (
                result.plot()
            )

            # Frame information
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_number}/{total_frames}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            writer.write(
                annotated_frame
            )

            # ------------------------------------------------
            # TERMINAL PROGRESS
            # ------------------------------------------------

            if (
                frame_number % 100 == 0
                or frame_number == total_frames
            ):

                percent = (
                    frame_number
                    / total_frames
                    * 100
                    if total_frames > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{frame_number}/"
                    f"{total_frames} "
                    f"({percent:.1f}%)"
                )

        # ----------------------------------------------------
        # FINISH
        # ----------------------------------------------------

        cap.release()
        writer.release()

        elapsed = (
            time.time() - start_time
        )

        print()
        print("-" * 60)
        print("VIDEO RESULT")
        print("-" * 60)

        print(
            f"Frames processed : "
            f"{frame_number}"
        )

        print(
            f"Processing time   : "
            f"{elapsed:.2f}s"
        )

        if elapsed > 0:
            processing_fps = (
                frame_number / elapsed
            )

            print(
                f"Processing FPS    : "
                f"{processing_fps:.2f}"
            )

        print()
        print("Detection counts:")

        for class_name, count in class_counts.items():

            print(
                f"  {class_name:12}: "
                f"{count}"
            )

        print()
        print("Saved:")
        print(output_path)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()