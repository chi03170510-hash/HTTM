from pathlib import Path
from ultralytics import YOLO
import cv2


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "test_results"


# ============================================================
# CONFIG
# ============================================================

CONF_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TEST YOLO - STUDENT BEHAVIOR")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Không tìm thấy model:")
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

    image_files = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("[ERROR] Chưa có ảnh trong:")
        print(INPUT_DIR)
        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading model:")
    print(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    print()
    print("Classes:")
    print(model.names)

    print()
    print(f"Found {len(image_files)} test images")
    print()

    # ========================================================
    # PREDICT
    # ========================================================

    total_detections = 0

    for image_path in image_files:

        print("-" * 60)
        print(f"Image: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes

        print(
            f"Detections: {len(boxes)}"
        )

        # ====================================================
        # PRINT DETECTIONS
        # ====================================================

        for index, box in enumerate(
            boxes,
            start=1
        ):

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = (
                model.names[class_id]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            print(
                f"  {index}. "
                f"{class_name:12} | "
                f"confidence = "
                f"{confidence:.2%} | "
                f"box = "
                f"({x1}, {y1}) "
                f"({x2}, {y2})"
            )

            total_detections += 1

        # ====================================================
        # VẼ BOX
        # Ultralytics tự vẽ class + confidence
        # ====================================================

        annotated_image = (
            result.plot()
        )

        output_path = (
            OUTPUT_DIR /
            f"result_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            annotated_image
        )

        print(
            f"Saved -> {output_path.name}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Images tested     : "
        f"{len(image_files)}"
    )

    print(
        f"Total detections  : "
        f"{total_detections}"
    )

    print()
    print("Results folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()