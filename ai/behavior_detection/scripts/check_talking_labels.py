from pathlib import Path
import random
import cv2

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "talking"

OUTPUT_DIR = BASE_DIR / "label_preview" / "talking"

SPLITS = ["train", "valid", "test"]

# Số ảnh lấy ngẫu nhiên ở mỗi split
SAMPLES_PER_SPLIT = 10

# Để lần chạy sau vẫn lấy cùng bộ ảnh
random.seed(42)


# =========================
# FUNCTIONS
# =========================

def get_images(images_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]


def draw_yolo_boxes(image_path, label_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Không đọc được ảnh: {image_path}")
        return None

    height, width = image.shape[:2]

    if not label_path.exists():
        print(f"[WARNING] Không tìm thấy label: {label_path.name}")
        return image

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # YOLO normalized -> pixel
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Giữ box trong ảnh
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        # Vẽ bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Tên class
        label_text = f"talking (class {class_id})"

        cv2.putText(
            image,
            label_text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


def main():

    print("========================================")
    print("CHECK TALKING LABELS")
    print("========================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for split in SPLITS:

        print(f"\nChecking: {split}")

        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        if not images_dir.exists():
            print(f"[WARNING] Không tìm thấy: {images_dir}")
            continue

        images = get_images(images_dir)

        if not images:
            print("[WARNING] Không có ảnh.")
            continue

        sample_count = min(SAMPLES_PER_SPLIT, len(images))

        selected_images = random.sample(
            images,
            sample_count
        )

        split_output = OUTPUT_DIR / split
        split_output.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:

            label_path = labels_dir / (
                image_path.stem + ".txt"
            )

            result = draw_yolo_boxes(
                image_path,
                label_path
            )

            if result is None:
                continue

            output_path = (
                split_output /
                f"preview_{image_path.name}"
            )

            cv2.imwrite(
                str(output_path),
                result
            )

            print(f"[OK] {output_path.name}")

            total_saved += 1

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(f"Đã tạo {total_saved} ảnh preview")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()