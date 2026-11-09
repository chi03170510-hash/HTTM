from pathlib import Path
import cv2
import random
import shutil


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR /
    "datasets" /
    "behavior_3class"
)

OUTPUT_DIR = (
    BASE_DIR /
    "label_preview" /
    "behavior_3class"
)

SPLITS = [
    "train",
    "valid",
    "test"
]

# Mỗi split lấy 20 ảnh
# Tổng tối đa = 60 ảnh
SAMPLES_PER_SPLIT = 20

RANDOM_SEED = 42


# ============================================================
# CLASS
# ============================================================

CLASS_NAMES = {
    0: "sleeping",
    1: "using_phone",
    2: "talking"
}


# ============================================================
# MÀU BOUNDING BOX
# OpenCV dùng BGR
# ============================================================

CLASS_COLORS = {
    0: (0, 255, 0),      # sleeping
    1: (255, 0, 0),      # using_phone
    2: (0, 0, 255)       # talking
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# VẼ LABEL
# ============================================================

def draw_labels(image, label_path):

    height, width = image.shape[:2]

    try:
        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

    except Exception as error:

        print(
            f"[ERROR] Không đọc được label: "
            f"{label_path}"
        )

        print(error)

        return image

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        # ====================================================
        # YOLO normalized -> pixel
        # ====================================================

        x_center *= width
        y_center *= height

        box_width *= width
        box_height *= height

        x1 = int(
            x_center -
            box_width / 2
        )

        y1 = int(
            y_center -
            box_height / 2
        )

        x2 = int(
            x_center +
            box_width / 2
        )

        y2 = int(
            y_center +
            box_height / 2
        )

        # Giới hạn box trong ảnh
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            width - 1,
            x2
        )

        y2 = min(
            height - 1,
            y2
        )

        class_name = (
            CLASS_NAMES[class_id]
        )

        color = (
            CLASS_COLORS[class_id]
        )

        # ====================================================
        # RECTANGLE
        # ====================================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ====================================================
        # LABEL TEXT
        # ====================================================

        text = (
            f"{class_id}: "
            f"{class_name}"
        )

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        text_y = max(
            y1,
            text_height + 8
        )

        # Background text
        cv2.rectangle(
            image,
            (
                x1,
                text_y -
                text_height -
                8
            ),
            (
                x1 +
                text_width +
                6,
                text_y +
                baseline
            ),
            color,
            -1
        )

        cv2.putText(
            image,
            text,
            (
                x1 + 3,
                text_y - 3
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return image


# ============================================================
# KIỂM TRA MỘT SPLIT
# ============================================================

def check_split(split):

    images_dir = (
        DATASET_DIR /
        split /
        "images"
    )

    labels_dir = (
        DATASET_DIR /
        split /
        "labels"
    )

    output_split_dir = (
        OUTPUT_DIR /
        split
    )

    output_split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not images_dir.exists():

        print(
            f"[WARNING] Không tồn tại: "
            f"{images_dir}"
        )

        return 0

    image_files = [
        path
        for path in images_dir.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    if not image_files:

        print(
            f"[WARNING] Không có ảnh "
            f"trong {split}"
        )

        return 0

    # ========================================================
    # RANDOM SAMPLE
    # ========================================================

    number_to_sample = min(
        SAMPLES_PER_SPLIT,
        len(image_files)
    )

    selected_images = random.sample(
        image_files,
        number_to_sample
    )

    saved = 0

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"[SKIP] Không có label: "
                f"{image_path.name}"
            )

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[SKIP] Không đọc được ảnh: "
                f"{image_path.name}"
            )

            continue

        image = draw_labels(
            image,
            label_path
        )

        output_name = (
            f"{index:02d}_"
            f"{image_path.name}"
        )

        output_path = (
            output_split_dir /
            output_name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        saved += 1

    return saved


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CHECK FINAL 3-CLASS LABELS")
    print("=" * 60)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    print()
    print("Classes:")
    print("0 = sleeping")
    print("1 = using_phone")
    print("2 = talking")

    # Xóa preview cũ của behavior_3class
    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    random.seed(
        RANDOM_SEED
    )

    total_saved = 0

    for split in SPLITS:

        print()
        print(
            f"Checking: {split}"
        )

        saved = check_split(
            split
        )

        total_saved += saved

        print(
            f"Saved: {saved} previews"
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Total preview images: "
        f"{total_saved}"
    )

    print()
    print(
        f"Preview folder:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()