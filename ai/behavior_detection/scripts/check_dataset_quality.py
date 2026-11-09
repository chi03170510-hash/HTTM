from pathlib import Path
import hashlib
import cv2


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = BASE_DIR / "datasets"

DATASETS = [
    "sleeping",
    "using_phone",
    "talking"
]

SPLITS = [
    "train",
    "valid",
    "test"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CẤU HÌNH KIỂM TRA
# ============================================================

# Box có width hoặc height nhỏ hơn ngưỡng này
# sẽ được đánh dấu là "very small".
#
# 0.01 = 1% chiều rộng / chiều cao ảnh.
SMALL_BOX_THRESHOLD = 0.01


# ============================================================
# HASH ẢNH
# Dùng để phát hiện file ảnh giống hệt nhau
# ============================================================

def calculate_md5(file_path):

    md5 = hashlib.md5()

    try:

        with open(file_path, "rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:
        return None


# ============================================================
# KIỂM TRA LABEL YOLO
# ============================================================

def check_label_file(label_path):

    result = {
        "invalid_lines": 0,
        "invalid_boxes": 0,
        "small_boxes": 0,
        "boxes": 0
    }

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

    except Exception:

        result["invalid_lines"] += 1
        return result

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # YOLO Detection:
        #
        # class x_center y_center width height
        #
        if len(parts) != 5:

            result["invalid_lines"] += 1
            continue

        try:

            class_id = int(float(parts[0]))

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            result["invalid_lines"] += 1
            continue

        # --------------------------------------------
        # Kiểm tra class
        #
        # Các dataset trung gian hiện tại đều chỉ có:
        #
        # 0 = behavior
        # --------------------------------------------

        if class_id != 0:

            result["invalid_lines"] += 1
            continue

        result["boxes"] += 1

        # --------------------------------------------
        # Kiểm tra tọa độ YOLO
        # --------------------------------------------

        if (
            x_center < 0 or
            x_center > 1 or
            y_center < 0 or
            y_center > 1 or
            width <= 0 or
            width > 1 or
            height <= 0 or
            height > 1
        ):

            result["invalid_boxes"] += 1
            continue

        # --------------------------------------------
        # Kiểm tra box có vượt khỏi ảnh không
        # --------------------------------------------

        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        if (
            x1 < 0 or
            y1 < 0 or
            x2 > 1 or
            y2 > 1
        ):

            result["invalid_boxes"] += 1
            continue

        # --------------------------------------------
        # Box quá nhỏ
        # --------------------------------------------

        if (
            width < SMALL_BOX_THRESHOLD or
            height < SMALL_BOX_THRESHOLD
        ):

            result["small_boxes"] += 1

    return result


# ============================================================
# KIỂM TRA MỘT DATASET
# ============================================================

def check_dataset(dataset_name):

    dataset_dir = DATASETS_DIR / dataset_name

    print()
    print("=" * 60)
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    total_images = 0
    total_labels = 0
    total_boxes = 0

    corrupt_images = []
    missing_labels = []
    orphan_labels = []

    invalid_label_files = []
    invalid_boxes = []
    small_boxes = []

    # Hash -> danh sách ảnh
    hashes = {}

    # ========================================================
    # KIỂM TRA TỪNG SPLIT
    # ========================================================

    for split in SPLITS:

        images_dir = (
            dataset_dir /
            split /
            "images"
        )

        labels_dir = (
            dataset_dir /
            split /
            "labels"
        )

        print()
        print(f"Checking split: {split}")

        if not images_dir.exists():

            print(
                f"[WARNING] Không tồn tại: "
                f"{images_dir}"
            )

            continue

        if not labels_dir.exists():

            print(
                f"[WARNING] Không tồn tại: "
                f"{labels_dir}"
            )

            continue

        # ----------------------------------------------------
        # Lấy danh sách ảnh
        # ----------------------------------------------------

        image_files = [
            file
            for file in images_dir.iterdir()
            if (
                file.is_file()
                and
                file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

        label_files = list(
            labels_dir.glob("*.txt")
        )

        print(
            f"Images: {len(image_files)} | "
            f"Labels: {len(label_files)}"
        )

        total_images += len(image_files)
        total_labels += len(label_files)

        # ----------------------------------------------------
        # Stem của ảnh / label
        # ----------------------------------------------------

        image_stems = {
            file.stem
            for file in image_files
        }

        label_stems = {
            file.stem
            for file in label_files
        }

        # ----------------------------------------------------
        # Ảnh không có label
        # ----------------------------------------------------

        for stem in (
            image_stems -
            label_stems
        ):

            missing_labels.append(
                f"{split}/{stem}"
            )

        # ----------------------------------------------------
        # Label không có ảnh
        # ----------------------------------------------------

        for stem in (
            label_stems -
            image_stems
        ):

            orphan_labels.append(
                f"{split}/{stem}"
            )

        # ----------------------------------------------------
        # Kiểm tra ảnh
        # ----------------------------------------------------

        for image_path in image_files:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                corrupt_images.append(
                    f"{split}/{image_path.name}"
                )

                continue

            # --------------------------------------------
            # Hash để tìm ảnh giống hệt nhau
            # --------------------------------------------

            image_hash = calculate_md5(
                image_path
            )

            if image_hash is not None:

                if image_hash not in hashes:

                    hashes[image_hash] = []

                hashes[image_hash].append(
                    f"{split}/{image_path.name}"
                )

        # ----------------------------------------------------
        # Kiểm tra labels
        # ----------------------------------------------------

        for label_path in label_files:

            result = check_label_file(
                label_path
            )

            total_boxes += (
                result["boxes"]
            )

            if (
                result["invalid_lines"] > 0
            ):

                invalid_label_files.append(
                    (
                        f"{split}/"
                        f"{label_path.name}",
                        result["invalid_lines"]
                    )
                )

            if (
                result["invalid_boxes"] > 0
            ):

                invalid_boxes.append(
                    (
                        f"{split}/"
                        f"{label_path.name}",
                        result["invalid_boxes"]
                    )
                )

            if (
                result["small_boxes"] > 0
            ):

                small_boxes.append(
                    (
                        f"{split}/"
                        f"{label_path.name}",
                        result["small_boxes"]
                    )
                )

    # ========================================================
    # DUPLICATE IMAGES
    # ========================================================

    duplicate_groups = []

    for image_hash, files in hashes.items():

        if len(files) > 1:

            duplicate_groups.append(
                files
            )

    duplicate_image_count = sum(
        len(group) - 1
        for group in duplicate_groups
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)

    print(
        f"Total images       : {total_images}"
    )

    print(
        f"Total labels       : {total_labels}"
    )

    print(
        f"Total boxes        : {total_boxes}"
    )

    print(
        f"Corrupt images     : "
        f"{len(corrupt_images)}"
    )

    print(
        f"Missing labels     : "
        f"{len(missing_labels)}"
    )

    print(
        f"Orphan labels      : "
        f"{len(orphan_labels)}"
    )

    print(
        f"Invalid label files: "
        f"{len(invalid_label_files)}"
    )

    print(
        f"Invalid box files  : "
        f"{len(invalid_boxes)}"
    )

    print(
        f"Small box files    : "
        f"{len(small_boxes)}"
    )

    print(
        f"Duplicate groups   : "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Duplicate images   : "
        f"{duplicate_image_count}"
    )

    # ========================================================
    # IN MỘT SỐ LỖI ĐỂ KIỂM TRA
    # ========================================================

    if corrupt_images:

        print()
        print("Ví dụ corrupt images:")

        for item in corrupt_images[:10]:
            print("  ", item)

    if missing_labels:

        print()
        print("Ví dụ images thiếu label:")

        for item in missing_labels[:10]:
            print("  ", item)

    if orphan_labels:

        print()
        print("Ví dụ label không có ảnh:")

        for item in orphan_labels[:10]:
            print("  ", item)

    if invalid_label_files:

        print()
        print("Ví dụ invalid label:")

        for item, count in invalid_label_files[:10]:

            print(
                f"   {item} "
                f"({count} lỗi)"
            )

    if invalid_boxes:

        print()
        print("Ví dụ invalid box:")

        for item, count in invalid_boxes[:10]:

            print(
                f"   {item} "
                f"({count} box)"
            )

    if small_boxes:

        print()
        print("Ví dụ very small box:")

        for item, count in small_boxes[:10]:

            print(
                f"   {item} "
                f"({count} box)"
            )

    if duplicate_groups:

        print()
        print("Ví dụ duplicate images:")

        # Chỉ in tối đa 5 nhóm
        for group in duplicate_groups[:5]:

            print("   ---")

            for file in group:

                print(
                    f"   {file}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DATASET QUALITY CHECK")
    print("=" * 60)

    for dataset_name in DATASETS:

        dataset_dir = (
            DATASETS_DIR /
            dataset_name
        )

        if not dataset_dir.exists():

            print()
            print(
                f"[WARNING] Dataset không tồn tại: "
                f"{dataset_dir}"
            )

            continue

        check_dataset(
            dataset_name
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "Script chỉ kiểm tra dữ liệu, "
        "KHÔNG xóa hoặc sửa file."
    )


if __name__ == "__main__":
    main()