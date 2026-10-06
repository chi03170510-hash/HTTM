from pathlib import Path
import shutil
import os


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset gốc được lưu ngoài project để tránh path quá dài
RAW_DIR = Path(r"E:\datasets")

# Dataset sau khi lọc Using Phone
OUTPUT_DIR = BASE_DIR / "datasets" / "using_phone"


# ============================================================
# DATASET CONFIG
# ============================================================

DATASETS = [
    {
        "name": "student_behaviors",
        "path": RAW_DIR / "student_behaviors",

        # Student Behaviors:
        # 4 = Using Phone
        "target_class_id": 4,

        # Prefix tránh trùng tên file
        "prefix": "sb"
    },

    {
        "name": "ambient_classroom",
        "path": RAW_DIR / "ambient_classroom",

        # Ambient Classroom:
        # 7 = Using-Phone
        "target_class_id": 7,

        # Prefix tránh trùng tên file
        "prefix": "ac"
    }
]


SPLITS = [
    "train",
    "valid",
    "test"
]


# ============================================================
# TẠO OUTPUT FOLDER
# ============================================================

def create_output_folders():

    for split in SPLITS:

        images_dir = OUTPUT_DIR / split / "images"
        labels_dir = OUTPUT_DIR / split / "labels"

        images_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        labels_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# TÌM ẢNH TƯƠNG ỨNG VỚI LABEL
# ============================================================

def find_image(images_dir, stem):

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]

    for ext in extensions:

        image_path = images_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


# ============================================================
# XỬ LÝ TỪNG DATASET
# ============================================================

def process_dataset(dataset):

    dataset_path = dataset["path"]
    target_class_id = dataset["target_class_id"]
    prefix = dataset["prefix"]

    print("\n========================================")
    print(f"Processing: {dataset['name']}")
    print("========================================")

    total_images = 0
    total_boxes = 0

    # --------------------------------------------------------
    # Xử lý train / valid / test
    # --------------------------------------------------------

    for split in SPLITS:

        images_dir = (
            dataset_path /
            split /
            "images"
        )

        labels_dir = (
            dataset_path /
            split /
            "labels"
        )

        output_images = (
            OUTPUT_DIR /
            split /
            "images"
        )

        output_labels = (
            OUTPUT_DIR /
            split /
            "labels"
        )

        split_images = 0
        split_boxes = 0

        # Kiểm tra folder labels
        if not labels_dir.exists():

            print(
                f"[WARNING] Không tìm thấy: "
                f"{labels_dir}"
            )

            continue

        # ----------------------------------------------------
        # Đọc tất cả file label
        # ----------------------------------------------------

        for entry in os.scandir(labels_dir):

            if not entry.is_file():
                continue

            if not entry.name.lower().endswith(".txt"):
                continue

            label_path = Path(entry.path)

            # Chỉ lưu annotation của Using Phone
            phone_annotations = []

            # ------------------------------------------------
            # Đọc label
            # ------------------------------------------------

            try:

                with open(
                    entry.path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    lines = file.readlines()

            except FileNotFoundError:

                print(
                    "[SKIP] Windows không đọc được "
                    f"file quá dài: {entry.name}"
                )

                continue

            # ------------------------------------------------
            # Lọc class Using Phone
            # ------------------------------------------------

            for line in lines:

                parts = line.strip().split()

                # YOLO label phải có:
                # class x_center y_center width height
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))

                except ValueError:
                    continue

                # Chỉ lấy Using Phone
                if class_id == target_class_id:

                    # Chuẩn hóa:
                    # Using Phone -> class 0
                    parts[0] = "0"

                    phone_annotations.append(
                        " ".join(parts)
                    )

            # ------------------------------------------------
            # Không có Using Phone -> bỏ ảnh
            # ------------------------------------------------

            if not phone_annotations:
                continue

            # ------------------------------------------------
            # Tìm ảnh tương ứng
            # ------------------------------------------------

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:

                print(
                    "[WARNING] Không tìm thấy ảnh cho "
                    f"{label_path.name}"
                )

                continue

            # ------------------------------------------------
            # Đổi tên để tránh trùng giữa 2 dataset
            #
            # Ví dụ:
            # sb_image001.jpg
            # ac_image001.jpg
            # ------------------------------------------------

            new_stem = (
                f"{prefix}_{label_path.stem}"
            )

            new_image_name = (
                new_stem +
                image_path.suffix.lower()
            )

            new_label_name = (
                new_stem +
                ".txt"
            )

            # ------------------------------------------------
            # Copy ảnh
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                output_images / new_image_name
            )

            # ------------------------------------------------
            # Ghi label mới
            # ------------------------------------------------

            with open(
                output_labels / new_label_name,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "\n".join(phone_annotations)
                )

                file.write("\n")

            # ------------------------------------------------
            # Thống kê
            # ------------------------------------------------

            split_images += 1

            split_boxes += len(
                phone_annotations
            )

        # ----------------------------------------------------
        # Tổng split
        # ----------------------------------------------------

        total_images += split_images
        total_boxes += split_boxes

        print(
            f"{split}: "
            f"{split_images} images | "
            f"{split_boxes} using_phone boxes"
        )

    # --------------------------------------------------------
    # Tổng dataset
    # --------------------------------------------------------

    print("----------------------------------------")

    print(
        f"TOTAL {dataset['name']}: "
        f"{total_images} images | "
        f"{total_boxes} boxes"
    )


# ============================================================
# TẠO DATA.YAML
# ============================================================

def create_yaml():

    yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

nc: 1

names:
  0: using_phone
"""

    yaml_path = (
        OUTPUT_DIR /
        "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(yaml_content)


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("FILTER USING PHONE DATASET")
    print("========================================")

    # Tạo folder output
    create_output_folders()

    # Xử lý từng dataset
    for dataset in DATASETS:

        process_dataset(dataset)

    # Tạo data.yaml
    create_yaml()

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(f"Output: {OUTPUT_DIR}")
    print("Class: 0 = using_phone")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()