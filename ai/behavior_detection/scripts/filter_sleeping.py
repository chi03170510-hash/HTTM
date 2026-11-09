from pathlib import Path
import shutil


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = Path(r"E:\datasets")
OUTPUT_DIR = BASE_DIR / "datasets" / "sleeping"


# ============================================================
# DATASET CONFIG
# ============================================================

DATASETS = [
    {
        "name": "student_behaviors",
        "path": RAW_DIR / "student_behaviors",

        # Student Behaviors:
        # 3 = Sleeping
        "sleep_class_id": 3,

        # Prefix tránh trùng tên ảnh khi merge
        "prefix": "sb"
    },

    {
        "name": "ambient_classroom",
        "path": RAW_DIR / "ambient_classroom",

        # Ambient Intelligence Classroom:
        # 0 = Drowsy-Sleeping
        "sleep_class_id": 0,

        "prefix": "ac"
    }
]


SPLITS = ["train", "valid", "test"]


# ============================================================
# TẠO OUTPUT FOLDER
# ============================================================

def create_output_folders():

    for split in SPLITS:

        images_dir = OUTPUT_DIR / split / "images"
        labels_dir = OUTPUT_DIR / split / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# TÌM ẢNH TƯƠNG ỨNG LABEL
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
# XỬ LÝ DATASET
# ============================================================

def process_dataset(dataset):

    dataset_path = dataset["path"]
    sleep_class_id = dataset["sleep_class_id"]
    prefix = dataset["prefix"]

    print("\n========================================")
    print(f"Processing: {dataset['name']}")
    print("========================================")

    total_images = 0
    total_boxes = 0

    for split in SPLITS:

        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        output_images = OUTPUT_DIR / split / "images"
        output_labels = OUTPUT_DIR / split / "labels"

        split_images = 0
        split_boxes = 0

        if not labels_dir.exists():

            print(f"[WARNING] Không tìm thấy: {labels_dir}")
            continue

        import os

        for entry in os.scandir(labels_dir):

            if not entry.is_file():
                continue

            if not entry.name.lower().endswith(".txt"):
                continue

            label_path = Path(entry.path)

            sleeping_annotations = []

            try:
                with open(entry.path, "r", encoding="utf-8") as file:
                    lines = file.readlines()

            except FileNotFoundError:
                print(f"[SKIP] Windows không đọc được file quá dài: {entry.name}")
                continue
            for line in lines:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))

                # Chỉ lấy Sleeping
                if class_id == sleep_class_id:

                    # Chuyển tất cả về class 0
                    parts[0] = "0"

                    sleeping_annotations.append(
                        " ".join(parts)
                    )

            # Không có sleeping → bỏ ảnh
            if not sleeping_annotations:
                continue

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:

                print(
                    f"[WARNING] Không tìm thấy ảnh cho "
                    f"{label_path.name}"
                )

                continue

            # Prefix tên file để tránh trùng
            new_stem = f"{prefix}_{label_path.stem}"

            new_image_name = (
                new_stem + image_path.suffix.lower()
            )

            new_label_name = new_stem + ".txt"

            # Copy ảnh
            shutil.copy2(
                image_path,
                output_images / new_image_name
            )

            # Tạo label mới
            with open(
                output_labels / new_label_name,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "\n".join(sleeping_annotations)
                )

                file.write("\n")

            split_images += 1
            split_boxes += len(sleeping_annotations)

        total_images += split_images
        total_boxes += split_boxes

        print(
            f"{split}: "
            f"{split_images} images | "
            f"{split_boxes} sleeping boxes"
        )

    print("----------------------------------------")

    print(
        f"TOTAL {dataset['name']}: "
        f"{total_images} images | "
        f"{total_boxes} boxes"
    )


# ============================================================
# DATA.YAML
# ============================================================

def create_yaml():

    yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

nc: 1

names:
  0: sleeping
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

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
    print("FILTER SLEEPING DATASET")
    print("========================================")

    create_output_folders()

    for dataset in DATASETS:

        process_dataset(dataset)

    create_yaml()

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(f"Output: {OUTPUT_DIR}")
    print("Class: 0 = sleeping")


if __name__ == "__main__":
    main()