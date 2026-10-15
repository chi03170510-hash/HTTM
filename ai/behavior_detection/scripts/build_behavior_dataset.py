from pathlib import Path
import shutil
import hashlib
import random
import cv2
import os


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = Path(r"E:\datasets")

OUTPUT_DIR = BASE_DIR / "datasets" / "behavior_3class"

# Chia lại toàn bộ dataset sau khi gộp + loại duplicate
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]


# ============================================================
# CLASS CUỐI
#
# 0 = sleeping
# 1 = using_phone
# 2 = talking
# ============================================================

FINAL_CLASSES = {
    0: "sleeping",
    1: "using_phone",
    2: "talking"
}


# ============================================================
# MAPPING DATASET GỐC -> CLASS CUỐI
# ============================================================

DATASETS = [
    {
        "name": "student_behaviors",
        "path": RAW_DIR / "student_behaviors",
        "prefix": "sb",

        # Dataset gốc:
        # 3 = Sleeping
        # 4 = Using Phone
        #
        # Dataset cuối:
        # 0 = sleeping
        # 1 = using_phone
        "class_map": {
            3: 0,
            4: 1
        }
    },

    {
        "name": "ambient_classroom",
        "path": RAW_DIR / "ambient_classroom",
        "prefix": "ac",

        # Dataset gốc:
        # 0 = Drowsy-Sleeping
        # 7 = Using-Phone
        #
        # Dataset cuối:
        # 0 = sleeping
        # 1 = using_phone
        "class_map": {
            0: 0,
            7: 1
        }
    },

    {
        "name": "student_classroom_behavior",
        "path": RAW_DIR / "student_classroom_behavior",
        "prefix": "scb",

        # Dataset gốc:
        # 3 = Talking
        #
        # Dataset cuối:
        # 2 = talking
        "class_map": {
            3: 2
        }
    }
]


SOURCE_SPLITS = [
    "train",
    "valid",
    "test"
]


# ============================================================
# TÌM ẢNH TƯƠNG ỨNG
# ============================================================

def find_image(images_dir, stem):

    for extension in IMAGE_EXTENSIONS:

        image_path = images_dir / f"{stem}{extension}"

        if image_path.exists():
            return image_path

    return None


# ============================================================
# HASH ẢNH
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
# ĐỌC LABEL VÀ MAP CLASS
# ============================================================

def read_target_annotations(label_path, class_map):

    annotations = []

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

    except (FileNotFoundError, OSError):

        return annotations

    for line in lines:

        parts = line.strip().split()

        # Cần ít nhất:
        # class x_center y_center width height
        if len(parts) < 5:
            continue

        try:

            old_class = int(float(parts[0]))

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            continue

        # Không phải class chúng ta cần
        if old_class not in class_map:
            continue

        # Kiểm tra tọa độ cơ bản
        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            continue

        # Kiểm tra box có vượt ảnh không
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        if (
            x1 < 0
            or y1 < 0
            or x2 > 1
            or y2 > 1
        ):
            continue

        new_class = class_map[old_class]

        # QUAN TRỌNG:
        # chỉ ghi đúng 5 trường YOLO Detection.
        annotation = (
            f"{new_class} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

        annotations.append(annotation)

    return annotations


# ============================================================
# THU THẬP DỮ LIỆU TỪ RAW
# ============================================================

def collect_samples():

    samples = []

    print("=" * 60)
    print("COLLECT RAW DATA")
    print("=" * 60)

    for dataset in DATASETS:

        dataset_path = dataset["path"]
        prefix = dataset["prefix"]
        class_map = dataset["class_map"]

        print()
        print(f"Dataset: {dataset['name']}")

        dataset_count = 0

        for split in SOURCE_SPLITS:

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

            if not images_dir.exists():

                print(
                    f"[WARNING] Không tìm thấy: "
                    f"{images_dir}"
                )

                continue

            if not labels_dir.exists():

                print(
                    f"[WARNING] Không tìm thấy: "
                    f"{labels_dir}"
                )

                continue

            split_count = 0

            # Dùng scandir để hạn chế vấn đề path dài
            for entry in os.scandir(labels_dir):

                if not entry.is_file():
                    continue

                if not entry.name.lower().endswith(".txt"):
                    continue

                label_path = Path(entry.path)

                annotations = read_target_annotations(
                    label_path,
                    class_map
                )

                # Ảnh không chứa 1 trong 3 hành vi mục tiêu
                # thì tạm thời không lấy ở bước này.
                if not annotations:
                    continue

                image_path = find_image(
                    images_dir,
                    label_path.stem
                )

                if image_path is None:
                    continue

                # Kiểm tra OpenCV đọc được ảnh
                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    continue

                image_hash = calculate_md5(
                    image_path
                )

                if image_hash is None:
                    continue

                samples.append(
                    {
                        "source": dataset["name"],
                        "prefix": prefix,
                        "original_split": split,
                        "image_path": image_path,
                        "annotations": annotations,
                        "hash": image_hash
                    }
                )

                split_count += 1
                dataset_count += 1

            print(
                f"  {split}: "
                f"{split_count} target images"
            )

        print(
            f"  TOTAL: "
            f"{dataset_count} target images"
        )

    return samples


# ============================================================
# GỘP ẢNH DUPLICATE
#
# Nếu cùng một ảnh xuất hiện ở nhiều dataset:
# - chỉ giữ 1 ảnh
# - gộp annotation
# ============================================================

def merge_duplicates(samples):

    print()
    print("=" * 60)
    print("MERGE DUPLICATE IMAGES")
    print("=" * 60)

    grouped = {}

    for sample in samples:

        image_hash = sample["hash"]

        if image_hash not in grouped:

            grouped[image_hash] = {
                "image_path": sample["image_path"],
                "prefix": sample["prefix"],
                "annotations": set(),
                "sources": set()
            }

        grouped[image_hash]["annotations"].update(
            sample["annotations"]
        )

        grouped[image_hash]["sources"].add(
            sample["source"]
        )

    merged_samples = []

    for image_hash, item in grouped.items():

        merged_samples.append(
            {
                "hash": image_hash,
                "image_path": item["image_path"],
                "prefix": item["prefix"],
                "annotations": sorted(
                    item["annotations"]
                ),
                "sources": sorted(
                    item["sources"]
                )
            }
        )

    print(
        f"Before merge : {len(samples)} images"
    )

    print(
        f"After merge  : {len(merged_samples)} images"
    )

    print(
        f"Duplicates removed/merged: "
        f"{len(samples) - len(merged_samples)}"
    )

    return merged_samples


# ============================================================
# ĐẾM CLASS TRONG SAMPLE
# ============================================================

def get_classes(sample):

    classes = set()

    for annotation in sample["annotations"]:

        parts = annotation.split()

        if not parts:
            continue

        classes.add(
            int(parts[0])
        )

    return classes


# ============================================================
# SHUFFLE + SPLIT
# ============================================================

def split_dataset(samples):

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        samples
    )

    total = len(samples)

    train_end = int(
        total * TRAIN_RATIO
    )

    valid_end = train_end + int(
        total * VALID_RATIO
    )

    train_samples = (
        samples[:train_end]
    )

    valid_samples = (
        samples[train_end:valid_end]
    )

    test_samples = (
        samples[valid_end:]
    )

    return {
        "train": train_samples,
        "valid": valid_samples,
        "test": test_samples
    }


# ============================================================
# TẠO OUTPUT FOLDERS
# ============================================================

def prepare_output():

    # Xóa dataset cũ nếu script đã chạy trước đó
    if OUTPUT_DIR.exists():

        print()
        print(
            f"Xóa output cũ: {OUTPUT_DIR}"
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    for split in [
        "train",
        "valid",
        "test"
    ]:

        (
            OUTPUT_DIR /
            split /
            "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_DIR /
            split /
            "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# GHI DATASET
# ============================================================

def write_dataset(split_data):

    print()
    print("=" * 60)
    print("WRITE FINAL DATASET")
    print("=" * 60)

    total_class_boxes = {
        0: 0,
        1: 0,
        2: 0
    }

    for split, samples in split_data.items():

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

        split_class_boxes = {
            0: 0,
            1: 0,
            2: 0
        }

        for index, sample in enumerate(
            samples,
            start=1
        ):

            image_path = sample["image_path"]

            # Dùng hash ngắn trong tên để tránh trùng
            short_hash = sample["hash"][:12]

            new_stem = (
                f"img_{index:06d}_"
                f"{short_hash}"
            )

            new_image_name = (
                new_stem +
                image_path.suffix.lower()
            )

            new_label_name = (
                new_stem +
                ".txt"
            )

            shutil.copy2(
                image_path,
                output_images / new_image_name
            )

            label_output = (
                output_labels /
                new_label_name
            )

            with open(
                label_output,
                "w",
                encoding="utf-8"
            ) as file:

                for annotation in sample["annotations"]:

                    file.write(
                        annotation + "\n"
                    )

                    class_id = int(
                        annotation.split()[0]
                    )

                    split_class_boxes[
                        class_id
                    ] += 1

                    total_class_boxes[
                        class_id
                    ] += 1

        print()
        print(
            f"{split}: "
            f"{len(samples)} images"
        )

        print(
            f"  sleeping boxes   : "
            f"{split_class_boxes[0]}"
        )

        print(
            f"  using_phone boxes: "
            f"{split_class_boxes[1]}"
        )

        print(
            f"  talking boxes    : "
            f"{split_class_boxes[2]}"
        )

    print()
    print("-" * 60)

    print(
        f"TOTAL sleeping boxes   : "
        f"{total_class_boxes[0]}"
    )

    print(
        f"TOTAL using_phone boxes: "
        f"{total_class_boxes[1]}"
    )

    print(
        f"TOTAL talking boxes    : "
        f"{total_class_boxes[2]}"
    )


# ============================================================
# DATA.YAML
# ============================================================

def create_yaml():

    yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

nc: 3

names:
  0: sleeping
  1: using_phone
  2: talking
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

        file.write(
            yaml_content
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILD 3-CLASS BEHAVIOR DATASET")
    print("=" * 60)

    print()
    print("Final classes:")
    print("0 = sleeping")
    print("1 = using_phone")
    print("2 = talking")

    # 1. Đọc dữ liệu raw
    samples = collect_samples()

    if not samples:

        print()
        print("[ERROR] Không tìm thấy dữ liệu.")
        return

    # 2. Merge duplicate
    samples = merge_duplicates(
        samples
    )

    # 3. Chia lại dataset
    split_data = split_dataset(
        samples
    )

    print()
    print("=" * 60)
    print("NEW SPLIT")
    print("=" * 60)

    print(
        f"Train: "
        f"{len(split_data['train'])}"
    )

    print(
        f"Valid: "
        f"{len(split_data['valid'])}"
    )

    print(
        f"Test : "
        f"{len(split_data['test'])}"
    )

    # 4. Tạo output
    prepare_output()

    # 5. Ghi dataset
    write_dataset(
        split_data
    )

    # 6. data.yaml
    create_yaml()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Dataset: {OUTPUT_DIR}"
    )

    print()
    print("Classes:")
    print("0 = sleeping")
    print("1 = using_phone")
    print("2 = talking")


if __name__ == "__main__":
    main()