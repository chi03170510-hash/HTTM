from pathlib import Path
import shutil
import hashlib
import random
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"E:\TTT\Project\HTTM")

RAW_STUDENT = Path(r"E:\datasets\student_behaviors")
RAW_AMBIENT = Path(r"E:\datasets\ambient_classroom")

OUTPUT_DIR = (
    PROJECT_DIR
    / "ai"
    / "behavior_detection"
    / "datasets"
    / "behavior_v1_1"
)

# ------------------------------------------------------------
# FINAL CLASSES
# ------------------------------------------------------------

SLEEPING = 0
USING_PHONE = 1

CLASS_NAMES = {
    0: "sleeping",
    1: "using_phone",
}

# ------------------------------------------------------------
# RAW DATASET MAPPING
# ------------------------------------------------------------

DATASETS = [
    {
        "name": "student_behaviors",
        "path": RAW_STUDENT,
        "prefix": "sb",

        # Raw -> final
        "mapping": {
            3: SLEEPING,
            4: USING_PHONE,
        },

        # Các class được phép dùng làm negative candidate
        #
        # Student Behaviors:
        # 0 Distracting
        # 1 Focusing
        # 2 Hand Raising
        # 3 Sleeping
        # 4 Using Phone
        # 5 Walking
        #
        # Không lấy Distracting vì semantic quá rộng.
        "negative_classes": {
            1,  # Focusing
            2,  # Hand Raising
            5,  # Walking
        },
    },

    {
        "name": "ambient_classroom",
        "path": RAW_AMBIENT,
        "prefix": "ac",

        # Raw -> final
        "mapping": {
            0: SLEEPING,
            7: USING_PHONE,
        },

        # Ambient Classroom:
        # 0 Drowsy-Sleeping
        # 1 Eating-Drinking
        # 2 Focused-Thinking
        # 3 Looking down
        # 4 Looking upfront
        # 5 Raising Hand
        # 6 Using-Laptop-Writing
        # 7 Using-Phone
        "negative_classes": {
            1,  # Eating-Drinking
            2,  # Focused-Thinking
            3,  # Looking down
            4,  # Looking upfront
            5,  # Raising Hand
            6,  # Using-Laptop-Writing
        },
    },
]

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]

# ------------------------------------------------------------
# SPLIT POSITIVE
# ------------------------------------------------------------

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# ------------------------------------------------------------
# HARD NEGATIVES
# ------------------------------------------------------------

# Chỉ thêm negative vào TRAIN.
MAX_NEGATIVE_IMAGES = 1000

# Cố gắng không để một nguồn chiếm toàn bộ negatives.
MAX_NEGATIVE_PER_SOURCE = 600

RANDOM_SEED = 42


# ============================================================
# WINDOWS LONG PATH
# ============================================================

def windows_long_path(path):
    path = str(Path(path).resolve())

    if not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + path

    return path


# ============================================================
# FILE HELPERS
# ============================================================

def image_hash(path):
    h = hashlib.md5()

    try:
        file_path = windows_long_path(path)

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)

                if not chunk:
                    break

                h.update(chunk)

    except Exception:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)

                if not chunk:
                    break

                h.update(chunk)

    return h.hexdigest()


def find_image(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


# ============================================================
# READ RAW YOLO LABELS
# ============================================================

def read_raw_labels(label_path):
    """
    Return:
        [
            (class_id, x, y, w, h),
            ...
        ]
    """

    labels = []

    if not label_path.exists():
        return labels

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    except Exception:
        return labels

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            continue

        labels.append(
            (cls, x, y, w, h)
        )

    return labels


def get_target_labels(raw_labels, mapping):
    """
    Chỉ giữ sleeping + using_phone,
    đồng thời remap về 0/1.
    """

    target_labels = []

    for (
        raw_class,
        x,
        y,
        w,
        h,
    ) in raw_labels:

        if raw_class not in mapping:
            continue

        final_class = mapping[
            raw_class
        ]

        target_labels.append(
            f"{final_class} "
            f"{x} {y} {w} {h}"
        )

    return target_labels


# ============================================================
# POSITIVE COLLECTION
# ============================================================

def collect_positive_dataset(dataset):

    dataset_name = dataset["name"]
    root = dataset["path"]
    prefix = dataset["prefix"]
    mapping = dataset["mapping"]

    records = []

    print()
    print("=" * 65)
    print(f"COLLECT POSITIVE: {dataset_name}")
    print("=" * 65)

    for split in [
        "train",
        "valid",
        "test",
    ]:

        images_dir = (
            root
            / split
            / "images"
        )

        labels_dir = (
            root
            / split
            / "labels"
        )

        if not images_dir.exists():
            print(
                f"[WARNING] Missing: {images_dir}"
            )
            continue

        if not labels_dir.exists():
            print(
                f"[WARNING] Missing: {labels_dir}"
            )
            continue

        image_count = 0
        box_counter = Counter()

        for label_path in labels_dir.glob(
            "*.txt"
        ):

            raw_labels = read_raw_labels(
                label_path
            )

            target_labels = get_target_labels(
                raw_labels,
                mapping
            )

            # Positive dataset:
            # bắt buộc phải có ít nhất 1 target bbox.
            if not target_labels:
                continue

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:
                continue

            try:
                h = image_hash(
                    image_path
                )
            except Exception as e:
                print(
                    f"[SKIP IMAGE] "
                    f"{image_path}: {e}"
                )
                continue

            for line in target_labels:
                cls = int(
                    line.split()[0]
                )

                box_counter[cls] += 1

            records.append(
                {
                    "type": "positive",
                    "dataset": dataset_name,
                    "prefix": prefix,
                    "original_split": split,
                    "image_path": image_path,
                    "labels": target_labels,
                    "hash": h,
                }
            )

            image_count += 1

        print(
            f"{split:<6} "
            f"images={image_count:<5} "
            f"sleeping={box_counter[0]:<5} "
            f"phone={box_counter[1]:<5}"
        )

    print(
        f"Positive records: "
        f"{len(records)}"
    )

    return records


# ============================================================
# POSITIVE DEDUPLICATION
# ============================================================

def merge_positive_duplicates(records):

    merged = {}
    duplicates = 0

    for record in records:
        h = record["hash"]

        if h not in merged:
            merged[h] = record.copy()

            merged[h]["labels"] = set(
                record["labels"]
            )

        else:
            duplicates += 1

            merged[h]["labels"].update(
                record["labels"]
            )

    final_records = []

    for record in merged.values():
        record["labels"] = sorted(
            record["labels"]
        )

        final_records.append(
            record
        )

    print()
    print("=" * 65)
    print("POSITIVE DEDUPLICATION")
    print("=" * 65)

    print(
        f"Before : {len(records)}"
    )

    print(
        f"After  : {len(final_records)}"
    )

    print(
        f"Merged : {duplicates}"
    )

    return final_records


# ============================================================
# SPLIT POSITIVES
# ============================================================

def split_positive_dataset(records):

    random.seed(
        RANDOM_SEED
    )

    records = records.copy()

    random.shuffle(
        records
    )

    total = len(records)

    train_end = int(
        total * TRAIN_RATIO
    )

    valid_end = (
        train_end
        + int(total * VALID_RATIO)
    )

    return {
        "train": records[:train_end],
        "valid": records[
            train_end:valid_end
        ],
        "test": records[
            valid_end:
        ],
    }


# ============================================================
# NEGATIVE COLLECTION
# ============================================================

def collect_negative_candidates(
    dataset,
    positive_hashes,
):
    """
    Negative hợp lệ khi:

    1. Ảnh KHÔNG chứa sleeping/using_phone.
    2. Có ít nhất một annotation thuộc nhóm negative_classes.
    3. Không duplicate với positive.
    """

    dataset_name = dataset["name"]
    root = dataset["path"]
    prefix = dataset["prefix"]
    mapping = dataset["mapping"]
    negative_classes = dataset[
        "negative_classes"
    ]

    records = []

    print()
    print("=" * 65)
    print(
        f"COLLECT NEGATIVE CANDIDATES: "
        f"{dataset_name}"
    )
    print("=" * 65)

    # Ta có thể đọc candidate từ raw train/valid/test,
    # nhưng tất cả negative cuối cùng chỉ được đưa
    # vào FINAL TRAIN.
    for original_split in [
        "train",
        "valid",
        "test",
    ]:

        images_dir = (
            root
            / original_split
            / "images"
        )

        labels_dir = (
            root
            / original_split
            / "labels"
        )

        if not images_dir.exists():
            continue

        if not labels_dir.exists():
            continue

        count = 0

        for label_path in labels_dir.glob(
            "*.txt"
        ):

            raw_labels = read_raw_labels(
                label_path
            )

            if not raw_labels:
                continue

            raw_classes = {
                item[0]
                for item in raw_labels
            }

            # --------------------------------------------
            # QUAN TRỌNG:
            # Nếu ảnh có sleeping hoặc using_phone
            # thì KHÔNG được biến nó thành negative.
            # --------------------------------------------

            target_raw_classes = set(
                mapping.keys()
            )

            if raw_classes & target_raw_classes:
                continue

            # Phải có ít nhất một class
            # mà mình chủ động chọn làm negative.
            if not (
                raw_classes
                & negative_classes
            ):
                continue

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:
                continue

            try:
                h = image_hash(
                    image_path
                )

            except Exception:
                continue

            # Không cho negative duplicate
            # với bất kỳ positive image nào.
            if h in positive_hashes:
                continue

            records.append(
                {
                    "type": "negative",
                    "dataset": dataset_name,
                    "prefix": prefix,
                    "original_split": (
                        original_split
                    ),
                    "image_path": image_path,

                    # Negative YOLO label = empty
                    "labels": [],

                    "raw_classes": sorted(
                        raw_classes
                    ),

                    "hash": h,
                }
            )

            count += 1

        print(
            f"{original_split:<6} "
            f"candidates={count}"
        )

    print(
        f"Negative candidates: "
        f"{len(records)}"
    )

    return records


# ============================================================
# SELECT NEGATIVES
# ============================================================

def select_negatives(
    candidates_by_source
):
    """
    Chọn tối đa 1000 negatives.

    Mỗi source tối đa 600 để tránh
    một dataset áp đảo hoàn toàn.
    """

    random.seed(
        RANDOM_SEED + 1
    )

    selected = []
    selected_hashes = set()

    # --------------------------------------------
    # Deduplicate trong từng source trước
    # --------------------------------------------

    clean_sources = {}

    for source, records in (
        candidates_by_source.items()
    ):

        unique = {}

        for record in records:
            h = record["hash"]

            if h not in unique:
                unique[h] = record

        source_records = list(
            unique.values()
        )

        random.shuffle(
            source_records
        )

        clean_sources[source] = (
            source_records
        )

    # --------------------------------------------
    # Round 1:
    # lấy cân bằng giữa các source
    # --------------------------------------------

    source_names = list(
        clean_sources.keys()
    )

    if source_names:
        target_per_source = (
            MAX_NEGATIVE_IMAGES
            // len(source_names)
        )
    else:
        target_per_source = 0

    for source in source_names:

        source_records = (
            clean_sources[source]
        )

        limit = min(
            target_per_source,
            MAX_NEGATIVE_PER_SOURCE,
            len(source_records),
        )

        for record in (
            source_records[:limit]
        ):

            if (
                record["hash"]
                in selected_hashes
            ):
                continue

            selected.append(
                record
            )

            selected_hashes.add(
                record["hash"]
            )

    # --------------------------------------------
    # Round 2:
    # nếu chưa đủ 1000 thì fill thêm
    # --------------------------------------------

    if (
        len(selected)
        < MAX_NEGATIVE_IMAGES
    ):

        source_selected_count = Counter(
            record["dataset"]
            for record in selected
        )

        pool = []

        for source in source_names:

            for record in (
                clean_sources[source]
            ):

                if (
                    record["hash"]
                    in selected_hashes
                ):
                    continue

                pool.append(
                    record
                )

        random.shuffle(
            pool
        )

        for record in pool:

            if (
                len(selected)
                >= MAX_NEGATIVE_IMAGES
            ):
                break

            source = record[
                "dataset"
            ]

            if (
                source_selected_count[
                    source
                ]
                >= MAX_NEGATIVE_PER_SOURCE
            ):
                continue

            if (
                record["hash"]
                in selected_hashes
            ):
                continue

            selected.append(
                record
            )

            selected_hashes.add(
                record["hash"]
            )

            source_selected_count[
                source
            ] += 1

    random.shuffle(
        selected
    )

    return selected


# ============================================================
# OUTPUT
# ============================================================

def prepare_output():

    if OUTPUT_DIR.exists():

        print()
        print(
            f"Removing old output: "
            f"{OUTPUT_DIR}"
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    for split in [
        "train",
        "valid",
        "test",
    ]:

        (
            OUTPUT_DIR
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_DIR
            / split
            / "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )


def copy_record(
    record,
    split,
    index,
):
    images_dir = (
        OUTPUT_DIR
        / split
        / "images"
    )

    labels_dir = (
        OUTPUT_DIR
        / split
        / "labels"
    )

    source_image = record[
        "image_path"
    ]

    extension = (
        source_image
        .suffix
        .lower()
    )

    if record["type"] == "negative":
        type_prefix = "neg"
    else:
        type_prefix = "pos"

    new_stem = (
        f"{record['prefix']}_"
        f"{type_prefix}_"
        f"{split}_"
        f"{index:06d}"
    )

    destination_image = (
        images_dir
        / f"{new_stem}{extension}"
    )

    destination_label = (
        labels_dir
        / f"{new_stem}.txt"
    )

    # COPY IMAGE
    try:
        shutil.copy2(
            windows_long_path(
                source_image
            ),
            destination_image
        )

    except Exception as e:

        print(
            f"[COPY ERROR] "
            f"{source_image}: {e}"
        )

        return False

    # WRITE LABEL
    try:

        labels = record[
            "labels"
        ]

        if labels:
            destination_label.write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8"
            )

        else:
            # Negative image:
            # YOLO label file rỗng.
            destination_label.write_text(
                "",
                encoding="utf-8"
            )

    except Exception as e:

        print(
            f"[LABEL ERROR] "
            f"{destination_label}: {e}"
        )

        try:
            destination_image.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return False

    return True


def write_dataset(
    positive_splits,
    negatives,
):

    stats = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:

        records = list(
            positive_splits[split]
        )

        # Chỉ TRAIN có negatives.
        if split == "train":
            records.extend(
                negatives
            )

        # Shuffle để positive/negative
        # không nằm thành từng cụm.
        rng = random.Random(
            RANDOM_SEED + 100
        )

        rng.shuffle(
            records
        )

        written = 0
        positive_count = 0
        negative_count = 0

        boxes = Counter()

        for index, record in enumerate(
            records
        ):

            success = copy_record(
                record,
                split,
                index
            )

            if not success:
                continue

            written += 1

            if (
                record["type"]
                == "negative"
            ):

                negative_count += 1

            else:

                positive_count += 1

                for line in (
                    record["labels"]
                ):

                    cls = int(
                        line.split()[0]
                    )

                    boxes[cls] += 1

        stats[split] = {
            "images": written,
            "positive": positive_count,
            "negative": negative_count,
            "sleeping": boxes[0],
            "phone": boxes[1],
        }

    return stats


# ============================================================
# YAML
# ============================================================

def write_yaml():

    content = """path: .
train: train/images
val: valid/images
test: test/images

names:
  0: sleeping
  1: using_phone
"""

    (
        OUTPUT_DIR
        / "data.yaml"
    ).write_text(
        content,
        encoding="utf-8"
    )


# ============================================================
# VERIFY
# ============================================================

def verify_dataset():

    print()
    print("=" * 65)
    print("VERIFY FINAL DATASET")
    print("=" * 65)

    total_images = 0
    total_labels = 0
    total_empty = 0
    total_boxes = Counter()

    for split in [
        "train",
        "valid",
        "test",
    ]:

        images_dir = (
            OUTPUT_DIR
            / split
            / "images"
        )

        labels_dir = (
            OUTPUT_DIR
            / split
            / "labels"
        )

        images = []

        for ext in IMAGE_EXTENSIONS:
            images.extend(
                images_dir.glob(
                    f"*{ext}"
                )
            )

        labels = list(
            labels_dir.glob(
                "*.txt"
            )
        )

        empty_labels = 0
        boxes = Counter()

        for label_path in labels:

            content = (
                label_path
                .read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
                .strip()
            )

            if not content:
                empty_labels += 1
                continue

            for line in (
                content.splitlines()
            ):

                parts = (
                    line
                    .strip()
                    .split()
                )

                if len(parts) < 5:
                    continue

                try:
                    cls = int(
                        float(parts[0])
                    )

                except ValueError:
                    continue

                boxes[cls] += 1

        print()
        print(split.upper())
        print(
            f"Images          : "
            f"{len(images)}"
        )
        print(
            f"Label files     : "
            f"{len(labels)}"
        )
        print(
            f"Empty negatives : "
            f"{empty_labels}"
        )
        print(
            f"Sleeping boxes  : "
            f"{boxes[0]}"
        )
        print(
            f"Phone boxes     : "
            f"{boxes[1]}"
        )

        total_images += len(
            images
        )

        total_labels += len(
            labels
        )

        total_empty += (
            empty_labels
        )

        total_boxes.update(
            boxes
        )

    print()
    print("-" * 65)
    print(
        f"TOTAL IMAGES          : "
        f"{total_images}"
    )
    print(
        f"TOTAL LABEL FILES     : "
        f"{total_labels}"
    )
    print(
        f"TOTAL NEGATIVE IMAGES : "
        f"{total_empty}"
    )
    print(
        f"TOTAL SLEEPING BOXES  : "
        f"{total_boxes[0]}"
    )
    print(
        f"TOTAL PHONE BOXES     : "
        f"{total_boxes[1]}"
    )
    print("-" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print(
        "BUILD BEHAVIOR DATASET V1.1 "
        "+ HARD NEGATIVES"
    )
    print("=" * 65)

    print()
    print("Final classes:")
    print("0 = sleeping")
    print("1 = using_phone")

    # ========================================================
    # 1. COLLECT POSITIVES
    # ========================================================

    all_positive = []

    for dataset in DATASETS:

        records = (
            collect_positive_dataset(
                dataset
            )
        )

        all_positive.extend(
            records
        )

    print()
    print(
        f"Collected positive records: "
        f"{len(all_positive)}"
    )

    # ========================================================
    # 2. DEDUPLICATE POSITIVES
    # ========================================================

    positive_records = (
        merge_positive_duplicates(
            all_positive
        )
    )

    positive_hashes = {
        record["hash"]
        for record
        in positive_records
    }

    # ========================================================
    # 3. SPLIT POSITIVES
    # ========================================================

    positive_splits = (
        split_positive_dataset(
            positive_records
        )
    )

    print()
    print("=" * 65)
    print("POSITIVE SPLIT")
    print("=" * 65)

    print(
        f"Train: "
        f"{len(positive_splits['train'])}"
    )

    print(
        f"Valid: "
        f"{len(positive_splits['valid'])}"
    )

    print(
        f"Test : "
        f"{len(positive_splits['test'])}"
    )

    # ========================================================
    # 4. COLLECT NEGATIVE CANDIDATES
    # ========================================================

    candidates_by_source = {}

    for dataset in DATASETS:

        candidates = (
            collect_negative_candidates(
                dataset,
                positive_hashes,
            )
        )

        candidates_by_source[
            dataset["name"]
        ] = candidates

    # ========================================================
    # 5. SELECT NEGATIVES
    # ========================================================

    negatives = select_negatives(
        candidates_by_source
    )

    print()
    print("=" * 65)
    print("SELECTED HARD NEGATIVES")
    print("=" * 65)

    negative_sources = Counter(
        record["dataset"]
        for record
        in negatives
    )

    print(
        f"Total selected: "
        f"{len(negatives)}"
    )

    for source, count in (
        negative_sources.items()
    ):

        print(
            f"{source:<25}: "
            f"{count}"
        )

    # ========================================================
    # 6. PREPARE OUTPUT
    # ========================================================

    prepare_output()

    # ========================================================
    # 7. WRITE FINAL DATASET
    # ========================================================

    stats = write_dataset(
        positive_splits,
        negatives,
    )

    write_yaml()

    print()
    print("=" * 65)
    print("WRITE SUMMARY")
    print("=" * 65)

    for split in [
        "train",
        "valid",
        "test",
    ]:

        s = stats[split]

        print()
        print(split.upper())

        print(
            f"Images         : "
            f"{s['images']}"
        )

        print(
            f"Positive       : "
            f"{s['positive']}"
        )

        print(
            f"Negative       : "
            f"{s['negative']}"
        )

        print(
            f"Sleeping boxes : "
            f"{s['sleeping']}"
        )

        print(
            f"Phone boxes    : "
            f"{s['phone']}"
        )

    # ========================================================
    # 8. VERIFY
    # ========================================================

    verify_dataset()

    print()
    print("=" * 65)
    print("BUILD COMPLETE")
    print("=" * 65)

    print()
    print(
        "Dataset:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "IMPORTANT: DO NOT TRAIN YET."
    )

    print(
        "Next step: preview hard-negative images."
    )


if __name__ == "__main__":
    main()