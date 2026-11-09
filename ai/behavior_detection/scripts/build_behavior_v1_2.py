from pathlib import Path
from PIL import Image
import imagehash
import shutil
import hashlib
import random
from collections import Counter, defaultdict


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
    / "behavior_v1_2"
)

# ============================================================
# FINAL CLASSES
# ============================================================

SLEEPING = 0
USING_PHONE = 1

CLASS_NAMES = {
    0: "sleeping",
    1: "using_phone",
}

# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    {
        "name": "student_behaviors",
        "path": RAW_STUDENT,
        "prefix": "sb",

        # Raw class -> final class
        "mapping": {
            3: SLEEPING,
            4: USING_PHONE,
        },

        # Student Behaviors
        # 0 Distracting
        # 1 Focusing
        # 2 Hand Raising
        # 3 Sleeping
        # 4 Using Phone
        # 5 Walking
        #
        # Không lấy Distracting làm negative vì quá rộng.
        "negative_classes": {
            1,
            2,
            5,
        },
    },

    {
        "name": "ambient_classroom",
        "path": RAW_AMBIENT,
        "prefix": "ac",

        # Raw class -> final class
        "mapping": {
            0: SLEEPING,
            7: USING_PHONE,
        },

        # Ambient
        # 0 Drowsy-Sleeping
        # 1 Eating-Drinking
        # 2 Focused-Thinking
        # 3 Looking down
        # 4 Looking upfront
        # 5 Raising Hand
        # 6 Using-Laptop-Writing
        # 7 Using-Phone
        "negative_classes": {
            1,
            2,
            3,
            4,
            5,
            6,
        },
    },
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10

RANDOM_SEED = 42

# ============================================================
# NEAR-DUPLICATE SETTINGS
# ============================================================

# IMPORTANT:
# V1.1 audit cho thấy distance 0 và 2 rất nhiều.
#
# Dùng <= 2 để GROUP positive.
#
# Không dùng <= 6 ngay vì có nguy cơ gom các ảnh lớp học
# khác nhau chỉ vì bố cục tương tự.
GROUP_PHASH_THRESHOLD = 2

# Negative train không được gần giống positive valid/test.
#
# Chỗ này dùng ngưỡng mạnh hơn một chút:
NEGATIVE_BLOCK_THRESHOLD = 4

MAX_NEGATIVE_IMAGES = 1000
MAX_NEGATIVE_PER_SOURCE = 600


# ============================================================
# WINDOWS PATH
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


def compute_phash(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return int(str(imagehash.phash(img)), 16)

    except Exception as e:
        print(
            f"[PHASH ERROR] {path}: {e}"
        )

        return None


def hamming_distance(a, b):
    return (a ^ b).bit_count()


def find_image(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


# ============================================================
# READ LABEL
# ============================================================

def read_raw_labels(label_path):
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

        # Bỏ bbox kỹ thuật không hợp lệ
        if w <= 0 or h <= 0:
            continue

        # Clip center / size về khoảng hợp lệ cơ bản.
        #
        # Không tự "sửa semantic" bbox.
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        labels.append(
            (cls, x, y, w, h)
        )

    return labels


def get_target_labels(raw_labels, mapping):
    target_labels = []

    for raw_cls, x, y, w, h in raw_labels:

        if raw_cls not in mapping:
            continue

        final_cls = mapping[raw_cls]

        target_labels.append(
            f"{final_cls} "
            f"{x:.8f} "
            f"{y:.8f} "
            f"{w:.8f} "
            f"{h:.8f}"
        )

    return target_labels


# ============================================================
# COLLECT POSITIVE
# ============================================================

def collect_positives(dataset):
    records = []

    root = dataset["path"]
    mapping = dataset["mapping"]

    print()
    print("=" * 70)
    print(
        f"COLLECT POSITIVE: "
        f"{dataset['name']}"
    )
    print("=" * 70)

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

        if (
            not images_dir.exists()
            or not labels_dir.exists()
        ):
            continue

        image_count = 0
        box_count = Counter()

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

            if not target_labels:
                continue

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:
                continue

            try:
                md5 = image_hash(image_path)
                phash = compute_phash(image_path)

            except Exception:
                continue

            if phash is None:
                continue

            for line in target_labels:
                cls = int(
                    line.split()[0]
                )

                box_count[cls] += 1

            records.append(
                {
                    "type": "positive",
                    "dataset": dataset["name"],
                    "prefix": dataset["prefix"],
                    "image_path": image_path,
                    "original_split": split,
                    "labels": target_labels,
                    "md5": md5,
                    "phash": phash,
                }
            )

            image_count += 1

        print(
            f"{split:<6} "
            f"images={image_count:<5} "
            f"sleeping={box_count[0]:<5} "
            f"phone={box_count[1]:<5}"
        )

    return records


# ============================================================
# EXACT DEDUPE POSITIVES
# ============================================================

def dedupe_positives(records):
    merged = {}

    for record in records:

        key = record["md5"]

        if key not in merged:

            copy = record.copy()

            copy["labels"] = set(
                record["labels"]
            )

            merged[key] = copy

        else:

            merged[key]["labels"].update(
                record["labels"]
            )

    result = []

    for record in merged.values():

        record["labels"] = sorted(
            record["labels"]
        )

        result.append(record)

    print()
    print("=" * 70)
    print("EXACT POSITIVE DEDUPE")
    print("=" * 70)

    print(
        f"Before : {len(records)}"
    )

    print(
        f"After  : {len(result)}"
    )

    print(
        f"Removed/Merged : "
        f"{len(records) - len(result)}"
    )

    return result


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

        self.rank = [0] * n

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):

            root_a, root_b = (
                root_b,
                root_a
            )

        self.parent[root_b] = root_a

        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):

            self.rank[root_a] += 1


# ============================================================
# GROUP NEAR-DUPLICATE POSITIVES
# ============================================================

def group_positive_near_duplicates(records):

    print()
    print("=" * 70)
    print("GROUP POSITIVE NEAR-DUPLICATES")
    print("=" * 70)

    print(
        f"Images    : {len(records)}"
    )

    print(
        f"Threshold : "
        f"pHash <= {GROUP_PHASH_THRESHOLD}"
    )

    n = len(records)

    uf = UnionFind(n)

    #
    # Bucket optimization.
    #
    # pHash là 64-bit.
    #
    # Với threshold <= 2, nếu hai hash có <=2 bit khác nhau,
    # chúng chắc chắn chia sẻ ít nhất một block giống nhau
    # khi chia thành 4 block 16-bit.
    #
    # Ta chỉ so sánh candidate có chung ít nhất một block.
    #

    buckets = defaultdict(list)

    for i, record in enumerate(records):

        h = record["phash"]

        blocks = [
            (h >> 48) & 0xFFFF,
            (h >> 32) & 0xFFFF,
            (h >> 16) & 0xFFFF,
            h & 0xFFFF,
        ]

        candidate_indices = set()

        for block_id, value in enumerate(
            blocks
        ):

            key = (
                block_id,
                value
            )

            candidate_indices.update(
                buckets[key]
            )

        for j in candidate_indices:

            distance = hamming_distance(
                h,
                records[j]["phash"]
            )

            if (
                distance
                <= GROUP_PHASH_THRESHOLD
            ):

                uf.union(i, j)

        for block_id, value in enumerate(
            blocks
        ):

            buckets[
                (block_id, value)
            ].append(i)

        if (
            (i + 1) % 500 == 0
            or i + 1 == n
        ):

            print(
                f"Processed "
                f"{i + 1}/{n}"
            )

    groups = defaultdict(list)

    for i, record in enumerate(records):

        root = uf.find(i)

        groups[root].append(
            record
        )

    group_list = list(
        groups.values()
    )

    sizes = [
        len(group)
        for group in group_list
    ]

    multi_groups = sum(
        1
        for size in sizes
        if size > 1
    )

    grouped_images = sum(
        size
        for size in sizes
        if size > 1
    )

    print()
    print(
        f"Total groups        : "
        f"{len(group_list)}"
    )

    print(
        f"Multi-image groups  : "
        f"{multi_groups}"
    )

    print(
        f"Images in groups>1  : "
        f"{grouped_images}"
    )

    print(
        f"Largest group       : "
        f"{max(sizes) if sizes else 0}"
    )

    return group_list


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

def group_aware_split(groups):

    print()
    print("=" * 70)
    print("GROUP-AWARE POSITIVE SPLIT")
    print("=" * 70)

    rng = random.Random(
        RANDOM_SEED
    )

    groups = groups.copy()

    #
    # Random trước, sau đó sort theo size.
    #
    # Nhờ random trước nên các group cùng size
    # không luôn có thứ tự giống raw dataset.
    #
    rng.shuffle(groups)

    groups.sort(
        key=len,
        reverse=True
    )

    total_images = sum(
        len(g)
        for g in groups
    )

    target_train = round(
        total_images * TRAIN_RATIO
    )

    target_valid = round(
        total_images * VALID_RATIO
    )

    target_test = (
        total_images
        - target_train
        - target_valid
    )

    targets = {
        "train": target_train,
        "valid": target_valid,
        "test": target_test,
    }

    splits = {
        "train": [],
        "valid": [],
        "test": [],
    }

    counts = {
        "train": 0,
        "valid": 0,
        "test": 0,
    }

    for group in groups:

        #
        # Chọn split đang thiếu nhiều nhất
        # theo tỷ lệ target.
        #

        def deficit_ratio(split):

            target = targets[split]

            if target <= 0:
                return -999

            return (
                target
                - counts[split]
            ) / target

        chosen = max(
            [
                "train",
                "valid",
                "test",
            ],
            key=deficit_ratio
        )

        splits[chosen].extend(
            group
        )

        counts[chosen] += len(
            group
        )

    print(
        f"Target train : {target_train}"
    )

    print(
        f"Actual train : "
        f"{len(splits['train'])}"
    )

    print()

    print(
        f"Target valid : {target_valid}"
    )

    print(
        f"Actual valid : "
        f"{len(splits['valid'])}"
    )

    print()

    print(
        f"Target test  : {target_test}"
    )

    print(
        f"Actual test  : "
        f"{len(splits['test'])}"
    )

    return splits


# ============================================================
# COLLECT NEGATIVE CANDIDATES
# ============================================================

def collect_negative_candidates(
    dataset,
    all_positive_md5,
):

    print()
    print("=" * 70)
    print(
        f"COLLECT NEGATIVE CANDIDATES: "
        f"{dataset['name']}"
    )
    print("=" * 70)

    root = dataset["path"]

    mapping = dataset[
        "mapping"
    ]

    negative_classes = dataset[
        "negative_classes"
    ]

    target_raw_classes = set(
        mapping.keys()
    )

    candidates = []

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

        if (
            not images_dir.exists()
            or not labels_dir.exists()
        ):
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

            # Nếu có target -> không phải negative.
            if (
                raw_classes
                & target_raw_classes
            ):
                continue

            # Chỉ lấy semantic negative đã chọn.
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

                md5 = image_hash(
                    image_path
                )

                if (
                    md5
                    in all_positive_md5
                ):
                    continue

                phash = compute_phash(
                    image_path
                )

                if phash is None:
                    continue

            except Exception:
                continue

            candidates.append(
                {
                    "type": "negative",
                    "dataset": (
                        dataset["name"]
                    ),
                    "prefix": (
                        dataset["prefix"]
                    ),
                    "image_path": image_path,
                    "original_split": split,
                    "labels": [],
                    "md5": md5,
                    "phash": phash,
                }
            )

            count += 1

        print(
            f"{split:<6} "
            f"candidates={count}"
        )

    print(
        f"Total candidates: "
        f"{len(candidates)}"
    )

    return candidates


# ============================================================
# PHASH INDEX
# ============================================================

def build_phash_block_index(records):

    index = defaultdict(list)

    for record in records:

        h = record["phash"]

        blocks = [
            (h >> 48) & 0xFFFF,
            (h >> 32) & 0xFFFF,
            (h >> 16) & 0xFFFF,
            h & 0xFFFF,
        ]

        for block_id, value in enumerate(
            blocks
        ):

            index[
                (block_id, value)
            ].append(h)

    return index


def has_near_match(
    phash_value,
    index,
    threshold,
):

    blocks = [
        (phash_value >> 48) & 0xFFFF,
        (phash_value >> 32) & 0xFFFF,
        (phash_value >> 16) & 0xFFFF,
        phash_value & 0xFFFF,
    ]

    candidates = set()

    for block_id, value in enumerate(
        blocks
    ):

        candidates.update(
            index.get(
                (block_id, value),
                []
            )
        )

    for other_hash in candidates:

        if (
            hamming_distance(
                phash_value,
                other_hash
            )
            <= threshold
        ):

            return True

    return False


# ============================================================
# FILTER NEGATIVES
# ============================================================

def filter_negative_candidates(
    candidates,
    valid_records,
    test_records,
):

    print()
    print("=" * 70)
    print("FILTER NEGATIVES AGAINST VALID/TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Tất cả positive trong VALID + TEST cần được bảo vệ.
    # Negative train không được quá giống các ảnh này.
    # --------------------------------------------------------

    protected_records = (
        list(valid_records)
        + list(test_records)
    )

    protected_hashes = [
        record["phash"]
        for record in protected_records
    ]

    print(
        f"Protected valid/test positives : "
        f"{len(protected_hashes)}"
    )

    print(
        f"Blocking threshold             : "
        f"pHash <= {NEGATIVE_BLOCK_THRESHOLD}"
    )

    safe = []

    blocked = 0
    blocked_exact = 0
    seen_md5 = set()

    for i, record in enumerate(
        candidates,
        start=1
    ):

        # ----------------------------------------------
        # Exact duplicate giữa negative candidates
        # ----------------------------------------------

        if record["md5"] in seen_md5:
            continue

        seen_md5.add(
            record["md5"]
        )

        current_hash = record["phash"]

        # ----------------------------------------------
        # DIRECT pHash CHECK
        #
        # Không dùng block-index ở đây vì threshold=4
        # có thể làm candidate bị bỏ sót.
        # ----------------------------------------------

        is_blocked = False

        for protected_hash in protected_hashes:

            distance = hamming_distance(
                current_hash,
                protected_hash
            )

            if (
                distance
                <= NEGATIVE_BLOCK_THRESHOLD
            ):
                is_blocked = True

                if distance == 0:
                    blocked_exact += 1

                break

        if is_blocked:
            blocked += 1
        else:
            safe.append(record)

        if (
            i % 500 == 0
            or i == len(candidates)
        ):

            print(
                f"Checked "
                f"{i}/{len(candidates)}"
            )

    print()
    print(
        f"Input candidates : "
        f"{len(candidates)}"
    )

    print(
        f"Blocked          : "
        f"{blocked}"
    )

    print(
        f"Blocked dist=0   : "
        f"{blocked_exact}"
    )

    print(
        f"Safe candidates  : "
        f"{len(safe)}"
    )

    return safe


# ============================================================
# SELECT NEGATIVES
# ============================================================

def select_negatives(
    candidates_by_source,
):

    rng = random.Random(
        RANDOM_SEED + 1
    )

    source_records = {}

    for source, records in (
        candidates_by_source.items()
    ):

        # Exact dedupe trong source.
        unique = {}

        for record in records:

            if (
                record["md5"]
                not in unique
            ):

                unique[
                    record["md5"]
                ] = record

        records = list(
            unique.values()
        )

        rng.shuffle(records)

        source_records[
            source
        ] = records

    selected = []

    selected_md5 = set()

    source_count = Counter()

    sources = list(
        source_records.keys()
    )

    # Round-robin để cân bằng source.
    positions = {
        source: 0
        for source in sources
    }

    while (
        len(selected)
        < MAX_NEGATIVE_IMAGES
    ):

        added_this_round = False

        for source in sources:

            if (
                len(selected)
                >= MAX_NEGATIVE_IMAGES
            ):
                break

            if (
                source_count[source]
                >= MAX_NEGATIVE_PER_SOURCE
            ):
                continue

            records = (
                source_records[source]
            )

            while (
                positions[source]
                < len(records)
            ):

                record = records[
                    positions[source]
                ]

                positions[source] += 1

                if (
                    record["md5"]
                    in selected_md5
                ):
                    continue

                selected.append(
                    record
                )

                selected_md5.add(
                    record["md5"]
                )

                source_count[
                    source
                ] += 1

                added_this_round = True

                break

        if not added_this_round:
            break

    rng.shuffle(selected)

    print()
    print("=" * 70)
    print("SELECTED NEGATIVES")
    print("=" * 70)

    print(
        f"Total: {len(selected)}"
    )

    for source, count in (
        source_count.items()
    ):

        print(
            f"{source:<25}: "
            f"{count}"
        )

    return selected


# ============================================================
# PREPARE OUTPUT
# ============================================================

def prepare_output():

    if OUTPUT_DIR.exists():

        print()
        print(
            f"Removing old V1.2: "
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


# ============================================================
# COPY
# ============================================================

def copy_record(
    record,
    split,
    index,
):

    image_dir = (
        OUTPUT_DIR
        / split
        / "images"
    )

    label_dir = (
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

    type_name = (
        "neg"
        if record["type"] == "negative"
        else "pos"
    )

    stem = (
        f"{record['prefix']}_"
        f"{type_name}_"
        f"{split}_"
        f"{index:06d}"
    )

    dst_image = (
        image_dir
        / f"{stem}{extension}"
    )

    dst_label = (
        label_dir
        / f"{stem}.txt"
    )

    try:

        shutil.copy2(
            windows_long_path(
                source_image
            ),
            dst_image
        )

    except Exception as e:

        print(
            f"[COPY ERROR] "
            f"{source_image}: {e}"
        )

        return False

    try:

        labels = record[
            "labels"
        ]

        if labels:

            dst_label.write_text(
                "\n".join(labels)
                + "\n",
                encoding="utf-8"
            )

        else:

            dst_label.write_text(
                "",
                encoding="utf-8"
            )

    except Exception as e:

        print(
            f"[LABEL ERROR] "
            f"{dst_label}: {e}"
        )

        try:
            dst_image.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return False

    return True


# ============================================================
# WRITE DATASET
# ============================================================

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

        if split == "train":
            records.extend(
                negatives
            )

        rng = random.Random(
            RANDOM_SEED + 100
        )

        rng.shuffle(records)

        positive_count = 0
        negative_count = 0
        boxes = Counter()
        written = 0

        for i, record in enumerate(
            records
        ):

            if not copy_record(
                record,
                split,
                i,
            ):
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

def verify():

    print()
    print("=" * 70)
    print("VERIFY V1.2")
    print("=" * 70)

    total_images = 0
    total_labels = 0
    total_empty = 0

    for split in [
        "train",
        "valid",
        "test",
    ]:

        image_dir = (
            OUTPUT_DIR
            / split
            / "images"
        )

        label_dir = (
            OUTPUT_DIR
            / split
            / "labels"
        )

        images = [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        labels = list(
            label_dir.glob(
                "*.txt"
            )
        )

        empty = 0
        boxes = Counter()

        for label_path in labels:

            content = (
                label_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
                .strip()
            )

            if not content:

                empty += 1
                continue

            for line in (
                content.splitlines()
            ):

                parts = (
                    line.split()
                )

                if len(parts) < 5:
                    continue

                cls = int(
                    float(parts[0])
                )

                boxes[cls] += 1

        print()
        print(split.upper())

        print(
            f"Images          : "
            f"{len(images)}"
        )

        print(
            f"Labels          : "
            f"{len(labels)}"
        )

        print(
            f"Empty negatives : "
            f"{empty}"
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

        total_empty += empty

    print()
    print("-" * 70)

    print(
        f"TOTAL IMAGES      : "
        f"{total_images}"
    )

    print(
        f"TOTAL LABEL FILES : "
        f"{total_labels}"
    )

    print(
        f"TOTAL NEGATIVES   : "
        f"{total_empty}"
    )

    if total_images == total_labels:

        print(
            "[OK] Image/label counts match."
        )

    else:

        print(
            "[ERROR] Image/label mismatch!"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILD BEHAVIOR V1.2 "
        "- LEAKAGE-AWARE SPLIT"
    )
    print("=" * 70)

    print()
    print(
        f"Positive group threshold : "
        f"{GROUP_PHASH_THRESHOLD}"
    )

    print(
        f"Negative block threshold : "
        f"{NEGATIVE_BLOCK_THRESHOLD}"
    )

    # ========================================================
    # 1. POSITIVES
    # ========================================================

    all_positive = []

    for dataset in DATASETS:

        records = collect_positives(
            dataset
        )

        all_positive.extend(
            records
        )

    print()
    print(
        f"Collected positives: "
        f"{len(all_positive)}"
    )

    # ========================================================
    # 2. EXACT DEDUPE
    # ========================================================

    positive_records = (
        dedupe_positives(
            all_positive
        )
    )

    all_positive_md5 = {
        record["md5"]
        for record
        in positive_records
    }

    # ========================================================
    # 3. GROUP NEAR DUPLICATES
    # ========================================================

    groups = (
        group_positive_near_duplicates(
            positive_records
        )
    )

    # ========================================================
    # 4. GROUP-AWARE SPLIT
    # ========================================================

    positive_splits = (
        group_aware_split(
            groups
        )
    )

    # ========================================================
    # 5. NEGATIVE CANDIDATES
    # ========================================================

    raw_negative_candidates = {}

    for dataset in DATASETS:

        candidates = (
            collect_negative_candidates(
                dataset,
                all_positive_md5,
            )
        )

        raw_negative_candidates[
            dataset["name"]
        ] = candidates

    # ========================================================
    # 6. FILTER NEGATIVE AGAINST VAL/TEST
    # ========================================================

    safe_candidates = {}

    for source, candidates in (
        raw_negative_candidates.items()
    ):

        print()
        print(
            f"SOURCE: {source}"
        )

        safe_candidates[
            source
        ] = (
            filter_negative_candidates(
                candidates,
                positive_splits[
                    "valid"
                ],
                positive_splits[
                    "test"
                ],
            )
        )

    # ========================================================
    # 7. SELECT 1000 NEGATIVES
    # ========================================================

    negatives = select_negatives(
        safe_candidates
    )

    # ========================================================
    # 8. WRITE
    # ========================================================

    prepare_output()

    stats = write_dataset(
        positive_splits,
        negatives,
    )

    write_yaml()

    print()
    print("=" * 70)
    print("WRITE SUMMARY")
    print("=" * 70)

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

    verify()

    print()
    print("=" * 70)
    print("BUILD V1.2 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Dataset:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "IMPORTANT: DO NOT TRAIN YET."
    )

    print(
        "Run leakage check on V1.2 first."
    )


if __name__ == "__main__":
    main()