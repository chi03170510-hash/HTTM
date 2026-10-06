from pathlib import Path
import cv2
import random
import shutil


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"E:\TTT\Project\HTTM")

DATASET_DIR = (
    PROJECT_DIR
    / "ai"
    / "behavior_detection"
    / "datasets"
    / "behavior_v1_1"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "ai"
    / "behavior_detection"
    / "label_preview"
    / "behavior_v1_1"
)

CLASS_NAMES = {
    0: "sleeping",
    1: "using_phone",
}

# Số ảnh preview mỗi nhóm
NUM_SLEEPING = 30
NUM_PHONE = 30
NUM_MIXED = 20

RANDOM_SEED = 42

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]


# ============================================================
# HELPERS
# ============================================================

def find_image(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        path = images_dir / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def read_labels(label_path):
    labels = []

    if not label_path.exists():
        return labels

    lines = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            continue

        labels.append(
            (
                cls,
                x_center,
                y_center,
                width,
                height,
            )
        )

    return labels


def classify_image(labels):
    classes = {
        label[0]
        for label in labels
    }

    has_sleeping = 0 in classes
    has_phone = 1 in classes

    if has_sleeping and has_phone:
        return "mixed"

    if has_sleeping:
        return "sleeping"

    if has_phone:
        return "using_phone"

    return "none"


# ============================================================
# DRAW
# ============================================================

def draw_boxes(image_path, labels):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    height, width = image.shape[:2]

    for (
        cls,
        x_center,
        y_center,
        box_width,
        box_height,
    ) in labels:

        x1 = int(
            (x_center - box_width / 2)
            * width
        )

        y1 = int(
            (y_center - box_height / 2)
            * height
        )

        x2 = int(
            (x_center + box_width / 2)
            * width
        )

        y2 = int(
            (y_center + box_height / 2)
            * height
        )

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        class_name = CLASS_NAMES.get(
            cls,
            f"class_{cls}"
        )

        # Không cần quan tâm màu cụ thể,
        # chỉ cần 2 class nhìn khác nhau.
        if cls == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label_text = class_name

        (text_width, text_height), _ = (
            cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
        )

        text_y = max(
            y1 - 5,
            text_height + 5
        )

        cv2.rectangle(
            image,
            (
                x1,
                text_y - text_height - 5
            ),
            (
                x1 + text_width + 5,
                text_y + 3
            ),
            color,
            -1
        )

        cv2.putText(
            image,
            label_text,
            (
                x1 + 2,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return image


# ============================================================
# COLLECT
# ============================================================

def collect_candidates():

    groups = {
        "sleeping": [],
        "using_phone": [],
        "mixed": [],
    }

    # Kiểm tra cả train/valid/test
    for split in [
        "train",
        "valid",
        "test",
    ]:

        images_dir = (
            DATASET_DIR
            / split
            / "images"
        )

        labels_dir = (
            DATASET_DIR
            / split
            / "labels"
        )

        for label_path in labels_dir.glob(
            "*.txt"
        ):

            labels = read_labels(
                label_path
            )

            group = classify_image(
                labels
            )

            if group not in groups:
                continue

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:
                continue

            groups[group].append(
                {
                    "split": split,
                    "image": image_path,
                    "label": label_path,
                    "labels": labels,
                }
            )

    return groups


# ============================================================
# SAMPLE
# ============================================================

def sample_group(items, count):

    if len(items) <= count:
        return items

    return random.sample(
        items,
        count
    )


# ============================================================
# SAVE PREVIEW
# ============================================================

def save_group(
    group_name,
    records
):

    output_group = (
        OUTPUT_DIR
        / group_name
    )

    output_group.mkdir(
        parents=True,
        exist_ok=True
    )

    success = 0

    for index, record in enumerate(
        records,
        start=1
    ):

        preview = draw_boxes(
            record["image"],
            record["labels"]
        )

        if preview is None:
            print(
                f"[READ ERROR] "
                f"{record['image']}"
            )
            continue

        output_name = (
            f"{index:03d}_"
            f"{record['split']}_"
            f"{record['image'].stem}.jpg"
        )

        output_path = (
            output_group
            / output_name
        )

        ok = cv2.imwrite(
            str(output_path),
            preview
        )

        if ok:
            success += 1

    print(
        f"{group_name}: "
        f"{success} preview images"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PREVIEW BEHAVIOR DATASET V1.1")
    print("=" * 60)

    if not DATASET_DIR.exists():

        print(
            f"ERROR: Dataset not found:\n"
            f"{DATASET_DIR}"
        )

        return

    # Xóa preview cũ
    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    random.seed(
        RANDOM_SEED
    )

    groups = collect_candidates()

    print()
    print("AVAILABLE IMAGES")
    print("-" * 60)

    print(
        f"Sleeping only : "
        f"{len(groups['sleeping'])}"
    )

    print(
        f"Phone only    : "
        f"{len(groups['using_phone'])}"
    )

    print(
        f"Mixed         : "
        f"{len(groups['mixed'])}"
    )

    sleeping_sample = sample_group(
        groups["sleeping"],
        NUM_SLEEPING
    )

    phone_sample = sample_group(
        groups["using_phone"],
        NUM_PHONE
    )

    mixed_sample = sample_group(
        groups["mixed"],
        NUM_MIXED
    )

    print()
    print("CREATING PREVIEWS")
    print("-" * 60)

    save_group(
        "sleeping",
        sleeping_sample
    )

    save_group(
        "using_phone",
        phone_sample
    )

    save_group(
        "mixed",
        mixed_sample
    )

    print()
    print("=" * 60)
    print("PREVIEW COMPLETE")
    print("=" * 60)

    print()
    print(
        "Preview folder:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Check:"
    )

    print(
        "1. Box có đúng người không?"
    )

    print(
        "2. Sleeping có thực sự đang ngủ không?"
    )

    print(
        "3. Using_phone có thực sự đang sử dụng điện thoại không?"
    )

    print(
        "4. Có người vi phạm nhưng bị thiếu bbox không?"
    )

    print(
        "5. Có bbox quá lớn/quá nhỏ/sai vị trí không?"
    )


if __name__ == "__main__":
    main()