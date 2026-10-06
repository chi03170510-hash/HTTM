from pathlib import Path
from collections import Counter, defaultdict
import cv2
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
    / "label_review"
    / "behavior_v1_1"
)

CLASS_NAMES = {
    0: "sleeping",
    1: "using_phone",
}

IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
]

# ============================================================
# FLAG THRESHOLDS
# Chỉ dùng để FLAG, KHÔNG tự xóa
# ============================================================

# Diện tích bbox / diện tích ảnh
LARGE_AREA_RATIO = 0.55

# Width hoặc height bbox quá lớn so với ảnh
LARGE_WIDTH_RATIO = 0.85
LARGE_HEIGHT_RATIO = 0.95

# Bbox cực nhỏ
SMALL_AREA_RATIO = 0.001

# Aspect ratio quá bất thường
MAX_ASPECT_RATIO = 5.0

# Hai bbox cùng class overlap quá mạnh
HIGH_IOU = 0.80


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

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception:
        return labels

    for line_no, line in enumerate(lines, start=1):

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        labels.append({
            "line_no": line_no,
            "class": cls,
            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h,
        })

    return labels


def yolo_to_xyxy(label):
    xc = label["xc"]
    yc = label["yc"]
    w = label["w"]
    h = label["h"]

    return (
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2,
    )


def calculate_iou(a, b):
    ax1, ay1, ax2, ay2 = yolo_to_xyxy(a)
    bx1, by1, bx2, by2 = yolo_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0

    return intersection / union


# ============================================================
# AUDIT ONE BOX
# ============================================================

def audit_box(label):
    reasons = []

    cls = label["class"]
    xc = label["xc"]
    yc = label["yc"]
    w = label["w"]
    h = label["h"]

    # Class không hợp lệ
    if cls not in CLASS_NAMES:
        reasons.append("invalid_class")

    # Giá trị YOLO cơ bản
    if w <= 0 or h <= 0:
        reasons.append("invalid_size")

    if xc < 0 or xc > 1 or yc < 0 or yc > 1:
        reasons.append("center_out_of_range")

    if w > 1 or h > 1:
        reasons.append("size_out_of_range")

    # Box vượt ra ngoài ảnh
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        reasons.append("outside_image")

    area = w * h

    if area >= LARGE_AREA_RATIO:
        reasons.append("huge_area")

    if w >= LARGE_WIDTH_RATIO:
        reasons.append("huge_width")

    if h >= LARGE_HEIGHT_RATIO:
        reasons.append("huge_height")

    if area <= SMALL_AREA_RATIO:
        reasons.append("tiny_box")

    if w > 0 and h > 0:
        aspect = max(w / h, h / w)

        if aspect >= MAX_ASPECT_RATIO:
            reasons.append("extreme_aspect")

    return reasons


# ============================================================
# DRAW REVIEW IMAGE
# ============================================================

def draw_review(image_path, labels, flagged_indices, image_reasons):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    for index, label in enumerate(labels):

        cls = label["class"]

        x1n, y1n, x2n, y2n = yolo_to_xyxy(label)

        x1 = int(x1n * width)
        y1 = int(y1n * height)
        x2 = int(x2n * width)
        y2 = int(y2n * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Flagged box = đỏ
        if index in flagged_indices:
            color = (0, 0, 255)
            thickness = 4
        else:
            color = (0, 255, 0)
            thickness = 2

        class_name = CLASS_NAMES.get(
            cls,
            f"class_{cls}"
        )

        text = f"{index}: {class_name}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    # Ghi lý do ở trên ảnh
    reason_text = ", ".join(sorted(image_reasons))

    cv2.putText(
        image,
        reason_text[:120],
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return image


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 65)
    print("AUDIT BEHAVIOR DATASET V1.1")
    print("=" * 65)

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    reason_counter = Counter()
    class_flag_counter = Counter()

    total_images = 0
    total_boxes = 0

    flagged_images = 0
    flagged_boxes = 0

    source_counter = Counter()
    source_flagged = Counter()

    # Dùng để xuất report text
    report_rows = []

    for split in ["train", "valid", "test"]:

        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        split_output = OUTPUT_DIR / split
        split_output.mkdir(
            parents=True,
            exist_ok=True
        )

        for label_path in labels_dir.glob("*.txt"):

            total_images += 1

            image_path = find_image(
                images_dir,
                label_path.stem
            )

            if image_path is None:
                reason_counter["missing_image"] += 1
                continue

            labels = read_labels(label_path)

            total_boxes += len(labels)

            # Filename bắt đầu sb_ hoặc ac_
            if label_path.stem.startswith("sb_"):
                source = "student_behaviors"
            elif label_path.stem.startswith("ac_"):
                source = "ambient_classroom"
            else:
                source = "unknown"

            source_counter[source] += 1

            flagged_indices = set()
            image_reasons = set()

            # -----------------------------------------------
            # Kiểm tra từng bbox
            # -----------------------------------------------

            for index, label in enumerate(labels):

                reasons = audit_box(label)

                if reasons:
                    flagged_indices.add(index)

                    for reason in reasons:
                        reason_counter[reason] += 1
                        image_reasons.add(reason)

                    class_flag_counter[
                        label["class"]
                    ] += 1

            # -----------------------------------------------
            # Kiểm tra bbox overlap cùng class
            # -----------------------------------------------

            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):

                    # Chỉ flag overlap nếu cùng class
                    if (
                        labels[i]["class"]
                        != labels[j]["class"]
                    ):
                        continue

                    iou = calculate_iou(
                        labels[i],
                        labels[j]
                    )

                    if iou >= HIGH_IOU:

                        flagged_indices.add(i)
                        flagged_indices.add(j)

                        image_reasons.add(
                            "high_same_class_iou"
                        )

                        reason_counter[
                            "high_same_class_iou"
                        ] += 1

            # -----------------------------------------------
            # Nếu ảnh bị flag
            # -----------------------------------------------

            if flagged_indices:

                flagged_images += 1
                flagged_boxes += len(
                    flagged_indices
                )

                source_flagged[source] += 1

                preview = draw_review(
                    image_path,
                    labels,
                    flagged_indices,
                    image_reasons
                )

                if preview is not None:

                    output_name = (
                        f"{source}_"
                        f"{label_path.stem}.jpg"
                    )

                    output_path = (
                        split_output
                        / output_name
                    )

                    cv2.imwrite(
                        str(output_path),
                        preview
                    )

                report_rows.append(
                    {
                        "split": split,
                        "source": source,
                        "file": label_path.name,
                        "flagged_boxes": len(
                            flagged_indices
                        ),
                        "reasons": ", ".join(
                            sorted(image_reasons)
                        ),
                    }
                )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = OUTPUT_DIR / "audit_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BEHAVIOR V1.1 BBOX AUDIT REPORT\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Total images: {total_images}\n"
        )

        f.write(
            f"Total boxes: {total_boxes}\n"
        )

        f.write(
            f"Flagged images: {flagged_images}\n"
        )

        f.write(
            f"Flagged boxes: {flagged_boxes}\n\n"
        )

        f.write("REASONS\n")
        f.write("-" * 70 + "\n")

        for reason, count in (
            reason_counter.most_common()
        ):
            f.write(
                f"{reason}: {count}\n"
            )

        f.write("\nFLAGGED BY CLASS\n")
        f.write("-" * 70 + "\n")

        for cls, count in sorted(
            class_flag_counter.items()
        ):
            f.write(
                f"{CLASS_NAMES.get(cls, cls)}: "
                f"{count}\n"
            )

        f.write("\nSOURCE SUMMARY\n")
        f.write("-" * 70 + "\n")

        for source in source_counter:

            total = source_counter[source]
            flagged = source_flagged[source]

            percent = (
                flagged / total * 100
                if total > 0
                else 0
            )

            f.write(
                f"{source}: "
                f"{flagged}/{total} "
                f"flagged images "
                f"({percent:.2f}%)\n"
            )

        f.write("\nFLAGGED FILES\n")
        f.write("-" * 70 + "\n")

        for row in report_rows:

            f.write(
                f"[{row['split']}] "
                f"[{row['source']}] "
                f"{row['file']} | "
                f"boxes={row['flagged_boxes']} | "
                f"{row['reasons']}\n"
            )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("DATASET")
    print("-" * 65)

    print(
        f"Total images : {total_images}"
    )

    print(
        f"Total boxes  : {total_boxes}"
    )

    print()
    print("FLAGS")
    print("-" * 65)

    print(
        f"Flagged images: {flagged_images}"
    )

    print(
        f"Flagged boxes : {flagged_boxes}"
    )

    if total_images > 0:
        print(
            f"Flagged image rate: "
            f"{flagged_images / total_images * 100:.2f}%"
        )

    print()
    print("REASONS")
    print("-" * 65)

    for reason, count in (
        reason_counter.most_common()
    ):
        print(
            f"{reason:<25} {count}"
        )

    print()
    print("FLAGGED BY CLASS")
    print("-" * 65)

    for cls, count in sorted(
        class_flag_counter.items()
    ):

        print(
            f"{CLASS_NAMES.get(cls, cls):<15} "
            f"{count}"
        )

    print()
    print("SOURCE SUMMARY")
    print("-" * 65)

    for source in source_counter:

        total = source_counter[source]
        flagged = source_flagged[source]

        percentage = (
            flagged / total * 100
            if total
            else 0
        )

        print(
            f"{source:<25} "
            f"{flagged}/{total} "
            f"({percentage:.2f}%)"
        )

    print()
    print("=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)

    print()
    print("Review folder:")
    print(OUTPUT_DIR)

    print()
    print("Report:")
    print(report_path)

    print()
    print(
        "IMPORTANT: "
        "No labels were deleted or modified."
    )


if __name__ == "__main__":
    main()