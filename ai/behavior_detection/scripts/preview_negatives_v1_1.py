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

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

OUTPUT_DIR = (
    PROJECT_DIR
    / "ai"
    / "behavior_detection"
    / "label_preview"
    / "behavior_v1_1_negatives"
)

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]

# Preview 50 ảnh mỗi nguồn
NUM_SB = 50
NUM_AC = 50

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def find_image(stem):
    for ext in IMAGE_EXTENSIONS:
        path = TRAIN_IMAGES / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def is_empty_label(label_path):
    try:
        content = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        return content == ""

    except Exception:
        return False


# ============================================================
# COLLECT NEGATIVES
# ============================================================

def collect_negatives():

    groups = {
        "student_behaviors": [],
        "ambient_classroom": [],
    }

    for label_path in TRAIN_LABELS.glob("*.txt"):

        # Negative phải có label rỗng
        if not is_empty_label(label_path):
            continue

        stem = label_path.stem

        image_path = find_image(stem)

        if image_path is None:
            continue

        if stem.startswith("sb_neg_"):

            groups[
                "student_behaviors"
            ].append(image_path)

        elif stem.startswith("ac_neg_"):

            groups[
                "ambient_classroom"
            ].append(image_path)

    return groups


# ============================================================
# SAVE PREVIEW
# ============================================================

def save_group(name, images, number):

    output = OUTPUT_DIR / name

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    random.shuffle(images)

    selected = images[:number]

    success = 0

    for index, image_path in enumerate(
        selected,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[READ ERROR] {image_path}"
            )
            continue

        # Ghi chú đây là negative/background
        cv2.putText(
            image,
            "NEGATIVE / BACKGROUND",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        output_name = (
            f"{index:03d}_"
            f"{image_path.name}"
        )

        output_path = (
            output / output_name
        )

        if cv2.imwrite(
            str(output_path),
            image
        ):
            success += 1

    print(
        f"{name}: "
        f"{success}/{len(selected)} "
        f"preview images"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PREVIEW HARD NEGATIVES V1.1")
    print("=" * 65)

    if not DATASET_DIR.exists():

        print(
            f"Dataset not found:\n"
            f"{DATASET_DIR}"
        )

        return

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

    groups = collect_negatives()

    print()
    print("NEGATIVE DATASET")
    print("-" * 65)

    print(
        f"Student Behaviors : "
        f"{len(groups['student_behaviors'])}"
    )

    print(
        f"Ambient Classroom : "
        f"{len(groups['ambient_classroom'])}"
    )

    print(
        f"Total             : "
        f"{sum(len(x) for x in groups.values())}"
    )

    print()
    print("CREATING PREVIEW")
    print("-" * 65)

    save_group(
        "student_behaviors",
        groups["student_behaviors"],
        NUM_SB
    )

    save_group(
        "ambient_classroom",
        groups["ambient_classroom"],
        NUM_AC
    )

    print()
    print("=" * 65)
    print("PREVIEW COMPLETE")
    print("=" * 65)

    print()
    print("Preview folder:")
    print(OUTPUT_DIR)

    print()
    print("REVIEW RULES:")
    print(
        "[GOOD] writing / reading / "
        "looking down / laptop / "
        "raising hand / normal sitting"
    )

    print(
        "[BAD] any clearly sleeping student"
    )

    print(
        "[BAD] any clearly using-phone student"
    )

    print()
    print(
        "IMPORTANT: negative images have "
        "empty YOLO labels by design."
    )


if __name__ == "__main__":
    main()