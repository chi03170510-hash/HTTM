from pathlib import Path
from PIL import Image
import imagehash
from collections import Counter
import time


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"E:\TTT\Project\HTTM")

DATASET_DIR = (
    PROJECT_DIR
    / "ai"
    / "behavior_detection"
    / "datasets"
    / "behavior_v1_2"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# pHash Hamming distance:
# 0 = gần như giống hệt
# càng lớn = càng khác
#
# <= 4: rất giống
# <= 6: khá giống
# <= 8: có khả năng là frame gần nhau
THRESHOLD = 6

# Số cặp ví dụ tối đa in ra terminal
MAX_EXAMPLES = 30


# ============================================================
# GET IMAGES
# ============================================================

def get_images(split):

    image_dir = (
        DATASET_DIR
        / split
        / "images"
    )

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    return sorted(images)


# ============================================================
# COMPUTE PHASH
# ============================================================

def compute_hashes(images, split):

    results = []

    print()
    print(
        f"Computing pHash: {split}"
    )

    for i, path in enumerate(
        images,
        start=1
    ):

        try:

            with Image.open(path) as img:

                img = img.convert("RGB")

                h = imagehash.phash(
                    img
                )

                results.append(
                    (path, h)
                )

        except Exception as e:

            print(
                f"[READ ERROR] "
                f"{path.name}: {e}"
            )

        if (
            i % 500 == 0
            or i == len(images)
        ):

            print(
                f"  {i}/{len(images)}"
            )

    return results


# ============================================================
# HASH -> INTEGER
# ============================================================

def hash_to_int(h):

    return int(
        str(h),
        16
    )


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(a, b):

    return (
        a ^ b
    ).bit_count()


# ============================================================
# COMPARE SPLITS
# ============================================================

def compare_splits(
    name_a,
    hashes_a,
    name_b,
    hashes_b,
):

    print()
    print("=" * 70)

    print(
        f"CHECKING "
        f"{name_a.upper()} "
        f"<-> "
        f"{name_b.upper()}"
    )

    print("=" * 70)

    # Convert hashes thành integer
    a_data = [
        (
            path,
            hash_to_int(h)
        )
        for path, h
        in hashes_a
    ]

    b_data = [
        (
            path,
            hash_to_int(h)
        )
        for path, h
        in hashes_b
    ]

    matches = []

    distance_counter = Counter()

    start = time.time()

    # Với dataset hiện tại:
    #
    # train ~5256
    # valid ~532
    # test  ~532
    #
    # brute force vẫn có thể chạy được.
    for i, (
        path_a,
        hash_a
    ) in enumerate(a_data):

        for (
            path_b,
            hash_b
        ) in b_data:

            distance = (
                hamming_distance(
                    hash_a,
                    hash_b
                )
            )

            if distance <= THRESHOLD:

                matches.append(
                    (
                        distance,
                        path_a,
                        path_b,
                    )
                )

                distance_counter[
                    distance
                ] += 1

        if (
            (i + 1) % 500 == 0
            or i + 1 == len(a_data)
        ):

            print(
                f"  checked "
                f"{i + 1}/"
                f"{len(a_data)} "
                f"{name_a} images"
            )

    elapsed = (
        time.time()
        - start
    )

    matches.sort(
        key=lambda x: x[0]
    )

    print()
    print(
        f"Near-duplicate pairs : "
        f"{len(matches)}"
    )

    print(
        f"Threshold            : "
        f"pHash <= {THRESHOLD}"
    )

    print(
        f"Time                 : "
        f"{elapsed:.2f}s"
    )

    if distance_counter:

        print()
        print("DISTANCE DISTRIBUTION")

        for distance in sorted(
            distance_counter
        ):

            print(
                f"distance {distance}: "
                f"{distance_counter[distance]}"
            )

    else:

        print()
        print(
            "No near-duplicate "
            "pairs detected."
        )

    # ----------------------------------------
    # UNIQUE AFFECTED IMAGES
    # ----------------------------------------

    affected_a = {
        str(a)
        for _, a, _
        in matches
    }

    affected_b = {
        str(b)
        for _, _, b
        in matches
    }

    print()
    print(
        f"Affected {name_a} images : "
        f"{len(affected_a)}"
    )

    print(
        f"Affected {name_b} images : "
        f"{len(affected_b)}"
    )

    # ----------------------------------------
    # EXAMPLES
    # ----------------------------------------

    if matches:

        print()
        print(
            f"TOP {min(MAX_EXAMPLES, len(matches))} "
            f"MATCHES"
        )

        print("-" * 70)

        for (
            distance,
            path_a,
            path_b,
        ) in matches[:MAX_EXAMPLES]:

            print()
            print(
                f"distance={distance}"
            )

            print(
                f"  {name_a}: "
                f"{path_a.name}"
            )

            print(
                f"  {name_b}: "
                f"{path_b.name}"
            )

    return {
        "pair": (
            f"{name_a}<->{name_b}"
        ),
        "matches": len(matches),
        "affected_a": len(
            affected_a
        ),
        "affected_b": len(
            affected_b
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BEHAVIOR V1.2 - "
        "TRAIN/VALID/TEST LEAKAGE CHECK"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset: {DATASET_DIR}"
    )

    print(
        f"pHash threshold: "
        f"{THRESHOLD}"
    )

    # ========================================================
    # LOAD IMAGE LISTS
    # ========================================================

    splits = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:

        images = get_images(
            split
        )

        splits[split] = images

        print(
            f"{split:<6}: "
            f"{len(images)} images"
        )

    # ========================================================
    # COMPUTE HASHES
    # ========================================================

    hashes = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:

        hashes[split] = (
            compute_hashes(
                splits[split],
                split
            )
        )

    # ========================================================
    # CROSS-SPLIT CHECK
    # ========================================================

    results = []

    results.append(
        compare_splits(
            "train",
            hashes["train"],
            "valid",
            hashes["valid"],
        )
    )

    results.append(
        compare_splits(
            "train",
            hashes["train"],
            "test",
            hashes["test"],
        )
    )

    results.append(
        compare_splits(
            "valid",
            hashes["valid"],
            "test",
            hashes["test"],
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_matches = 0

    for result in results:

        print(
            f"{result['pair']:<20} "
            f"pairs={result['matches']:<6} "
            f"affected="
            f"{result['affected_a']}/"
            f"{result['affected_b']}"
        )

        total_matches += (
            result["matches"]
        )

    print()
    print(
        f"TOTAL CROSS-SPLIT "
        f"NEAR-DUPLICATE PAIRS: "
        f"{total_matches}"
    )

    print()

    if total_matches == 0:

        print(
            "[PASS] No cross-split "
            "near duplicates detected."
        )

        print(
            "Dataset is ready for "
            "the next training step."
        )

    else:

        print(
            "[REVIEW] Cross-split "
            "near duplicates detected."
        )

        print(
            "DO NOT delete anything yet."
        )

        print(
            "Review the counts/examples "
            "before changing the dataset."
        )


if __name__ == "__main__":
    main()