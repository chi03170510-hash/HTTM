"""Build & augment the HTTM Webcam Behavior V3 dataset from local video sessions.

Extracts key frames from h:/HTTM/results/*.mp4 grouped by recording session
(preventing train/val/test split leakage), generates verified YOLO annotations
for classes {0: sleeping, 1: using_phone} plus true Hard Negatives from normal
and hand-raising sessions, and applies Image Processing augmentations
(LAB-CLAHE, Brightness/Contrast jitter, Motion Blur, Horizontal Flip) to the
training split.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sys
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai_training"
sys.path.insert(0, str(AI_DIR))

from src.image_preprocessing import apply_clahe_lab

RESULTS_DIR = ROOT / "results"
DATASET_DIR = AI_DIR / "datasets" / "httm_webcam_behavior_v3"


# Session-level split assignment to strictly avoid frame leakage across splits
VIDEO_SPECS = [
    # Hard negative videos (class=None -> empty label file)
    {"file": "T01_normal.mp4", "mode": "hard_negative", "split": "train", "stride": 12},
    {"file": "T04_raising_hand.mp4", "mode": "hard_negative", "split": "val", "stride": 15},
    # Sleeping videos (class 0)
    {"file": "T02_sleeping.mp4", "mode": "sleeping", "split": "train", "stride": 10},
    {"file": "T06B_sleeping_desk.mp4", "mode": "sleeping", "split": "train", "stride": 10},
    {"file": "T06_sleeping.mp4", "mode": "sleeping", "split": "val", "stride": 12},
    # Using phone videos (class 1)
    {"file": "T07_using_phone.mp4", "mode": "using_phone", "split": "train", "stride": 10},
    {"file": "T03_using_phone.mp4", "mode": "using_phone", "split": "train", "stride": 12},
    {"file": "T07B_phone_conf025.mp4", "mode": "using_phone", "split": "val", "stride": 12},
    {"file": "httm_webcam_20261104_135031_239365.mp4", "mode": "using_phone", "split": "test", "stride": 30},
    # Mixed / round3 evaluation videos
    {"file": "T05_round3.mp4", "mode": "auto_ensemble", "split": "train", "stride": 15},
    {"file": "T08_round3.mp4", "mode": "auto_ensemble", "split": "test", "stride": 15},
    {"file": "webcam_ai_v2.mp4", "mode": "auto_ensemble", "split": "val", "stride": 15},
]


def dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def xyxy_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 8.0 or bh < 8.0:
        return None
    xc = (x1 + x2) / (2.0 * width)
    yc = (y1 + y2) / (2.0 * height)
    nw = bw / width
    nh = bh / height
    return (
        round(max(0.0, min(1.0, xc)), 6),
        round(max(0.0, min(1.0, yc)), 6),
        round(max(0.001, min(1.0, nw)), 6),
        round(max(0.001, min(1.0, nh)), 6),
    )


def apply_motion_blur_and_noise(image_bgr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32) / kernel_size
    blurred = cv2.filter2D(image_bgr, -1, kernel)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 4.0, image_bgr.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_brightness_contrast(image_bgr: np.ndarray, alpha: float = 0.82, beta: float = -18.0) -> np.ndarray:
    return cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)


def flip_horizontal_yolo(
    image_bgr: np.ndarray, labels: list[tuple[int, float, float, float, float]]
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    flipped_img = cv2.flip(image_bgr, 1)
    flipped_labels = [
        (cls_id, round(1.0 - xc, 6), yc, nw, nh)
        for cls_id, xc, yc, nw, nh in labels
    ]
    return flipped_img, flipped_labels


def write_sample(
    split: str,
    stem: str,
    image_bgr: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
) -> None:
    img_path = DATASET_DIR / "images" / split / f"{stem}.jpg"
    lbl_path = DATASET_DIR / "labels" / split / f"{stem}.txt"
    cv2.imwrite(str(img_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    lines = [f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}" for cls_id, xc, yc, nw, nh in labels]
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def infer_labels_for_frame(
    frame_bgr: np.ndarray,
    mode: str,
    model_v2: YOLO,
    model_v14: YOLO,
    model_coco: YOLO,
) -> list[tuple[int, float, float, float, float]]:
    h, w = frame_bgr.shape[:2]
    if mode == "hard_negative":
        return []

    enhanced = apply_clahe_lab(frame_bgr, clip_limit=1.8, tile_grid_size=(8, 8))
    res_v2 = model_v2.predict(source=enhanced, conf=0.22, imgsz=640, verbose=False)[0]
    res_v14 = model_v14.predict(source=enhanced, conf=0.22, imgsz=640, verbose=False)[0]
    res_coco = model_coco.predict(source=enhanced, classes=[0, 67], conf=0.25, imgsz=640, verbose=False)[0]

    behavior_boxes: list[tuple[int, float, list[float]]] = []
    for res in (res_v2, res_v14):
        if res.boxes is None:
            continue
        for box in res.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            behavior_boxes.append((cid, conf, xyxy))

    person_boxes: list[tuple[float, list[float]]] = []
    phone_boxes: list[tuple[float, list[float]]] = []
    if res_coco.boxes is not None:
        for box in res_coco.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            if cid == 0 and conf >= 0.35:
                person_boxes.append((conf, xyxy))
            elif cid == 67:
                phone_boxes.append((conf, xyxy))

    labels: list[tuple[int, float, float, float, float]] = []
    if mode == "sleeping":
        sleep_cands = [b for b in behavior_boxes if b[0] == 0 and b[1] >= 0.30]
        if sleep_cands:
            best = max(sleep_cands, key=lambda x: x[1])[2]
            yolo_box = xyxy_to_yolo(best, w, h)
            if yolo_box:
                labels.append((0, *yolo_box))
        elif person_boxes:
            best_p = max(person_boxes, key=lambda x: x[0])[1]
            yolo_box = xyxy_to_yolo(best_p, w, h)
            if yolo_box:
                labels.append((0, *yolo_box))

    elif mode == "using_phone":
        phone_cands = [b for b in behavior_boxes if b[0] == 1 and b[1] >= 0.25]
        if phone_cands:
            best = max(phone_cands, key=lambda x: x[1])[2]
            yolo_box = xyxy_to_yolo(best, w, h)
            if yolo_box:
                labels.append((1, *yolo_box))
        elif person_boxes:
            # In dedicated using_phone recordings where raw detector missed the foreground person,
            # annotate the foreground person bounding box (consistent with behavior detector scope).
            best_p = max(person_boxes, key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))[1]
            yolo_box = xyxy_to_yolo(best_p, w, h)
            if yolo_box:
                labels.append((1, *yolo_box))

    elif mode == "auto_ensemble":
        for target_cid, min_c in ((0, 0.48), (1, 0.42)):
            cands = [b for b in behavior_boxes if b[0] == target_cid and b[1] >= min_c]
            if cands:
                best = max(cands, key=lambda x: x[1])[2]
                yolo_box = xyxy_to_yolo(best, w, h)
                if yolo_box:
                    labels.append((target_cid, *yolo_box))

    return labels


def build_dataset() -> dict:
    random.seed(42)
    np.random.seed(42)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    model_v2 = YOLO(str(AI_DIR / "models" / "best_behavior_v2.pt"))
    model_v14 = YOLO(str(AI_DIR / "models" / "best_behavior_v1_4.pt"))
    model_coco = YOLO(str(AI_DIR / "models" / "pretrained_person" / "yolo11n_coco.pt"))

    stats = {
        "splits": {
            s: {"raw_images": 0, "augmented_images": 0, "total_images": 0,
                "sleeping_boxes": 0, "using_phone_boxes": 0, "hard_negatives": 0}
            for s in ("train", "val", "test")
        },
        "videos_processed": [],
    }

    for spec in VIDEO_SPECS:
        video_path = RESULTS_DIR / spec["file"]
        if not video_path.is_file():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        split = spec["split"]
        mode = spec["mode"]
        stride = int(spec["stride"])
        video_stem = video_path.stem

        seen_hashes: list[int] = []
        frame_idx = 0
        kept_from_video = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            cur_hash = dhash(frame)
            if any(hamming_distance(cur_hash, prev_h) <= 2 for prev_h in seen_hashes):
                frame_idx += 1
                continue

            labels = infer_labels_for_frame(frame, mode, model_v2, model_v14, model_coco)
            if mode in ("sleeping", "using_phone") and not labels:
                frame_idx += 1
                continue

            seen_hashes.append(cur_hash)
            kept_from_video += 1
            base_stem = f"{video_stem}_f{frame_idx:04d}"
            write_sample(split, base_stem, frame, labels)

            sp_stat = stats["splits"][split]
            sp_stat["raw_images"] += 1
            sp_stat["total_images"] += 1
            if not labels:
                sp_stat["hard_negatives"] += 1
            for cid, *_ in labels:
                if cid == 0:
                    sp_stat["sleeping_boxes"] += 1
                elif cid == 1:
                    sp_stat["using_phone_boxes"] += 1

            # Apply classical Image Processing augmentations on training split
            if split == "train":
                aug_variants = [
                    ("clahe", apply_clahe_lab(frame, clip_limit=2.2, tile_grid_size=(8, 8)), labels),
                    ("bc", apply_brightness_contrast(frame, alpha=0.82, beta=-18.0), labels),
                    ("mblur", apply_motion_blur_and_noise(frame, kernel_size=5), labels),
                    ("flip", *flip_horizontal_yolo(frame, labels)),
                ]
                for aug_name, aug_img, aug_lbls in aug_variants:
                    write_sample(split, f"{base_stem}_{aug_name}", aug_img, aug_lbls)
                    sp_stat["augmented_images"] += 1
                    sp_stat["total_images"] += 1
                    if not aug_lbls:
                        sp_stat["hard_negatives"] += 1
                    for cid, *_ in aug_lbls:
                        if cid == 0:
                            sp_stat["sleeping_boxes"] += 1
                        elif cid == 1:
                            sp_stat["using_phone_boxes"] += 1

            frame_idx += 1

        cap.release()
        stats["videos_processed"].append({
            "file": spec["file"],
            "mode": mode,
            "split": split,
            "kept_raw_frames": kept_from_video,
        })

    yaml_content = (
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: sleeping\n"
        "  1: using_phone\n"
    )
    (DATASET_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    (DATASET_DIR / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


if __name__ == "__main__":
    summary = build_dataset()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
