"""Fine-tune YOLO11 behavior detector on HTTM Webcam Behavior V3 dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai_training"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 behavior model on HTTM Webcam V3 dataset")
    parser.add_argument(
        "--weights",
        type=Path,
        default=AI_DIR / "models" / "best_behavior_v2.pt",
        help="Initial YOLO11 checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=AI_DIR / "datasets" / "httm_webcam_behavior_v3" / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=0.0005,
        mosaic=0.5,
        fliplr=0.5,
        project=str(AI_DIR / "runs" / "behavior_v3"),
        name="webcam_finetune",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
