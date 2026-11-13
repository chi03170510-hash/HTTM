"""Select an existing behavior checkpoint without changing inference settings."""

import os
from pathlib import Path


CHECKPOINT_ENV = "HTTM_BEHAVIOR_CHECKPOINT"
DEFAULT_CHECKPOINT = "behavior_yolo.pt"
APPROVED_CHECKPOINTS = (
    DEFAULT_CHECKPOINT,
    "best_behavior_v1_4.pt",
    "best_behavior_v2.pt",
    "best_behavior_v3.pt",
)


def behavior_checkpoint_path():
    name = os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
    if name not in APPROVED_CHECKPOINTS:
        raise ValueError(
            f"Invalid {CHECKPOINT_ENV}={name!r}: expected one of "
            f"{APPROVED_CHECKPOINTS!r}. Unset it to use {DEFAULT_CHECKPOINT}."
        )
    root = Path(__file__).resolve().parents[2]
    if name == "best_behavior_v1_4.pt":
        path = root / "ai/behavior_detection/models" / name
    else:
        path = root / "ai_training/models" / name
        if not path.is_file() and name == DEFAULT_CHECKPOINT:
            fallback = root.parent / "models" / name
            if fallback.is_file():
                path = fallback
    if not path.is_file():
        raise FileNotFoundError(f"Behavior checkpoint not found: {path}")
    return path
