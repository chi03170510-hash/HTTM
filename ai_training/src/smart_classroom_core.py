"""Self-contained AI-Smart-Classroom-Monitoring-System core inference module for HTTM.

Ports ObjectDetector (yolov8n.pt with classroom allowlist), BehaviorDetector
(behavior_yolo.pt filtered to Sleeping, Using Phone, Eating), and UnifiedPipeline.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

AI_TRAINING_DIR = Path(__file__).resolve().parents[1]
HTTM_ROOT = AI_TRAINING_DIR.parent
OUTER_ROOT = HTTM_ROOT.parent

CONFIDENCE_THRESHOLD = 0.35
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.25
IMG_SIZE = 640

CLASSROOM_OBJECT_ALLOWLIST = {
    "person": "Person",
    "chair": "Chair",
    "dining table": "Table",
    "table": "Table",
    "laptop": "Laptop",
    "tv": "Monitor",
    "tvmonitor": "Monitor",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "cell phone": "Cell Phone",
    "phone": "Cell Phone",
    "book": "Book",
    "backpack": "Backpack",
    "bottle": "Bottle",
    "cup": "Cup",
    "clock": "Clock",
    "scissors": "Scissors",
    "handbag": "Handbag",
}

BEHAVIOR_CLASSES = {
    1: "Sleeping",
    2: "Using Phone",
    6: "Eating",
}


def select_default_device(device: str | None = None) -> str:
    """Resolve inference device, preferring GPU CUDA:0 when available."""
    if device is None:
        return "0" if torch.cuda.is_available() else "cpu"
    normalized = str(device).strip()
    lowered = normalized.lower()
    if lowered in ("", "auto"):
        return "0" if torch.cuda.is_available() else "cpu"
    if lowered in ("cuda", "cuda:0"):
        return "0"
    return normalized


def resolve_object_model_path() -> Path:
    candidates = (
        AI_TRAINING_DIR / "models" / "yolov8n.pt",
        HTTM_ROOT / "models" / "yolov8n.pt",
        OUTER_ROOT / "models" / "yolov8n.pt",
        OUTER_ROOT / "yolov8n.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Object detection checkpoint yolov8n.pt not found in expected paths: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_behavior_model_path() -> Path:
    candidates = (
        AI_TRAINING_DIR / "models" / "behavior_yolo.pt",
        HTTM_ROOT / "models" / "behavior_yolo.pt",
        OUTER_ROOT / "models" / "behavior_yolo.pt",
        OUTER_ROOT / "models" / "behavior_train_run" / "weights" / "best.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Behavior detection checkpoint behavior_yolo.pt not found in expected paths: "
        + ", ".join(str(p) for p in candidates)
    )


class ObjectDetector:
    """Model 1 - Classroom Object Detection (YOLOv8n filtered by allowlist)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        conf_thresh: float = CONFIDENCE_THRESHOLD,
        imgsz: int = IMG_SIZE,
        device: str | None = None,
    ):
        self.conf_thresh = float(conf_thresh)
        self.imgsz = int(imgsz)
        self.device = select_default_device(device)
        self.model_path = Path(model_path) if model_path is not None else resolve_object_model_path()
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Object model not found: {self.model_path}")
        logger.info("Loading Object Detection Model from: %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        if self.device != "cpu":
            self.model.to("cuda:0" if self.device == "0" else self.device)
        logger.info("Object Detection Model loaded successfully (device=%s).", self.device)

    def detect(self, frame: np.ndarray) -> dict:
        if self.model is None or frame is None:
            return {"detections": [], "occupancy": 0, "inference_time_ms": 0.0}

        start_time = time.perf_counter()
        predict_kwargs = {
            "conf": self.conf_thresh,
            "imgsz": self.imgsz,
            "device": self.device,
            "half": self.device != "cpu",
            "verbose": False,
        }
        results = self.model.predict(frame, **predict_kwargs)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        detections = []
        person_count = 0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    raw_cls_name = str(result.names.get(cls_id, str(cls_id))).lower()
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    if raw_cls_name not in CLASSROOM_OBJECT_ALLOWLIST:
                        continue

                    display_name = CLASSROOM_OBJECT_ALLOWLIST[raw_cls_name]
                    if raw_cls_name == "person":
                        person_count += 1

                    detections.append({
                        "class_id": cls_id,
                        "class_name": display_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(float(c), 2) for c in xyxy],
                    })

        return {
            "detections": detections,
            "occupancy": person_count,
            "inference_time_ms": inference_time_ms,
        }


class BehaviorDetector:
    """Model 2 - Classroom Behavior Detection (Sleeping, Using Phone, Eating)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        conf_thresh: float = BEHAVIOR_CONFIDENCE_THRESHOLD,
        imgsz: int = IMG_SIZE,
        device: str | None = None,
    ):
        self.conf_thresh = float(conf_thresh)
        self.imgsz = int(imgsz)
        self.device = select_default_device(device)
        self.behavior_classes = dict(BEHAVIOR_CLASSES)
        self.model_path = Path(model_path) if model_path is not None else resolve_behavior_model_path()
        self.model = None
        self.is_custom_model = False
        self._load_model()

    def _load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Behavior model not found: {self.model_path}")
        logger.info("Loading Behavior Detection Model from: %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        if self.device != "cpu":
            self.model.to("cuda:0" if self.device == "0" else self.device)
        names = self.model.names
        name_items = names.items() if isinstance(names, dict) else enumerate(names)
        actual_names = {int(k): str(v).strip() for k, v in name_items}
        mismatches = {
            cls_id: (expected_name, actual_names.get(cls_id))
            for cls_id, expected_name in self.behavior_classes.items()
            if actual_names.get(cls_id) != expected_name
        }
        if mismatches:
            raise RuntimeError(
                f"Unexpected behavior model classes in {self.model_path}: "
                f"expected subset {self.behavior_classes}, mismatches={mismatches}, actual={actual_names}"
            )
        self.is_custom_model = True
        logger.info("Behavior Detection Model loaded successfully (device=%s).", self.device)

    def detect(self, frame: np.ndarray) -> dict:
        if self.model is None or frame is None:
            return {"behaviors": [], "inference_time_ms": 0.0}

        start_time = time.perf_counter()
        behaviors = []

        predict_kwargs = {
            "conf": self.conf_thresh,
            "imgsz": self.imgsz,
            "device": self.device,
            "half": self.device != "cpu",
            "classes": sorted(self.behavior_classes.keys()),
            "verbose": False,
        }
        results = self.model.predict(frame, **predict_kwargs)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    if cls_id not in self.behavior_classes:
                        continue
                    cls_name = self.behavior_classes[cls_id]

                    behaviors.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(float(c), 2) for c in xyxy],
                    })

        return {
            "behaviors": behaviors,
            "inference_time_ms": inference_time_ms,
        }


class UnifiedPipeline:
    """Combines ObjectDetector (yolov8n) and BehaviorDetector (behavior_yolo) into one flow."""

    def __init__(
        self,
        object_model_path: Path | str | None = None,
        behavior_model_path: Path | str | None = None,
        object_conf: float = CONFIDENCE_THRESHOLD,
        behavior_conf: float = BEHAVIOR_CONFIDENCE_THRESHOLD,
        imgsz: int = IMG_SIZE,
        device: str | None = None,
    ):
        logger.info("Initializing AI-Smart Unified Pipeline...")
        self.object_detector = ObjectDetector(
            model_path=object_model_path,
            conf_thresh=object_conf,
            imgsz=imgsz,
            device=device,
        )
        self.behavior_detector = BehaviorDetector(
            model_path=behavior_model_path,
            conf_thresh=behavior_conf,
            imgsz=imgsz,
            device=device,
        )
        logger.info("AI-Smart Unified Pipeline initialized.")

    def process_frame(
        self,
        frame: np.ndarray,
        classroom_id: str = "H305",
        *,
        annotate: bool = False,
    ) -> dict | None:
        if frame is None:
            return None

        now = datetime.now()
        timestamp = now.isoformat()
        formatted_time = now.strftime("%I:%M:%S %p")
        formatted_date = now.strftime("%Y-%m-%d")

        obj_res = self.object_detector.detect(frame)
        objects = obj_res["detections"]
        occupancy = obj_res["occupancy"]

        beh_res = self.behavior_detector.detect(frame)
        behaviors = beh_res["behaviors"]

        total_inference_time = round(
            obj_res["inference_time_ms"] + beh_res["inference_time_ms"], 2
        )

        output = {
            "classroom_id": classroom_id,
            "timestamp": timestamp,
            "date": formatted_date,
            "time": formatted_time,
            "objects": objects,
            "behaviors": behaviors,
            "occupancy": occupancy,
            "object_inference_time_ms": obj_res["inference_time_ms"],
            "behavior_inference_time_ms": beh_res["inference_time_ms"],
            "inference_time_ms": total_inference_time,
        }

        if annotate:
            annotated_frame = frame.copy()
            for obj in objects:
                x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                color = (0, 255, 0) if obj["class_name"].lower() == "person" else (255, 165, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            for beh in behaviors:
                x1, y1, x2, y2 = [int(v) for v in beh["bbox"]]
                label = f"BEH: {beh['class_name']} {beh['confidence']:.2f}"
                color = (255, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 25, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            info_text = (
                f"Classroom: {classroom_id} | Occupancy: {occupancy} | "
                f"FPS latency: {total_inference_time}ms"
            )
            cv2.putText(
                annotated_frame,
                info_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            output["annotated_frame"] = annotated_frame
            output["raw_frame"] = frame

        return output
