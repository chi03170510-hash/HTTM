"""Self-contained AI-Smart-Classroom-Monitoring-System core inference module for HTTM.

Ports ObjectDetector (yolov8n.pt with classroom allowlist), BehaviorDetector
(behavior_yolo.pt + webcam behavior & MediaPipe face/object cues filtered to
Sleeping, Using Phone, Eating), and UnifiedPipeline.
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
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.15
OBJECT_CUE_CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 640

CLASSROOM_OBJECT_ALLOWLIST = {
    "person": "Person",
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

FOOD_OBJECT_CLASSES = {
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "fork",
    "spoon",
}

DRINK_OBJECT_CLASSES = {
    "bottle",
    "cup",
    "wine glass",
}

PHONE_OBJECT_CLASSES = {
    "cell phone",
    "phone",
}

BEHAVIOR_CLASSES = {
    1: "Sleeping",
    2: "Using Phone",
    6: "Eating",
}

AUX_WEBCAM_BEHAVIOR_MAP = {
    "sleeping": (1, "Sleeping"),
    "using_phone": (2, "Using Phone"),
    "Using Phone": (2, "Using Phone"),
    "Sleeping": (1, "Sleeping"),
    "eating": (6, "Eating"),
    "Eating": (6, "Eating"),
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


def resolve_aux_webcam_behavior_model_path() -> Path | None:
    candidates = (
        AI_TRAINING_DIR / "models" / "best_behavior_v2.pt",
        AI_TRAINING_DIR / "models" / "best_behavior_v3.pt",
        HTTM_ROOT / "models" / "best_behavior_v2.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _bbox_area(bbox: list[float] | tuple[float, ...]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_intersection(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    inter = _bbox_intersection(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return inter / union if union > 1e-6 else 0.0


def _bbox_overlap_ratio(inner_box: list[float] | tuple[float, ...], outer_box: list[float] | tuple[float, ...]) -> float:
    inner_area = _bbox_area(inner_box)
    if inner_area <= 1e-6:
        return 0.0
    return _bbox_intersection(inner_box, outer_box) / inner_area


def _expand_bbox(
    bbox: list[float] | tuple[float, ...],
    frame_shape: tuple[int, ...],
    scale_x: float = 0.12,
    scale_y: float = 0.12,
) -> list[float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    return [
        max(0.0, x1 - bw * scale_x),
        max(0.0, y1 - bh * scale_y),
        min(float(w), x2 + bw * scale_x),
        min(float(h), y2 + bh * scale_y),
    ]


def _deduplicate_behaviors(behaviors: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    if not behaviors:
        return []
    sorted_beh = sorted(behaviors, key=lambda b: float(b.get("confidence", 0.0)), reverse=True)
    kept: list[dict] = []
    for cand in sorted_beh:
        cand_cls = cand["class_id"]
        cand_box = cand["bbox"]
        duplicate = False
        for existing in kept:
            if existing["class_id"] != cand_cls:
                continue
            if (
                _bbox_iou(cand_box, existing["bbox"]) >= iou_thresh
                or _bbox_overlap_ratio(cand_box, existing["bbox"]) >= 0.70
                or _bbox_overlap_ratio(existing["bbox"], cand_box) >= 0.70
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


class ObjectDetector:
    """Model 1 - Classroom Object Detection (YOLOv8n filtered by allowlist + behavior cues)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        conf_thresh: float = CONFIDENCE_THRESHOLD,
        imgsz: int = IMG_SIZE,
        device: str | None = None,
    ):
        self.conf_thresh = float(conf_thresh)
        self.cue_conf_thresh = min(float(conf_thresh), OBJECT_CUE_CONFIDENCE_THRESHOLD)
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
        empty_cues = {"persons": [], "phones": [], "foods": [], "drinks": []}
        if self.model is None or frame is None:
            return {
                "detections": [],
                "occupancy": 0,
                "inference_time_ms": 0.0,
                "raw_cues": empty_cues,
            }

        start_time = time.perf_counter()
        predict_kwargs = {
            "conf": self.cue_conf_thresh,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        results = self.model.predict(frame, **predict_kwargs)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        detections = []
        person_count = 0
        raw_cues: dict[str, list[dict]] = {"persons": [], "phones": [], "foods": [], "drinks": []}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    raw_cls_name = str(result.names.get(cls_id, str(cls_id))).lower()
                    conf = float(box.conf[0].item())
                    xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                    item = {
                        "class_id": cls_id,
                        "raw_name": raw_cls_name,
                        "confidence": round(conf, 4),
                        "bbox": xyxy,
                    }

                    if raw_cls_name == "person" and conf >= min(self.conf_thresh, 0.25):
                        raw_cues["persons"].append(item)
                    elif raw_cls_name in PHONE_OBJECT_CLASSES and conf >= self.cue_conf_thresh:
                        raw_cues["phones"].append(item)
                    elif raw_cls_name in FOOD_OBJECT_CLASSES and conf >= self.cue_conf_thresh:
                        raw_cues["foods"].append(item)
                    elif raw_cls_name in DRINK_OBJECT_CLASSES and conf >= 0.20:
                        raw_cues["drinks"].append(item)

                    if raw_cls_name not in CLASSROOM_OBJECT_ALLOWLIST:
                        continue

                    min_display_conf = self.conf_thresh
                    if raw_cls_name == "person":
                        min_display_conf = min(self.conf_thresh, 0.25)
                    elif raw_cls_name in PHONE_OBJECT_CLASSES:
                        min_display_conf = min(self.conf_thresh, 0.18)

                    if conf < min_display_conf:
                        continue

                    display_name = CLASSROOM_OBJECT_ALLOWLIST[raw_cls_name]
                    if raw_cls_name == "person":
                        person_count += 1

                    detections.append({
                        "class_id": cls_id,
                        "class_name": display_name,
                        "confidence": round(conf, 4),
                        "bbox": xyxy,
                    })

        return {
            "detections": detections,
            "occupancy": person_count,
            "inference_time_ms": inference_time_ms,
            "raw_cues": raw_cues,
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
        self.aux_webcam_model = None
        self.face_extractor = None
        self.is_custom_model = False
        self._load_model()
        self._load_auxiliary_detectors()

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

    def _load_auxiliary_detectors(self):
        aux_path = resolve_aux_webcam_behavior_model_path()
        if aux_path is not None and aux_path.resolve() != self.model_path.resolve():
            try:
                self.aux_webcam_model = YOLO(str(aux_path))
                if self.device != "cpu":
                    self.aux_webcam_model.to("cuda:0" if self.device == "0" else self.device)
                logger.info("Loaded auxiliary webcam behavior model: %s", aux_path)
            except Exception as exc:
                logger.warning("Could not load auxiliary webcam behavior model %s: %s", aux_path, exc)
                self.aux_webcam_model = None

        try:
            try:
                from src.face_features import FaceFeatureExtractor
            except ImportError:
                from face_features import FaceFeatureExtractor  # type: ignore

            usable, reason = FaceFeatureExtractor.availability()
            if usable:
                self.face_extractor = FaceFeatureExtractor(min_detection_confidence=0.35)
                logger.info("Loaded MediaPipe FaceFeatureExtractor (%s).", reason)
        except Exception as exc:
            logger.debug("FaceFeatureExtractor unavailable: %s", exc)
            self.face_extractor = None

    def _extract_person_face_features(self, frame: np.ndarray, person_bbox: list[float]) -> dict | None:
        if self.face_extractor is None or frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        frame_area = max(float(h * w), 1.0)
        p_area = _bbox_area(person_bbox)

        x1 = max(0, int(person_bbox[0]))
        y1 = max(0, int(person_bbox[1]))
        x2 = min(w, int(person_bbox[2]))
        y2 = min(h, int(person_bbox[3]))
        if x2 - x1 < 45 or y2 - y1 < 45:
            return None

        try:
            if p_area >= 0.22 * frame_area:
                feat_full = self.face_extractor.extract(frame)
                if feat_full.get("eyes_valid"):
                    mc = feat_full.get("mouth_center_xy")
                    if mc is None or (x1 - 30 <= mc[0] <= x2 + 30 and y1 - 30 <= mc[1] <= y2 + 30):
                        return feat_full
            crop = frame[y1:y2, x1:x2]
            feat_crop = self.face_extractor.extract(crop)
            if feat_crop.get("eyes_valid") and feat_crop.get("mouth_center_xy") is not None:
                mc = feat_crop["mouth_center_xy"]
                feat_crop = dict(feat_crop)
                feat_crop["mouth_center_xy"] = [float(mc[0]) + x1, float(mc[1]) + y1]
            return feat_crop
        except Exception:
            return None

    def detect(self, frame: np.ndarray, object_cues: dict | None = None) -> dict:
        if self.model is None or frame is None:
            return {"behaviors": [], "inference_time_ms": 0.0}

        start_time = time.perf_counter()
        raw_primary_behaviors: list[dict] = []

        predict_kwargs = {
            "conf": min(self.conf_thresh, 0.14),
            "imgsz": self.imgsz,
            "device": self.device,
            "classes": sorted(self.behavior_classes.keys()),
            "verbose": False,
        }
        results = self.model.predict(frame, **predict_kwargs)

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                    if cls_id not in self.behavior_classes:
                        continue
                    cls_name = self.behavior_classes[cls_id]
                    raw_primary_behaviors.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": xyxy,
                    })

        aux_behaviors: list[dict] = []
        if self.aux_webcam_model is not None and frame.size > 0 and int(np.ptp(frame)) > 5:
            try:
                aux_results = self.aux_webcam_model.predict(
                    frame,
                    conf=0.20,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                if aux_results and len(aux_results) > 0:
                    aux_res = aux_results[0]
                    aux_boxes = aux_res.boxes
                    if aux_boxes is not None:
                        for box in aux_boxes:
                            raw_id = int(box.cls[0].item())
                            raw_name = str(aux_res.names.get(raw_id, str(raw_id))).strip()
                            mapped = AUX_WEBCAM_BEHAVIOR_MAP.get(raw_name)
                            if mapped is None:
                                continue
                            mapped_id, mapped_name = mapped
                            conf = float(box.conf[0].item())
                            xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                            aux_behaviors.append({
                                "class_id": mapped_id,
                                "class_name": mapped_name,
                                "confidence": round(conf, 4),
                                "bbox": xyxy,
                            })
            except Exception as exc:
                logger.debug("Auxiliary behavior inference skipped: %s", exc)

        fused_behaviors = self._fuse_behavior_cues(
            frame=frame,
            primary_behaviors=raw_primary_behaviors,
            aux_behaviors=aux_behaviors,
            object_cues=object_cues,
        )
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            "behaviors": fused_behaviors,
            "inference_time_ms": inference_time_ms,
        }

    def _fuse_behavior_cues(
        self,
        frame: np.ndarray,
        primary_behaviors: list[dict],
        aux_behaviors: list[dict],
        object_cues: dict | None,
    ) -> list[dict]:
        cues = object_cues or {}
        persons = cues.get("persons", [])
        phones = cues.get("phones", [])
        foods = cues.get("foods", [])
        drinks = cues.get("drinks", [])

        candidates: list[dict] = []

        if not persons:
            for beh in primary_behaviors:
                if beh["confidence"] >= max(self.conf_thresh, 0.22):
                    candidates.append(beh)
            for beh in aux_behaviors:
                if beh["confidence"] >= 0.35:
                    candidates.append(beh)
            for phone in phones:
                if phone["confidence"] >= 0.20:
                    candidates.append({
                        "class_id": 2,
                        "class_name": "Using Phone",
                        "confidence": round(min(0.95, max(phone["confidence"] + 0.30, 0.55)), 4),
                        "bbox": phone["bbox"],
                    })
            for food in foods:
                if food["confidence"] >= 0.20:
                    candidates.append({
                        "class_id": 6,
                        "class_name": "Eating",
                        "confidence": round(min(0.95, max(food["confidence"] + 0.35, 0.55)), 4),
                        "bbox": food["bbox"],
                    })
            return _deduplicate_behaviors(candidates)

        h, w = frame.shape[:2]
        frame_area = max(float(h * w), 1.0)
        max_person_area = max((_bbox_area(p["bbox"]) for p in persons), default=1.0)

        for person in persons:
            p_box = person["bbox"]
            p_area = max(_bbox_area(p_box), 1.0)
            # Skip tiny background/reflection person boxes when a prominent foreground person is present
            if max_person_area >= 0.20 * frame_area and p_area < 0.15 * max_person_area:
                continue

            exp_p_box = _expand_bbox(p_box, frame.shape, scale_x=0.12, scale_y=0.12)
            px1, py1, px2, py2 = p_box
            pw = max(px2 - px1, 1.0)
            ph = max(py2 - py1, 1.0)

            has_phone = False
            has_eating = False

            # --- 1. USING PHONE CUES (class_id = 2) ---
            best_phone_conf = 0.0
            for phone in phones:
                overlap = _bbox_overlap_ratio(phone["bbox"], exp_p_box)
                if overlap >= 0.35:
                    boosted = min(0.96, max(float(phone["confidence"]) + 0.35, 0.58))
                    best_phone_conf = max(best_phone_conf, boosted)

            for beh in aux_behaviors:
                if beh["class_id"] == 2 and beh["confidence"] >= 0.35:
                    if (
                        _bbox_overlap_ratio(beh["bbox"], exp_p_box) >= 0.35
                        or _bbox_iou(beh["bbox"], p_box) >= 0.25
                    ):
                        boosted = min(0.95, max(float(beh["confidence"]) + 0.20, 0.56))
                        best_phone_conf = max(best_phone_conf, boosted)

            for beh in primary_behaviors:
                if beh["class_id"] == 2 and beh["confidence"] >= 0.22:
                    if (
                        _bbox_overlap_ratio(beh["bbox"], exp_p_box) >= 0.35
                        and _bbox_area(beh["bbox"]) >= 0.10 * p_area
                    ):
                        best_phone_conf = max(best_phone_conf, float(beh["confidence"]))

            if best_phone_conf > 0.0:
                has_phone = True
                candidates.append({
                    "class_id": 2,
                    "class_name": "Using Phone",
                    "confidence": round(best_phone_conf, 4),
                    "bbox": list(p_box),
                })

            # --- 2. EATING CUES (class_id = 6) ---
            best_eating_conf = 0.0
            face_feat = self._extract_person_face_features(frame, p_box)
            mouth_xy = face_feat.get("mouth_center_xy") if face_feat else None

            for food in foods:
                f_box = food["bbox"]
                f_cx = 0.5 * (f_box[0] + f_box[2])
                overlap = _bbox_overlap_ratio(f_box, exp_p_box)
                in_central_span = (px1 + 0.10 * pw) <= f_cx <= (px2 - 0.10 * pw)
                not_bottom_border_artifact = float(f_box[3]) < (h - 12.0)
                if overlap >= 0.35 and in_central_span and not_bottom_border_artifact:
                    boosted = min(0.95, max(float(food["confidence"]) + 0.40, 0.60))
                    best_eating_conf = max(best_eating_conf, boosted)

            for drink in drinks:
                d_box = drink["bbox"]
                d_cx = 0.5 * (d_box[0] + d_box[2])
                d_cy = 0.5 * (d_box[1] + d_box[3])
                overlap = _bbox_overlap_ratio(d_box, exp_p_box)
                in_central_span = (px1 + 0.18 * pw) <= d_cx <= (px2 - 0.18 * pw)
                in_upper_body = d_cy <= (py1 + 0.68 * ph)
                near_mouth = (
                    mouth_xy is not None
                    and abs(d_cx - float(mouth_xy[0])) <= 0.25 * pw
                    and abs(d_cy - float(mouth_xy[1])) <= 0.28 * ph
                )
                if overlap >= 0.40 and ((in_central_span and in_upper_body) or near_mouth):
                    boosted = min(0.92, max(float(drink["confidence"]) + 0.30, 0.58))
                    best_eating_conf = max(best_eating_conf, boosted)

            for beh in primary_behaviors:
                if beh["class_id"] == 6 and beh["confidence"] >= 0.22:
                    if (
                        _bbox_overlap_ratio(beh["bbox"], exp_p_box) >= 0.35
                        and _bbox_area(beh["bbox"]) >= 0.10 * p_area
                    ):
                        best_eating_conf = max(best_eating_conf, float(beh["confidence"]))

            if face_feat is not None:
                mouth_open = face_feat.get("mouth_open_ratio")
                if mouth_open is not None and float(mouth_open) >= 0.095 and not has_phone:
                    mouth_conf = min(0.90, 0.56 + (float(mouth_open) - 0.095) * 2.5)
                    best_eating_conf = max(best_eating_conf, mouth_conf)

            if best_eating_conf > 0.0:
                has_eating = True
                candidates.append({
                    "class_id": 6,
                    "class_name": "Eating",
                    "confidence": round(best_eating_conf, 4),
                    "bbox": list(p_box),
                })

            # --- 3. SLEEPING CUES (class_id = 1) - Focus on bending over / slumping (ignore eye blinking) ---
            if not has_phone and not has_eating:
                best_sleeping_conf = 0.0
                upright_head = False
                face_hidden = True

                if face_feat is not None:
                    eyes_valid = bool(face_feat.get("eyes_valid"))
                    pose = face_feat.get("head_pose") or {}
                    pose_valid = bool(pose.get("valid"))
                    roll_deg = abs(float(pose.get("roll_deg") or 0.0)) if pose_valid else 0.0
                    pitch_deg = abs(float(pose.get("pitch_deg") or 0.0)) if pose_valid else 0.0
                    mc = face_feat.get("mouth_center_xy")
                    mc_rel_y = ((float(mc[1]) - py1) / ph) if (mc is not None and ph > 1.0) else 0.5

                    if eyes_valid or pose_valid:
                        face_hidden = False

                    # Sitting upright (even if blinking/closing eyes) -> never trigger Sleeping
                    if pose_valid and roll_deg < 19.0 and pitch_deg < 19.0 and mc_rel_y < 0.74:
                        upright_head = True

                    # Slumped / bent head posture (strong head drop/roll or low head position on desk)
                    if pose_valid and not upright_head:
                        if roll_deg >= 20.0 or pitch_deg >= 20.0:
                            tilt_score = max(roll_deg - 20.0, pitch_deg - 20.0, 0.0)
                            pose_conf = min(0.94, 0.65 + tilt_score * 0.02)
                            best_sleeping_conf = max(best_sleeping_conf, pose_conf)
                        elif (roll_deg >= 15.0 or pitch_deg >= 15.0) and mc_rel_y >= 0.74:
                            best_sleeping_conf = max(best_sleeping_conf, 0.72)

                # Torso bent-over geometry when upright face is hidden (buried on desk / folded arms)
                aspect_ratio = pw / ph
                if (
                    face_hidden
                    and p_area >= 0.22 * frame_area
                    and py1 >= 0.32 * h
                    and py2 >= 0.88 * h
                    and aspect_ratio >= 1.38
                ):
                    slump_conf = min(0.90, 0.64 + (aspect_ratio - 1.38) * 0.35)
                    best_sleeping_conf = max(best_sleeping_conf, slump_conf)

                for beh in aux_behaviors:
                    if beh["class_id"] == 1 and beh["confidence"] >= 0.22 and not upright_head:
                        if (
                            _bbox_overlap_ratio(beh["bbox"], exp_p_box) >= 0.30
                            or _bbox_iou(beh["bbox"], p_box) >= 0.20
                        ):
                            boosted = min(0.94, max(float(beh["confidence"]) + 0.30, 0.60))
                            best_sleeping_conf = max(best_sleeping_conf, boosted)

                for beh in primary_behaviors:
                    if beh["class_id"] == 1 and beh["confidence"] >= 0.12 and not upright_head:
                        overlap = _bbox_overlap_ratio(beh["bbox"], exp_p_box)
                        area_ratio = _bbox_area(beh["bbox"]) / p_area
                        if (overlap >= 0.30 and area_ratio >= 0.10) or (face_hidden and beh["confidence"] >= 0.12):
                            boosted = min(0.90, max(float(beh["confidence"]) + 0.40, 0.58))
                            best_sleeping_conf = max(best_sleeping_conf, boosted)

                if best_sleeping_conf > 0.0:
                    candidates.append({
                        "class_id": 1,
                        "class_name": "Sleeping",
                        "confidence": round(best_sleeping_conf, 4),
                        "bbox": list(p_box),
                    })

        return _deduplicate_behaviors(candidates)


class UnifiedPipeline:
    """Combines ObjectDetector (yolov8n) and BehaviorDetector (behavior_yolo + cues) into one flow."""

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
        raw_cues = obj_res.get("raw_cues")

        beh_res = self.behavior_detector.detect(frame, object_cues=raw_cues)
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
                is_person = obj["class_name"].lower() == "person"
                color = (0, 255, 0) if is_person else (255, 165, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                if not is_person:
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
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
                label = f"{beh['class_name']} {beh['confidence']:.2f}"
                color = (255, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 8, 18)),
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
