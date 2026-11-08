"""Single-frame person/phone/pose/hand evidence for AI-R6D.

All coordinates exposed by this module are in source-frame pixels. Static
geometry is reported as evidence and is not a behavior label or ground truth.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
try:
    import mediapipe as mp
except ImportError:  # COCO-only fallback remains available.
    mp = None
import numpy as np
import ultralytics
from ultralytics import YOLO


class UsingPhonePrototype:
    def __init__(self, model_path: str | Path | None = None, conf: float = 0.12,
                 pose_model_path: str | Path | None = None,
                 hand_model_path: str | Path | None = None, device: str | None = None):
        base = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else base / "models" / "pretrained_person" / "yolo11n_coco.pt"
        self.pose_model_path = Path(pose_model_path) if pose_model_path else base / "models" / "pose_landmarker_full.task"
        self.hand_model_path = Path(hand_model_path) if hand_model_path else base / "models" / "hand_landmarker.task"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"COCO checkpoint not found: {self.model_path}")
        # Pose/Hand are optional evidence sources. Missing assets disable that
        # branch while leaving COCO person/phone/food detection available.
        self.conf = float(conf)
        if device is None:
            try:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.model = YOLO(str(self.model_path))
        self.class_names = {int(k): str(v).strip().lower() for k, v in self.model.names.items()}
        self.food_classes = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
        self.object_class_ids = [idx for idx, name in self.class_names.items()
                                 if name in {"person", "cell phone"} | self.food_classes]
        tasks = mp.tasks if mp is not None else None
        image_mode = tasks.vision.RunningMode.IMAGE if tasks is not None else None
        self.pose = None; self.hand = None
        if tasks is not None and self.pose_model_path.is_file():
            self.pose = tasks.vision.PoseLandmarker.create_from_options(tasks.vision.PoseLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(self.pose_model_path)),
                running_mode=image_mode, num_poses=2,
                min_pose_detection_confidence=0.35, min_pose_presence_confidence=0.35,
                min_tracking_confidence=0.35))
        if tasks is not None and self.hand_model_path.is_file():
            self.hand = tasks.vision.HandLandmarker.create_from_options(tasks.vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(self.hand_model_path)),
                running_mode=image_mode, num_hands=2))
        self.model_version = f"Ultralytics {ultralytics.__version__}; COCO {self.model_path.name}"
        mp_version = getattr(mp,"__version__","package unavailable") if mp is not None else "package unavailable"
        self.pose_version = f"MediaPipe Tasks PoseLandmarker {mp_version}; {self.pose_model_path.name}" if self.pose else "unavailable; COCO-only fallback"
        self.hand_version = f"MediaPipe Tasks HandLandmarker {mp_version}; {self.hand_model_path.name}" if self.hand else "unavailable; no fingertip evidence"

    @staticmethod
    def _empty(status: str, reason: str) -> dict[str, Any]:
        return {"person_detections": [], "phone_detections": [], "food_detections": [], "food_person_associations": [],
                "wrist_landmarks": [], "pose_landmarks": [],
                "hand_detections": [], "associations": [], "hand_phone_associations": [],
                "interaction_evidence": [], "using_phone_status": status, "reasons": [reason],
                "model_versions": {}, "inference_ms": None}

    @staticmethod
    def _person_crop(frame: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, int, int]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = (x2-x1)*0.08, (y2-y1)*0.06
        cx1, cy1 = max(0, int(x1-pad_x)), max(0, int(y1-pad_y))
        cx2, cy2 = min(w, int(x2+pad_x)), min(h, int(y2+pad_y))
        return frame[cy1:cy2, cx1:cx2], cx1, cy1

    @staticmethod
    def _inside(point: tuple[float, float], bbox: list[float]) -> bool:
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def predict(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if not isinstance(frame_bgr, np.ndarray):
            return self._empty("NOT_ASSESSABLE", "frame_must_be_numpy_array")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return self._empty("NOT_ASSESSABLE", "frame_shape_must_be_HxWx3_BGR")
        if frame_bgr.dtype != np.uint8:
            return self._empty("NOT_ASSESSABLE", "frame_dtype_must_be_uint8")
        h, w = frame_bgr.shape[:2]
        if not h or not w:
            return self._empty("NOT_ASSESSABLE", "frame_is_empty")
        if min(h, w) < 64:
            return self._empty("NOT_ASSESSABLE", "frame_too_small_for_reliable_inference")

        started = perf_counter()
        yolo_started = perf_counter()
        result = self.model.predict(source=frame_bgr, classes=self.object_class_ids, conf=self.conf,
                                    imgsz=640, device=self.device, verbose=False)[0]
        yolo_ms = (perf_counter() - yolo_started) * 1000
        persons, phones, foods = [], [], []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.item())
                label = self.class_names.get(cls_id, f"class_{cls_id}")
                kind = "person" if label == "person" else "phone" if label == "cell phone" else "food"
                item = {"detection_id": f"{kind}_{i}", "class_id": cls_id, "class_name": label,
                        "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                        "confidence": round(float(box.conf.item()), 4)}
                if kind == "person": persons.append(item)
                elif kind == "phone": phones.append(item)
                elif kind == "food": foods.append(item)
        output: dict[str, Any] = {
            "image_width": w, "image_height": h, "person_detections": persons,
            "phone_detections": phones, "food_detections": foods, "food_person_associations": [],
            "wrist_landmarks": [], "hand_detections": [],
            "pose_landmarks": [],
            "associations": [], "hand_phone_associations": [], "interaction_evidence": [],
            "using_phone_status": "UNCERTAIN", "reasons": [],
            "model_versions": {"object_detector": self.model_version,
                               "pose_estimator": self.pose_version,
                               "hand_estimator": self.hand_version},
        }
        if not persons:
            output.update(using_phone_status="NOT_ASSESSABLE", reasons=["no_person_detected"])
            total_ms = (perf_counter()-started)*1000
            output["inference_ms"] = round(total_ms, 2)
            output["timing_stages_ms"] = {"yolo_ms": round(yolo_ms, 3), "pose_ms": 0.0,
                "hand_ms": 0.0, "association_rules_ms": round(max(0.0, total_ms-yolo_ms), 3),
                "total_ms": round(total_ms, 3)}
            return output

        pose_by_person: dict[str, list[dict[str, Any]]] = {}
        pose_ms = 0.0
        hand_ms = 0.0
        for person in persons:
            crop, ox, oy = self._person_crop(frame_bgr, person["bbox_xyxy"])
            pose_records = []
            mp_image = None
            if crop.size and (self.pose is not None or self.hand is not None):
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if crop.size and self.pose is not None:
                pose_started = perf_counter()
                pose_result = self.pose.detect(mp_image)
                pose_ms += (perf_counter() - pose_started) * 1000
                for pose in pose_result.pose_landmarks if pose_result is not None else []:
                    pts = {}
                    for name, idx in (("nose", 0), ("left_ear", 7), ("right_ear", 8),
                                      ("left_shoulder", 11), ("right_shoulder", 12),
                                      ("left_elbow", 13), ("right_elbow", 14),
                                      ("left_wrist", 15), ("right_wrist", 16)):
                        lm = pose[idx]
                        pts[name] = {"x": ox+float(lm.x)*crop.shape[1],
                                     "y": oy+float(lm.y)*crop.shape[0],
                                     "visibility": float(lm.visibility) if lm.visibility is not None else None,
                                     "presence": float(lm.presence) if lm.presence is not None else None}
                    pose_records.append(pts)
                    output["pose_landmarks"].append({"person_id": person["detection_id"], "landmarks": pts})
                    for name in ("left_wrist", "right_wrist"):
                        p = pts[name]
                        if p["visibility"] is not None and p["visibility"] >= 0.35:
                            output["wrist_landmarks"].append({"person_id": person["detection_id"], "landmark": name, **p})
            pose_by_person[person["detection_id"]] = pose_records

            # Run hands on each padded person crop, then map normalized crop coordinates
            # directly back to source-frame pixels. The crop that produced a hand is
            # diagnostic provenance only; it does not assign ownership.
            if crop.size and self.hand is not None:
                hand_started = perf_counter()
                hand_result = self.hand.detect(mp_image)
                hand_ms += (perf_counter() - hand_started) * 1000
                for hand_idx, landmarks in enumerate(hand_result.hand_landmarks if hand_result is not None else []):
                    points = [{"x": ox+float(lm.x)*crop.shape[1],
                               "y": oy+float(lm.y)*crop.shape[0],
                               "z": float(lm.z)} for lm in landmarks]
                    handedness = hand_result.handedness[hand_idx] if hand_idx < len(hand_result.handedness) else []
                    category = handedness[0] if handedness else None
                    hand_id = f"hand_{len(output['hand_detections'])}"
                    output["hand_detections"].append({
                        "detection_id": hand_id, "source_crop_person_id": person["detection_id"],
                        "landmarks_xy": [[round(p["x"], 2), round(p["y"], 2)] for p in points],
                        "wrist_xy": [round(points[0]["x"], 2), round(points[0]["y"], 2)],
                        "palm_xy": [round(points[9]["x"], 2), round(points[9]["y"], 2)],
                        "fingertips_xy": [[round(points[i]["x"], 2), round(points[i]["y"], 2)] for i in (4, 8, 12, 16, 20)],
                        "handedness": getattr(category, "category_name", None),
                        "handedness_score": round(float(category.score), 4) if category is not None else None,
                        "person_association": None, "person_association_status": "UNCERTAIN"})

        # Associate detected hands with frame-level person boxes, never with their
        # source crop alone. Require wrist and palm to be inside the same candidate.
        for hand in output["hand_detections"]:
            wrist, palm = hand["wrist_xy"], hand["palm_xy"]
            candidates = [p for p in persons if self._inside(wrist, p["bbox_xyxy"]) and self._inside(palm, p["bbox_xyxy"])]
            hand["person_candidate_ids"] = [p["detection_id"] for p in candidates]
            if len(candidates) == 1:
                hand["person_association"] = candidates[0]["detection_id"]
                hand["person_association_status"] = "ASSOCIATED_GEOMETRICALLY"
            elif len(candidates) > 1:
                hand["person_association_status"] = "UNCERTAIN"
            else:
                hand["person_association_status"] = "UNASSOCIATED_OUTSIDE_PERSON_BBOX"

        phone_person: dict[str, str | None] = {}
        phone_ambiguous = False
        for phone in phones:
            x1, y1, x2, y2 = phone["bbox_xyxy"]
            pcx, pcy = (x1+x2)/2, (y1+y2)/2
            candidates = []
            for person in persons:
                a, b, c, d = person["bbox_xyxy"]
                pw, ph = max(1., c-a), max(1., d-b)
                inside = a <= pcx <= c and b <= pcy <= d
                dx, dy = max(a-pcx, 0., pcx-c), max(b-pcy, 0., pcy-d)
                rel = float(np.hypot(dx/pw, dy/ph))
                if inside or rel <= 0.12:
                    candidates.append((person, 0. if inside else rel))
            owner = candidates[0][0] if len(candidates) == 1 else None
            phone_person[phone["detection_id"]] = owner["detection_id"] if owner else None
            if len(candidates) == 1:
                output["associations"].append({"phone_detection_id": phone["detection_id"],
                    "person_detection_id": owner["detection_id"], "status": "ASSOCIATED_GEOMETRICALLY",
                    "candidate_count": 1, "relative_distance": round(candidates[0][1], 4)})
            elif len(candidates) > 1:
                phone_ambiguous = True
                output["associations"].append({"phone_detection_id": phone["detection_id"],
                    "person_detection_id": None, "status": "UNCERTAIN", "candidate_count": len(candidates),
                    "candidate_person_ids": [p["detection_id"] for p, _ in candidates]})
            else:
                output["associations"].append({"phone_detection_id": phone["detection_id"],
                    "person_detection_id": None, "status": "UNASSOCIATED", "candidate_count": 0})

        for food in foods:
            x1,y1,x2,y2=food["bbox_xyxy"]; cx,cy=(x1+x2)/2,(y1+y2)/2
            owners=[p for p in persons if self._inside((cx,cy),p["bbox_xyxy"])]
            output["food_person_associations"].append({
                "food_detection_id":food["detection_id"],
                "person_detection_id":owners[0]["detection_id"] if len(owners)==1 else None,
                "status":"ASSOCIATED_GEOMETRICALLY" if len(owners)==1 else "UNCERTAIN" if owners else "UNASSOCIATED",
                "candidate_person_ids":[p["detection_id"] for p in owners]})

        positive = False
        for phone in phones:
            x1, y1, x2, y2 = phone["bbox_xyxy"]
            diag = max(1., float(np.hypot(x2-x1, y2-y1)))
            owner_id = phone_person[phone["detection_id"]]
            pair_records = []
            for hand in output["hand_detections"]:
                tips = hand["fingertips_xy"]
                inside_count = sum(self._inside(tuple(tip), phone["bbox_xyxy"]) for tip in tips)
                distances = [float(np.hypot(max(x1-x, 0., x-x2), max(y1-y, 0., y-y2))) / diag for x, y in tips]
                min_distance = min(distances, default=None)
                associated_same_person = owner_id is not None and hand["person_association_status"] == "ASSOCIATED_GEOMETRICALLY" and hand["person_association"] == owner_id
                # Contact evidence uses multiple fingertip landmarks inside the detected
                # phone box; distance by itself is recorded but never makes a positive.
                contact = inside_count >= 2 and associated_same_person
                record = {"phone_detection_id": phone["detection_id"], "hand_detection_id": hand["detection_id"],
                          "person_detection_id": hand["person_association"],
                          "phone_person_detection_id": owner_id,
                          "hand_person_status": hand["person_association_status"],
                          "fingertip_landmarks_inside_phone": inside_count,
                          "nearest_fingertip_to_phone_normalized": round(min_distance, 4) if min_distance is not None else None,
                          "hand_phone_status": "CONTACT_GEOMETRY" if contact else ("UNCERTAIN" if min_distance is not None and min_distance <= 0.25 else "NO_CONTACT_EVIDENCE"),
                          "contact_evidence": contact}
                pair_records.append(record)
                output["hand_phone_associations"].append(record)
                if contact:
                    positive = True
            pose_records = pose_by_person.get(owner_id, []) if owner_id else []
            output["interaction_evidence"].append({"phone_detection_id": phone["detection_id"],
                "person_detection_id": owner_id, "phone_person_status": "ASSOCIATED_GEOMETRICALLY" if owner_id else "UNCERTAIN",
                "pose_candidate_count": len(pose_records), "hand_pair_count": len(pair_records),
                "qualifying_hand_contact_count": sum(r["contact_evidence"] for r in pair_records),
                "status": "USING_PHONE_EVIDENCE" if any(r["contact_evidence"] for r in pair_records) else "UNCERTAIN"})

        if not phones:
            output.update(using_phone_status="NO_CLEAR_PHONE_USE", reasons=["no_coco_cell_phone_detection"])
        elif positive:
            output.update(using_phone_status="USING_PHONE_EVIDENCE",
                          reasons=["unique_phone_person_and_hand_person_associations_with_multiple_fingertips_inside_phone_bbox"])
        elif phone_ambiguous or any(h["person_association_status"] == "UNCERTAIN" for h in output["hand_detections"]):
            output.update(using_phone_status="UNCERTAIN", reasons=["person_or_phone_association_ambiguous_or_contact_not_confirmed"])
        elif not any(phone_person.values()):
            output.update(using_phone_status="NO_CLEAR_PHONE_USE", reasons=["phone_not_associated_with_a_person"])
        else:
            output.update(using_phone_status="UNCERTAIN", reasons=["hand_contact_not_confirmed_by_multiple_fingertip_landmarks"])
        output["wrist_landmarks_assessable"] = len(output["wrist_landmarks"])
        total_ms = (perf_counter()-started)*1000
        output["inference_ms"] = round(total_ms, 2)
        output["timing_stages_ms"] = {"yolo_ms": round(yolo_ms, 3), "pose_ms": round(pose_ms, 3),
            "hand_ms": round(hand_ms, 3),
            "association_rules_ms": round(max(0.0, total_ms-yolo_ms-pose_ms-hand_ms), 3),
            "total_ms": round(total_ms, 3)}
        return output

    def close(self) -> None:
        for landmarker in (getattr(self, "pose", None), getattr(self, "hand", None)):
            if landmarker is not None:
                landmarker.close()
