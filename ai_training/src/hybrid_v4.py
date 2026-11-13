"""Hybrid V4: high-resolution person proposals and stricter evidence gates.

This layer is isolated from V1.4/V2/V3. It reuses the existing COCO model,
MediaPipe extractors, V3 association/fusion, and V3 timestamp lifecycle.
No detector confidence is treated as a calibrated behavior probability.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any
import math
import time
import numpy as np

from src.hybrid_v3 import BEHAVIORS, clip_box, infer_v3, spatial_v3, TemporalV3Fusion


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb > inter else 0.


def _intersection_over_smaller(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / max(1., min(aa, bb))


def _box_quality(box, confidence, width, height, config, source):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(0., x2-x1), max(0., y2-y1)
    area_ratio = bw*bh/max(1., width*height)
    width_ratio, height_ratio = bw/max(1., width), bh/max(1., height)
    coarse = (area_ratio > float(config["person_coarse_area_ratio"])
              or width_ratio > float(config["person_coarse_width_ratio"])
              or height_ratio > float(config["person_coarse_height_ratio"]))
    return {
        "source": source, "confidence_object_detector": float(confidence),
        "area_frame_ratio": round(area_ratio, 5),
        "width_frame_ratio": round(width_ratio, 5),
        "height_frame_ratio": round(height_ratio, 5),
        "quality": "coarse" if coarse else "usable",
        "coarse_box_cannot_confirm_behavior": coarse,
    }


def _person_class_id(objects):
    matches = [i for i, name in objects.class_names.items() if name == "person"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one COCO person class, got {matches}")
    return matches[0]


def _highres_person_proposals(frame, objects, config, device):
    started = time.perf_counter()
    cls = _person_class_id(objects)
    result = objects.model.predict(
        frame, classes=[cls], conf=float(config["person_highres_conf"]),
        iou=float(config["person_nms_iou"]), imgsz=int(config["person_highres_imgsz"]),
        device=device, verbose=False,
    )[0]
    proposals = []
    if result.boxes is not None:
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            clipped = clip_box(box, frame.shape[1], frame.shape[0], min_size=8.)
            if clipped is not None:
                proposals.append({"bbox_xyxy": clipped, "confidence": float(confidence),
                                  "source": "coco_person_fullframe_highres"})
    return proposals, (time.perf_counter()-started)*1000


def _unique_center_owner(box, people, near_ratio=.08):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    candidates = []
    for p in people:
        a, b, c, d = map(float, p["bbox_xyxy"])
        pw, ph = max(1., c-a), max(1., d-b)
        inside = a <= cx <= c and b <= cy <= d
        dx, dy = max(a-cx, 0., cx-c), max(b-cy, 0., cy-d)
        rel = math.hypot(dx/pw, dy/ph)
        if inside or rel <= near_ratio:
            candidates.append((p, 0. if inside else rel))
    if len(candidates) == 1:
        return candidates[0][0]["detection_id"], "ASSOCIATED_GEOMETRICALLY", [candidates[0][0]["detection_id"]]
    if candidates:
        return None, "UNCERTAIN", [p["detection_id"] for p, _ in candidates]
    return None, "UNASSOCIATED", []


def prepare_objects_v4(frame, objects, *, config, device="cpu"):
    """Reuse V3 COCO/MediaPipe pass, adding NMS-controlled 960px person proposals.

    Tiled inference is deliberately not used in production: the V4 audit trial
    on three classroom frames returned broad partial-person boxes. It remains a
    benchmark option to revisit with a validated tile-merging/tracker strategy.
    """
    base = objects.predict(frame)
    h, w = frame.shape[:2]
    people = [dict(p) for p in base.get("person_detections", [])]
    for p in people:
        p.setdefault("sources", ["coco_person_640"])
        p["person_quality"] = _box_quality(p["bbox_xyxy"], p.get("confidence", 0.), w, h,
                                           config, "coco_person_640")
    proposals, elapsed_ms = _highres_person_proposals(frame, objects, config, device)
    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    accepted_highres = []
    supplemental = 0
    for proposal in proposals:
        # Global high-resolution NMS for any duplicate class detections.
        if any(_iou(proposal["bbox_xyxy"], old["bbox_xyxy"]) >= float(config["person_nms_iou"])
               for old in accepted_highres):
            continue
        accepted_highres.append(proposal)
        match = max(people, key=lambda p: max(_iou(proposal["bbox_xyxy"], p["bbox_xyxy"]),
                                               _intersection_over_smaller(proposal["bbox_xyxy"], p["bbox_xyxy"])), default=None)
        overlap = max(_iou(proposal["bbox_xyxy"], match["bbox_xyxy"]),
                      _intersection_over_smaller(proposal["bbox_xyxy"], match["bbox_xyxy"])) if match else 0.
        if match is not None and (overlap >= float(config["person_cross_scale_match_iou"])
                                  or _intersection_over_smaller(proposal["bbox_xyxy"], match["bbox_xyxy"]) >= .72):
            match.setdefault("sources", ["coco_person_640"])
            match["sources"].append("coco_person_fullframe_highres")
            match["highres_match_iou"] = round(overlap, 4)
            continue
        quality = _box_quality(proposal["bbox_xyxy"], proposal["confidence"], w, h,
                               config, "coco_person_fullframe_highres")
        if quality["quality"] == "coarse":
            proposal["person_quality"] = quality
            proposal["person_quality"]["rejected_as_supplement"] = True
            continue
        supplement_min_conf = float(config.get("person_highres_supplement_min_conf", 0.35))
        if float(proposal.get("confidence", 0.0)) < supplement_min_conf:
            proposal["person_quality"] = quality
            proposal["person_quality"]["rejected_as_supplement"] = True
            proposal["person_quality"]["rejection_reason"] = "below_person_highres_supplement_min_conf"
            continue
        proposal.update({
            "detection_id": f"v4_person_{supplemental}",
            "class_name": "person", "class_id": _person_class_id(objects),
            "sources": ["coco_person_fullframe_highres"],
            "person_quality": quality, "highres_only": True,
        })
        people.append(proposal)
        supplemental += 1
    base["person_detections"] = people

    # Reassociate each phone/food against the expanded person set. Any box that
    # overlaps multiple students remains ambiguous rather than being copied.
    phone_associations = []
    phone_owner = {}
    for phone in base.get("phone_detections", []):
        owner, status, candidates = _unique_center_owner(phone["bbox_xyxy"], people)
        phone_owner[phone["detection_id"]] = owner
        phone_associations.append({"phone_detection_id": phone["detection_id"],
                                   "person_detection_id": owner, "status": status,
                                   "candidate_person_ids": candidates,
                                   "candidate_count": len(candidates)})
    food_associations = []
    food_owner = {}
    for food in base.get("food_detections", []):
        owner, status, candidates = _unique_center_owner(food["bbox_xyxy"], people, near_ratio=0.)
        food_owner[food["detection_id"]] = owner
        food_associations.append({"food_detection_id": food["detection_id"],
                                  "person_detection_id": owner,
                                  "status": status if status != "UNASSOCIATED" else "UNASSOCIATED",
                                  "candidate_person_ids": candidates})
    base["associations"] = phone_associations
    base["food_person_associations"] = food_associations

    # Rebuild direct phone/hand contact using person ownership after high-res
    # augmentation. Keep the same strict two-fingertip-inside-phone criterion.
    hand_pairs = []
    interaction_evidence = []
    for phone in base.get("phone_detections", []):
        px1, py1, px2, py2 = map(float, phone["bbox_xyxy"])
        diag = max(1., math.hypot(px2-px1, py2-py1))
        owner = phone_owner.get(phone["detection_id"])
        contact_count = 0
        for hand in base.get("hand_detections", []):
            tips = hand.get("fingertips_xy", [])
            inside = sum(px1 <= float(x) <= px2 and py1 <= float(y) <= py2 for x, y in tips)
            distances = [math.hypot(max(px1-x, 0., x-px2), max(py1-y, 0., y-py2))/diag for x, y in tips]
            contact = bool(owner is not None and hand.get("person_association_status") == "ASSOCIATED_GEOMETRICALLY"
                           and hand.get("person_association") == owner and inside >= 2)
            contact_count += int(contact)
            hand_pairs.append({"phone_detection_id": phone["detection_id"],
                               "hand_detection_id": hand.get("detection_id"),
                               "phone_person_detection_id": owner,
                               "person_detection_id": hand.get("person_association"),
                               "hand_person_status": hand.get("person_association_status"),
                               "fingertip_landmarks_inside_phone": inside,
                               "nearest_fingertip_to_phone_normalized": min(distances) if distances else None,
                               "contact_evidence": contact,
                               "hand_phone_status": "CONTACT_GEOMETRY" if contact else "NO_CONTACT_EVIDENCE"})
        interaction_evidence.append({"phone_detection_id": phone["detection_id"],
                                     "person_detection_id": owner,
                                     "status": "USING_PHONE_EVIDENCE" if contact_count else "UNCERTAIN",
                                     "qualifying_hand_contact_count": contact_count})
    base["hand_phone_associations"] = hand_pairs
    base["interaction_evidence"] = interaction_evidence
    base["person_detection_audit"] = {
        "baseline_640_person_count": len([p for p in people if "coco_person_640" in p.get("sources", [])]),
        "highres_960_nms_proposal_count": len(accepted_highres),
        "highres_supplemental_count": supplemental,
        "highres_inference_ms": round(elapsed_ms, 3),
        "tile_inference_enabled": False,
        "tile_trial_note": "partial tiles made broad boxes in the sampled video frames",
    }
    return base


def spatial_v4(frame, detections, object_evidence, face_extractor, config,
               spatial_fusion_fn, spatial_config):
    base = spatial_v3(frame, detections, object_evidence, face_extractor, config,
                      spatial_fusion_fn, spatial_config)
    quality = {p["detection_id"]: p.get("person_quality", {})
               for p in object_evidence.get("person_detections", [])}
    for person in base["people"]:
        pid = person["person_ref"]
        pq = quality.get(pid, {})
        person["person_detection_quality"] = pq
        person["person_sources"] = next((p.get("sources", []) for p in object_evidence.get("person_detections", [])
                                          if p.get("detection_id") == pid), [])
        for behavior in BEHAVIORS:
            dec = person["behaviors"][behavior]
            ev = dec["evidence"]
            if behavior == "sleeping":
                face = person.get("evidence", {}).get("face") or {}
                pose = person.get("evidence", {}).get("pose_landmarks", {})
                wrists = {k: [float(v["x"]), float(v["y"])] for k, v in pose.items()
                          if k in ("left_wrist", "right_wrist") and (v.get("visibility") or 0.) >= .45}
                ears = [face.get("ear_left"), face.get("ear_right")]
                ears = [float(x) for x in ears if x is not None]
                ear = sum(ears)/len(ears) if ears and face.get("eyes_valid") else None
                ev.update({"eye_valid": bool(face.get("eyes_valid")), "ear_eye_opening": ear,
                           "eyes_open_candidate": bool(ear is not None and ear >= float(config["sleeping_eye_open_threshold"])),
                           "head_pitch_deg": (face.get("head_pose") or {}).get("pitch_deg"),
                           "wrist_positions_frame": wrists,
                           "reading_writing_observable": False,
                           "missing_evidence": ["motion history required to assess writing/reading"] if len(wrists) < 1 else []})
            if behavior == "using_phone":
                phone_boxes = person.get("evidence", {}).get("associated_phone_boxes", [])
                contact = bool(person.get("evidence", {}).get("phone_hand_contacts", []))
                ev.update({"actual_phone_object_associated": bool(phone_boxes),
                           "same_person_phone_hand_contact": contact,
                           "missing_evidence": [] if contact else ["COCO phone box and fingertip contact not both verified"]})
                if pq.get("quality") == "coarse" and dec["state"] == "detected":
                    dec.update({"state": "suspected", "status": "SUSPECTED", "decision": "suspected",
                                "decision_reason": "person_box_too_coarse_to_assign_phone_interaction",
                                "reason": "person_box_too_coarse_to_assign_phone_interaction", "bounding_box": None})
                elif not contact and not phone_boxes and dec.get("model_confidence") is not None:
                    score = float(dec["model_confidence"])
                    if score > float(config["reporting_model_confidence_gt"]):
                        dec.update({"state": "unknown", "status": "UNKNOWN", "decision": "unknown",
                                    "decision_reason": "high_phone_model_score_without_phone_object_or_contact",
                                    "reason": "high_phone_model_score_without_phone_object_or_contact", "bounding_box": None})
                        dec["contradicting_evidence"].append("no_independent_phone_object_or_hand_contact")
                    elif score >= float(config["candidate_model_confidence_min"]):
                        dec.update({"state": "suspected", "status": "SUSPECTED", "decision": "suspected",
                                    "decision_reason": "phone_model_candidate_without_independent_phone_object_or_contact",
                                    "reason": "phone_model_candidate_without_independent_phone_object_or_contact", "bounding_box": None})
                elif phone_boxes and not contact:
                    dec.update({"state": "unknown", "status": "UNKNOWN", "decision": "unknown",
                                "decision_reason": "phone_object_present_but_interaction_not_verified",
                                "reason": "phone_object_present_but_interaction_not_verified", "bounding_box": None})
            if behavior == "turning_back" and config.get("turn_rear_yaw_target_deg") is None:
                ev["head_turn_candidate"] = bool(ev.get("head_turn_candidate") or ev.get("head_yaw_deg") is not None)
                ev["body_turn_candidate"] = bool(ev.get("body_turn_candidate"))
                ev["rear_direction_calibrated"] = False
                ev["missing_evidence"] = ["camera-relative classroom rear direction calibration"]
                dec.update({"state": "unknown", "status": "UNKNOWN", "decision": "unknown",
                            "decision_reason": "rear_direction_uncalibrated", "reason": "rear_direction_uncalibrated",
                            "bounding_box": None})
            if pq.get("quality") == "coarse" and dec["state"] == "detected":
                dec.update({"state": "suspected", "status": "SUSPECTED", "decision": "suspected",
                            "decision_reason": "person_box_too_coarse_for_person_level_confirmation",
                            "reason": "person_box_too_coarse_for_person_level_confirmation", "bounding_box": None})
                dec["contradicting_evidence"].append("person_box_exceeds_configured_geometry_limits")
            dec["person_id"] = pid
            dec["behavior"] = behavior
    return base


def debug_sidecar(frame, rows, *, panel_width=520, max_rows=28):
    """Append an evidence panel beside the untouched-size debug frame."""
    h, w = frame.shape[:2]
    panel = np.full((h, panel_width, 3), 28, dtype=np.uint8)
    import cv2
    cv2.putText(panel, "HYBRID V4 EVIDENCE", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, "score = raw model output, not probability", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, .30, (180, 180, 180), 1, cv2.LINE_AA)
    y = 64
    for row in rows[:max_rows]:
        color = row.get("color", (225, 225, 225))
        for text in row.get("lines", []):
            if y > h-6: break
            cv2.putText(panel, str(text)[:95], (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .29, color, 1, cv2.LINE_AA)
            y += 15
        y += 3
        if y > h-6: break
    cv2.putText(panel, "Full evidence: JSON/CSV sidecar", (12, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, .30, (170, 170, 170), 1, cv2.LINE_AA)
    return np.concatenate([frame, panel], axis=1)


def debug_evidence_rows(people, behavior_colors):
    rows=[]
    state_rank={"detected":0,"suspected":1,"unknown":2,"not_detected":3}
    for person in people:
        pid=person.get("person_id",person.get("person_ref"))
        for behavior, dec in person.get("behaviors",{}).items():
            ev=dec.get("evidence",{})
            score=dec.get("model_confidence")
            support=dec.get("supporting_evidence",[])
            missing=ev.get("missing_evidence",dec.get("contradicting_evidence",[]))
            lines=[f"{pid} {behavior} {dec.get('state')} raw={score:.3f}" if score is not None
                   else f"{pid} {behavior} {dec.get('state')} raw=none",
                   "+"+(",".join(map(str,support)) if support else "none")+
                   "  -"+(",".join(map(str,missing)) if missing else "none"),
                   "reason: "+str(dec.get("decision_reason",dec.get("reason","")))]
            rows.append({"priority":(state_rank.get(dec.get("state"),4), -(score or 0.)),
                         "lines":lines,"color":behavior_colors.get(behavior,(220,220,220))})
    return sorted(rows,key=lambda x:x["priority"])


class TemporalV4Fusion:
    """V3 timestamps plus observational eye/wrist context, without a sleep gate."""
    def __init__(self, config):
        self._impl = TemporalV3Fusion(config)
        self.cfg = config
        self.sleeping_history = defaultdict(lambda: deque(maxlen=64))

    @property
    def events(self): return self._impl.events

    @property
    def identity_discontinuities(self): return self._impl.identity_discontinuities

    def update(self, track_id, timestamp, behaviors, person_box=None):
        tid = int(track_id)
        sleep = dict(behaviors["sleeping"])
        evidence = dict(sleep.get("evidence", {}))
        box_height = max(1., float(person_box[3])-float(person_box[1])) if person_box else 1.
        wrists = evidence.get("wrist_positions_frame", {})
        hist = self.sleeping_history[tid]
        hist.append({"t": float(timestamp), "wrists": wrists,
                     "eyes_open": bool(evidence.get("eyes_open_candidate")),
                     "eye_valid": bool(evidence.get("eye_valid"))})
        window = float(self.cfg["sleeping_alternative_window_sec"])
        recent = [x for x in hist if timestamp-x["t"] <= window]
        valid_open = [x for x in recent if x["eye_valid"] and x["eyes_open"]]
        open_persistent = (len(valid_open) >= int(self.cfg["sleeping_open_eye_min_samples"])
                           and valid_open[-1]["t"]-valid_open[0]["t"] >= window*.60)
        motion = 0.
        if len(recent) >= 3:
            for left, right in zip(recent, recent[1:]):
                for wrist in ("left_wrist", "right_wrist"):
                    a, b = left["wrists"].get(wrist), right["wrists"].get(wrist)
                    if a and b:
                        motion += math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))/box_height
        writing_motion = len(recent) >= 4 and motion >= float(self.cfg["sleeping_wrist_path_person_height"])
        # Eye openness and wrist motion remain diagnostics only. They must not
        # downgrade the model decision, clear sleep history, or end an event.
        evidence.update({"wrist_path_person_height": round(motion, 5),
                         "persistent_open_eyes": open_persistent,
                         "writing_motion_candidate": writing_motion,
                         "awake_alternative_blocks_sleep_confirmation": False})
        sleep["evidence"] = evidence
        inputs = dict(behaviors); inputs["sleeping"] = sleep
        return self._impl.update(tid, timestamp, inputs, person_box)

    def expire(self, timestamp): return self._impl.expire(timestamp)
