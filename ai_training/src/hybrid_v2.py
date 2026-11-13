"""Evidence-first V2 spatial and timestamp-based temporal behavior fusion.

The v1.4 checkpoint remains an unchanged source of model scores. Scores are
kept separate from evidence and decisions; they are never treated as calibrated
probabilities. The V2 module adds per-person ROI proposals and explicit
cross-behavior conflict handling.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
import math
import cv2
import numpy as np


BEHAVIORS = ("sleeping", "using_phone", "turning_back", "eating")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.


def infer_global_and_rois(model, frame, object_evidence, names, *, imgsz=640,
                          conf=.05, iou=.50, device="cpu", roi_enabled=True,
                          padding=.12, dedupe_iou=.65):
    """Run the full-frame detector and person crops, map boxes back and dedupe.

    Crop detections carry `person_ref`, so fusion does not have to guess which
    student produced an ROI detection. Full-frame inference preserves context.
    """
    global_result = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou,
                                  device=device, verbose=False)[0]
    raw = []
    if global_result.boxes is not None:
        for box, score, cls in zip(global_result.boxes.xyxy.cpu().tolist(),
                                   global_result.boxes.conf.cpu().tolist(),
                                   global_result.boxes.cls.cpu().tolist()):
            raw.append({"bbox_xyxy": [round(float(x), 2) for x in box],
                        "confidence": float(score), "class_id": int(cls),
                        "class_name": names[int(cls)], "source": "full_frame"})
    people = object_evidence.get("person_detections", [])
    for det in raw:
        overlaps=sorted(((_iou(det["bbox_xyxy"],p["bbox_xyxy"]),p.get("detection_id")) for p in people),reverse=True)
        if overlaps and overlaps[0][0]>=.15:
            second=overlaps[1][0] if len(overlaps)>1 else 0.
            if second>=.15 and overlaps[0][0]-second<.05:
                det["association_ambiguous"]=True
                det["person_candidate_ids"]=[x[1] for x in overlaps[:2]]
            else:
                det["person_ref"]=overlaps[0][1]
    roi_raw = []
    if roi_enabled:
        h, w = frame.shape[:2]
        crops, origins = [], []
        for person in object_evidence.get("person_detections", []):
            x1, y1, x2, y2 = map(float, person.get("bbox_xyxy", (0, 0, 0, 0)))
            pw, ph = x2-x1, y2-y1
            if pw < 24 or ph < 32:
                continue
            dx, dy = pw*padding, ph*padding
            ax1, ay1 = max(0, int(x1-dx)), max(0, int(y1-dy))
            ax2, ay2 = min(w, int(x2+dx)), min(h, int(y2+dy))
            if ax2-ax1 < 24 or ay2-ay1 < 32:
                continue
            crops.append(frame[ay1:ay2, ax1:ax2])
            origins.append((ax1, ay1, person.get("detection_id")))
        # Batching avoids one model invocation per student.
        for start in range(0, len(crops), 16):
            batch, batch_origins = crops[start:start+16], origins[start:start+16]
            results = model.predict(batch, imgsz=imgsz, conf=conf, iou=iou,
                                    device=device, verbose=False, stream=False)
            for crop_result, (ox, oy, person_ref) in zip(results, batch_origins):
                if crop_result.boxes is None:
                    continue
                for box, score, cls in zip(crop_result.boxes.xyxy.cpu().tolist(),
                                           crop_result.boxes.conf.cpu().tolist(),
                                           crop_result.boxes.cls.cpu().tolist()):
                    mapped = [round(float(box[0]+ox), 2), round(float(box[1]+oy), 2),
                              round(float(box[2]+ox), 2), round(float(box[3]+oy), 2)]
                    roi_raw.append({"bbox_xyxy": mapped, "confidence": float(score),
                                    "class_id": int(cls), "class_name": names[int(cls)],
                                    "source": "person_roi", "person_ref": person_ref})
    # Keep one best same-class proposal when whole-frame and ROI boxes overlap.
    merged = []
    for det in sorted(raw + roi_raw, key=lambda d: d["confidence"], reverse=True):
        duplicate = next((old for old in merged if old["class_name"] == det["class_name"]
                          and not old.get("association_ambiguous") and not det.get("association_ambiguous")
                          and not (old.get("person_ref") is not None and det.get("person_ref") is not None
                                   and old.get("person_ref") != det.get("person_ref"))
                          and _iou(old["bbox_xyxy"], det["bbox_xyxy"]) >= dedupe_iou), None)
        if duplicate is None:
            merged.append(det)
        elif det.get("person_ref") is not None and duplicate.get("person_ref") is None:
            duplicate["person_ref"] = det["person_ref"]
            duplicate["sources"] = [duplicate.get("source"), det.get("source")]
        else:
            duplicate.setdefault("sources", [duplicate.get("source")])
            duplicate["sources"].append(det.get("source"))
    return {"detections": merged, "baseline_full_frame": raw,
            "roi_raw": roi_raw, "roi_enabled": bool(roi_enabled),
            "roi_person_count": len(crops) if roi_enabled else 0,
            "deduped_count": len(raw)+len(roi_raw)-len(merged)}


def _distance_point_box(point, box):
    x, y = point
    dx = max(box[0]-x, 0., x-box[2])
    dy = max(box[1]-y, 0., y-box[3])
    return math.hypot(dx, dy)


def _distance(a, b):
    return math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))


def spatial_v2(frame, detections, object_evidence, face_extractor, config,
               spatial_fusion_fn, spatial_config):
    """Use shared V1.4 feature extraction, then resolve evidence conflicts."""
    base = spatial_fusion_fn(frame, detections, object_evidence,
                             face_extractor, spatial_config)
    phone_gate = float(config["reporting_model_confidence_gt"])
    candidate_floor = float(config.get("candidate_model_confidence_min", .25))
    for person in base["people"]:
        ev = person["evidence"]
        validity = person["evidence_validity"]
        x1, y1, x2, y2 = map(float, person["bbox_xyxy"])
        height = max(1., y2-y1)
        face = ev.get("face") or {}
        mouth = face.get("mouth_center_xy_frame")
        foods = ev.get("associated_food_boxes", [])
        hands = ev.get("associated_hands", [])
        phones = ev.get("associated_phone_boxes", [])
        contacts = ev.get("phone_hand_contacts", [])
        # MediaPipe hands carry finger tips; wrist-only checks miss eating bites.
        points = [p for hand in hands for p in hand.get("fingertips_xy", [])]
        wrists = [hand.get("wrist_xy") for hand in hands if hand.get("wrist_xy")]
        food_mouth_pairs = []
        food_hand_pairs = []
        for food in foods:
            fb = list(map(float, food["bbox_xyxy"]))
            if mouth is not None:
                dm = _distance_point_box(mouth, fb) / height
                if dm <= float(config["food_mouth_distance_person_height"]):
                    food_mouth_pairs.append({"food_detection_id": food.get("detection_id"),
                                             "mouth_distance_person_height": dm})
            for tip in points:
                dh = _distance_point_box(tip, fb) / height
                if dh <= float(config["food_hand_distance_person_height"]):
                    food_hand_pairs.append({"food_detection_id": food.get("detection_id"),
                                            "fingertip_food_distance_person_height": dh})
                    break
        finger_mouth = min((_distance(tip, mouth)/height for tip in points), default=None) if mouth else None
        wrist_mouth = min((_distance(wrist, mouth)/height for wrist in wrists), default=None) if mouth else None
        mouth_ratio = face.get("mouth_open_ratio")
        eating_context = bool(food_mouth_pairs and food_hand_pairs)
        eating_candidate = bool(eating_context or (finger_mouth is not None and finger_mouth <= .16
                                 and mouth_ratio is not None and mouth_ratio >= .12))
        if eating_context:
            eat_state, eat_reason = "detected", "person_associated_food_near_mouth_and_fingertip_interaction"
        elif eating_candidate:
            eat_state, eat_reason = "suspected", "hand_mouth_or_food_interaction_candidate_without_food_specific_confirmation"
        else:
            eat_state, eat_reason = "unknown", "no_food_mouth_interaction_evidence"
        person["behaviors"]["eating"] = {
            "state": eat_state, "status": "IMAGE_ONLY" if eat_state == "detected" else eat_state.upper(),
            "model_confidence": None, "fusion_score": None, "decision": eat_state,
            "decision_reason": eat_reason,
            "evidence": {"food_associated_with_person": bool(foods),
                         "food_near_mouth": food_mouth_pairs,
                         "fingertip_near_food": food_hand_pairs,
                         "fingertip_mouth_distance_person_height": finger_mouth,
                         "wrist_mouth_distance_person_height": wrist_mouth,
                         "mouth_open_ratio": mouth_ratio,
                         "food_without_interaction_is_not_eating": True,
                         "drinking_not_assumed_eating": True}}
        # Phone requires direct phone-hand evidence or persistent video model
        # evidence (temporal module). Merely having a phone on the desk is weak.
        phone_score = max((float(d["confidence"]) for d in person["yolo_detections"]
                           if d["class_name"] == "using_phone"), default=None)
        phone_interaction = bool(contacts)
        food_conflict = eating_candidate
        if eating_context:
            phone_state, phone_reason = "unknown", "using_phone_suppressed_by_strong_eating_object_interaction_conflict"
        elif phone_interaction and not food_conflict:
            phone_state, phone_reason = "detected", "associated_coco_phone_and_fingertip_contact"
        elif phone_score is not None and phone_score >= candidate_floor:
            phone_state, phone_reason = "suspected", "behavior_model_candidate_without_verified_phone_interaction"
        elif phones and not validity.get("hands"):
            phone_state, phone_reason = "unknown", "phone_observed_but_interaction_not_observable"
        elif phones:
            phone_state, phone_reason = "unknown", "phone_present_without_hand_contact_phone_on_desk_not_usage"
        elif food_conflict:
            phone_state, phone_reason = "unknown", "phone_candidate_conflicts_with_eating_cues"
        else:
            phone_state, phone_reason = "unknown", "phone_not_observed_does_not_prove_not_using_phone"
        person["behaviors"]["using_phone"] = {
            "state": phone_state, "status": "IMAGE_ONLY" if phone_state == "detected" else phone_state.upper(),
            "model_confidence": phone_score, "fusion_score": None, "decision": phone_state,
            "decision_reason": phone_reason,
            "evidence": {"yolo_model_confidence": phone_score,
                         "model_confidence_reporting_gate_gt": phone_gate,
                         "associated_coco_phone_count": len(phones),
                         "associated_phone_hand_contacts": len(contacts),
                         "hands_visible": bool(validity.get("hands")),
                         "eating_conflict": food_conflict,
                         "yolo_score_is_not_calibrated_probability": True}}
        # Preserve model score independently from the decision for sleeping.
        sleep_score = max((float(d["confidence"]) for d in person["yolo_detections"]
                           if d["class_name"] == "sleeping"), default=None)
        ear = face.get("ear_left")
        if ear is not None and face.get("ear_right") is not None:
            ear = (float(ear)+float(face["ear_right"]))/2
        eye_closed = bool(ear is not None and ear <= float(config["sleeping_ear_closed"]))
        pose = ev.get("pose_landmarks", {})
        visible = {k:v for k,v in pose.items() if (v.get("visibility") or 0) >= .35}
        head_supported = False
        # Head-to-table support proxy only when nose and both wrists are visible.
        nose = visible.get("nose"); lw = visible.get("left_wrist"); rw = visible.get("right_wrist")
        if nose and (lw or rw):
            head_supported = any(_distance((nose["x"],nose["y"]),(w["x"],w["y"])) / height
                < float(config.get("sleeping_head_support_distance_person_height", .14))
                for w in (lw,rw) if w)
        shoulder_tilt=None
        ls,rs=visible.get("left_shoulder"),visible.get("right_shoulder")
        if ls and rs:
            shoulder_tilt=abs(math.degrees(math.atan2(float(rs["y"])-float(ls["y"]),float(rs["x"])-float(ls["x"]))))
            shoulder_tilt=min(shoulder_tilt,180.-shoulder_tilt)
        hp=face.get("head_pose",{})
        head_roll=abs(float(hp["roll_deg"])) if hp.get("valid") and hp.get("roll_deg") is not None else None
        tilt_min=float(config.get("sleeping_tilt_min_deg",15.))
        tilt_diff=float(config.get("sleeping_head_body_tilt_difference_deg",20.))
        head_body_tilt=bool(head_roll is not None and shoulder_tilt is not None and head_roll>=tilt_min
            and shoulder_tilt>=tilt_min and abs(head_roll-shoulder_tilt)<=tilt_diff)
        sleep_score_high = sleep_score is not None and sleep_score > float(config["sleeping_high_model_score"])
        # Eye/pose measurements are context only, never a sleeping gate.
        if sleep_score_high:
            sleep_state, sleep_reason = "detected", "behavior_yolo_sleeping_threshold_met"
        elif sleep_score is not None and sleep_score >= candidate_floor:
            sleep_state, sleep_reason = "suspected", "behavior_yolo_sleeping_below_detection_threshold"
        else:
            sleep_state, sleep_reason = "not_detected", "no_qualifying_sleeping_model_candidate"
        person["behaviors"]["sleeping"] = {
            "state": sleep_state, "status": "IMAGE_ONLY" if sleep_state == "detected" else sleep_state.upper(),
            "model_confidence": sleep_score, "fusion_score": None, "decision": sleep_state,
            "decision_reason": sleep_reason,
            "evidence": {"yolo_model_confidence": sleep_score, "ear": ear,
                         "ear_semantics": "eye_opening_measurement_not_sleep_probability",
                         "eye_valid": bool(validity.get("eyes")), "eye_closed": eye_closed,
                         "head_supported_proxy": head_supported,"head_roll_deg":head_roll,
                         "shoulder_tilt_deg":shoulder_tilt,"head_body_tilt":head_body_tilt,
                         "pose_valid": bool(validity.get("pose")),
                         "head_down_alone_used": False}}
        # No rear-facing calibration -> never promote a side glance to turning_back.
        hp = face.get("head_pose", {})
        yaw = hp.get("yaw_deg") if hp.get("valid") else None
        calibrated = config.get("turn_rear_yaw_target_deg") is not None
        head_turn_candidate=bool(yaw is not None and abs(float(yaw))>=45.)
        body_turn_candidate=bool(shoulder_tilt is not None and shoulder_tilt>=35.)
        person["behaviors"]["turning_back"] = {
            "state": "unknown", "status": "UNKNOWN", "model_confidence": None,
            "fusion_score": None, "decision": "unknown",
            "decision_reason": "rear_direction_not_calibrated" if not calibrated else "image_has_no_temporal_persistence",
            "evidence": {"head_yaw_deg": yaw, "rear_direction_calibrated": calibrated,
                         "pose_valid": bool(validity.get("pose")),
                         "head_turn_candidate":head_turn_candidate,"body_turn_candidate":body_turn_candidate,
                         "side_turn_alone_is_not_turning_back": True}}
        person["evidence"].setdefault("v2", {}).update({
            "eating_candidate": eating_candidate, "eating_strong": eating_context,
            "food_mouth_pairs": food_mouth_pairs, "food_hand_pairs": food_hand_pairs,
            "fingertip_mouth_distance_person_height": finger_mouth,
            "sleeping_model_confidence": sleep_score, "using_phone_model_confidence": phone_score})
    return base


class TemporalV2Fusion:
    """Timestamp-based per-track state and event lifecycle for video."""
    def __init__(self, config):
        self.cfg = config
        self.history = defaultdict(lambda: {b: deque(maxlen=512) for b in BEHAVIORS})
        self.states = {}
        self.last_seen = {}
        self.events = []
        self.event_open = {}
        self.event_records = {}
        self.last_end = {}

    def update(self, track_id, t, behaviors):
        tid = int(track_id)
        out = {}
        gap = t-self.last_seen.get(tid, t)
        reset = gap > float(self.cfg["active_grace_sec"])*2
        if reset:
            self.history[tid] = {b: deque(maxlen=512) for b in BEHAVIORS}
        self.last_seen[tid] = t
        for behavior in BEHAVIORS:
            src = behaviors[behavior]
            ev = src.get("evidence", {})
            row = {"t": float(t), "source": src, "positive": src["state"] == "detected"}
            hist = self.history[tid][behavior]
            hist.append(row)
            window = float(self.cfg.get("eating_window_sec",10.)) if behavior == "eating" else max(
                float(self.cfg.get(behavior+"_confirm_sec", 2.5))*2, 6.)
            recent = [r for r in hist if t-r["t"] <= window]
            state, reason = src["state"], src.get("decision_reason", "spatial_evidence")
            duration = float(self.cfg.get(behavior+"_confirm_sec", 1.75))
            positive = [r for r in recent if r["positive"]]
            sustained = len(positive) >= 3 and positive[-1]["t"]-positive[0]["t"] >= duration \
                and t-positive[-1]["t"] <= float(self.cfg["active_grace_sec"])
            if behavior == "sleeping":
                # Keep persistence checks, without requiring visible closed eyes or pose.
                valid = [r for r in recent if r["positive"]]
                times=[r["t"] for r in valid]
                sustained = len(valid) >= 3 and times[-1]-times[0] >= duration \
                    and t-times[-1] <= float(self.cfg["active_grace_sec"]) \
                    and max((b-a for a,b in zip(times,times[1:])),default=0.) <= float(self.cfg["active_grace_sec"])
                if sustained:
                    state, reason = "detected", "persistent_behavior_yolo_sleeping_candidate"
                elif src["state"] == "detected": state, reason = "suspected", "spatial_sleep_candidate_waiting_for_persistence"
            elif behavior == "using_phone":
                valid = [r for r in recent if r["positive"] and
                         r["source"].get("decision_reason") == "associated_coco_phone_and_fingertip_contact"]
                times=[r["t"] for r in valid]
                sustained = len(valid) >= 3 and times[-1]-times[0] >= duration \
                    and t-times[-1] <= float(self.cfg["active_grace_sec"]) \
                    and max((b-a for a,b in zip(times,times[1:])),default=0.) <= float(self.cfg["active_grace_sec"])
                # Persistent high detector outputs may confirm only if there is
                # no eating conflict and a phone object/contact is supported.
                if sustained: state, reason = "detected", "persistent_person_phone_fingertip_interaction"
                elif src["state"] == "detected": state, reason = "suspected", "phone_interaction_candidate_waiting_for_persistence"
            elif behavior == "turning_back":
                target = self.cfg.get("turn_rear_yaw_target_deg")
                yaw = ev.get("head_yaw_deg")
                hit = target is not None and yaw is not None and abs(float(yaw)-float(target)) <= float(self.cfg["turn_rear_yaw_tolerance_deg"])
                valid = [r for r in recent if r["source"].get("evidence",{}).get("rear_direction_calibrated") and
                         r["source"].get("evidence",{}).get("head_yaw_deg") is not None and
                         abs(float(r["source"]["evidence"]["head_yaw_deg"])-float(target)) <= float(self.cfg["turn_rear_yaw_tolerance_deg"])] if target is not None else []
                sustained = len(valid) >= 3 and valid[-1]["t"]-valid[0]["t"] >= duration
                state, reason = ("detected","persistent_head_orientation_toward_calibrated_rear") if sustained else ("unknown","rear_direction_not_calibrated_or_not_sustained")
            elif behavior == "eating":
                # IMAGE has no cycle assumption. VIDEO requires repeated
                # evidence events; drink/talk cues remain unresolved.
                active = [bool(r["source"].get("evidence",{}).get("food_near_mouth") and
                               r["source"].get("evidence",{}).get("fingertip_near_food")) for r in recent]
                cycles = sum(on and (i == 0 or not active[i-1]) for i,on in enumerate(active))
                if cycles >= int(self.cfg.get("eating_min_cycles",2)):
                    state, reason = "detected", "repeated_food_to_mouth_interaction_events"
                elif src["state"] == "detected": state, reason = "suspected", "eating_event_waiting_for_repeated_cycle"
            key = (tid, behavior)
            previous = self.states.get(key, "NORMAL")
            status = "CONFIRMED" if state == "detected" else "SUSPECTED" if state == "suspected" else "NORMAL"
            transition = previous != status
            ended_at = None
            if status == "CONFIRMED" and previous != "CONFIRMED":
                self.event_open[key] = float(t)
                event_record={"track_id":tid,"behavior":behavior,"start_sec":float(t),
                              "end_sec":None,"duration_sec":None,"reason":reason}
                self.event_records[key]=event_record
            elif previous == "CONFIRMED" and status != "CONFIRMED":
                ended_at = float(t)
                start = self.event_open.pop(key, float(t))
                self.last_end[key] = float(t)
                event_record=self.event_records.pop(key,None)
                if event_record is not None:
                    event_record.update({"end_sec":float(t),"duration_sec":max(0.,float(t)-start),"end_reason":"evidence_lost"})
            cooldown_until = self.last_end.get(key, -1e9)+float(self.cfg["cooldown_sec"])
            in_cooldown = t < cooldown_until
            event = status == "CONFIRMED" and previous != "CONFIRMED" and not in_cooldown
            if event:
                self.events.append(event_record)
            self.states[key] = status
            out[behavior] = {"state":state,"status":status,"model_confidence":src.get("model_confidence"),
                "fusion_score":None,"decision":state,"decision_reason":reason,
                "evidence":{**ev,"spatial_state":src["state"],"timestamp_sec":round(float(t),3),
                    "track_id":tid,"track_gap_sec":round(gap,3),"history_reset":reset,
                    "confirmation_duration_sec":duration,"active_grace_sec":self.cfg["active_grace_sec"],
                    "status_transition":f"{previous}->{status}" if transition else "steady",
                    "event_start_sec":self.event_open.get(key),"ended_at_sec":ended_at,
                    "cooldown_until_sec":cooldown_until if cooldown_until>-1e8 else None,
                    "confirmation_event":bool(event)}}
        return out

    def expire(self, t):
        """Close active alerts after track/evidence loss exceeds the grace."""
        grace=float(self.cfg["active_grace_sec"])
        for tid,last in list(self.last_seen.items()):
            if t-last<=grace:continue
            for behavior in BEHAVIORS:
                key=(tid,behavior)
                if self.states.get(key)!="CONFIRMED":continue
                self.states[key]="NORMAL"
                start=self.event_open.pop(key,float(last))
                event=self.event_records.pop(key,None)
                self.last_end[key]=float(t)
                if event is not None:
                    event.update({"end_sec":float(t),"duration_sec":max(0.,float(t)-start),
                                  "end_reason":"track_or_evidence_lost"})
