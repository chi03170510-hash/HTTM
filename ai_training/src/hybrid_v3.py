"""Hybrid V3: conservative, person-scoped evidence and explainable decisions.

V1.4 remains the behavior model. This module wraps the already audited V2
detectors/fusion and adds safer proposal ownership, decision schema, and
independent conflict handling. It does not learn or calibrate probabilities.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from src.hybrid_v2 import BEHAVIORS, infer_global_and_rois, spatial_v2


def clip_box(box, width: int, height: int, min_size: float = 2.0):
    """Clip xyxy to frame bounds; return None for malformed/collapsed boxes."""
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, x2 = max(0., min(width - 1., x1)), max(0., min(width - 1., x2))
    y1, y2 = max(0., min(height - 1., y1)), max(0., min(height - 1., y2))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def map_roi_box(box, origin_xy, frame_shape):
    """Map crop-local xyxy into source frame, with clipping/validity checking."""
    h, w = frame_shape[:2]
    ox, oy = map(float, origin_xy)
    local = list(map(float, box))
    return clip_box([local[0] + ox, local[1] + oy,
                     local[2] + ox, local[3] + oy], w, h)


def associate_detection_to_person(det_box, people, min_containment=.60,
                                  ambiguity_margin=.12):
    """Associate a behavior box by fraction of its area inside a person box.

    Returns an explicit ambiguous/unassigned result rather than choosing the
    nearest person in a crowded overlap. ROI detections already carry a trusted
    source person ID and should bypass this function.
    """
    x1, y1, x2, y2 = map(float, det_box)
    area = max(0., x2 - x1) * max(0., y2 - y1)
    if area <= 0 or not people:
        return {"person_ref": None, "status": "unassigned", "candidates": []}
    ranked = []
    for p in people:
        a, b, c, d = map(float, p["bbox_xyxy"])
        overlap = max(0., min(x2, c) - max(x1, a)) * max(0., min(y2, d) - max(y1, b))
        ranked.append((overlap / area, p.get("detection_id")))
    ranked.sort(reverse=True)
    top, pid = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.
    if top < min_containment:
        return {"person_ref": None, "status": "unassigned", "candidates": ranked[:2]}
    if second >= min_containment and top - second < ambiguity_margin:
        return {"person_ref": None, "status": "ambiguous", "candidates": ranked[:2]}
    return {"person_ref": pid, "status": "associated", "candidates": ranked[:2]}


def infer_v3(model, frame, object_evidence, names, **kwargs):
    """V2 inference plus V3 clipping, provenance, and safer global association."""
    result = infer_global_and_rois(model, frame, object_evidence, names, **kwargs)
    h, w = frame.shape[:2]
    people = object_evidence.get("person_detections", [])
    for det in result["detections"]:
        clipped = clip_box(det.get("bbox_xyxy", []), w, h)
        det["bbox_xyxy_unclipped"] = det.get("bbox_xyxy")
        det["bbox_xyxy"] = clipped
        det["roi_provenance"] = {
            "source": det.get("source"),
            "sources": det.get("sources", [det.get("source")]),
            "person_ref": det.get("person_ref"),
        }
        if clipped is None:
            det["association_ambiguous"] = True
            det["association_status"] = "invalid_box"
            continue
        # An ROI proposal is generated from one known person's crop. Otherwise
        # use box containment, not IoU, which penalizes a small head proposal.
        has_roi = "person_roi" in det.get("sources", []) or det.get("source") == "person_roi"
        if has_roi and det.get("person_ref") is not None:
            det["association_status"] = "associated_by_person_roi"
        else:
            # Recompute every full-frame association from containment. A previous
            # V2 IoU owner is not accepted as sufficient evidence by itself.
            assoc = associate_detection_to_person(clipped, people)
            det.update(assoc)
            det["association_ambiguous"] = assoc["status"] != "associated"
            det["association_status"] = assoc["status"]
    return result


def _decision(state, *, model_score=None, model_source=None, roi_provenance=None,
              supporting=None, contradicting=None, association_quality="unknown",
              reason="unspecified", bbox=None, evidence=None):
    return {
        # Keep lifecycle-compatible aliases alongside the requested V3 schema.
        "state": state, "status": "IMAGE_ONLY" if state == "detected" else state.upper(),
        "decision": state, "decision_reason": reason, "model_confidence": model_score,
        "model_source": model_source, "roi_provenance": roi_provenance,
        "supporting_evidence": supporting or [],
        "contradicting_evidence": contradicting or [],
        "association_quality": association_quality, "reason": reason,
        "bounding_box": bbox, "evidence": evidence or {},
    }


def spatial_v3(frame, detections, object_evidence, face_extractor, config,
               spatial_fusion_fn, spatial_config):
    """V3 decisions per person; image mode has no temporal persistence."""
    base = spatial_v2(frame, detections, object_evidence, face_extractor,
                      config, spatial_fusion_fn, spatial_config)
    h, w = frame.shape[:2]
    for person in base["people"]:
        pid = person["person_ref"]
        person["person_id"] = pid
        person_box = clip_box(person["bbox_xyxy"], w, h, min_size=8.)
        person_dets = person.get("yolo_detections", [])
        valid_dets = [d for d in person_dets if d.get("bbox_xyxy")]
        decision_rows = {}
        for behavior in BEHAVIORS:
            old = person["behaviors"][behavior]
            ev = old.get("evidence", {})
            model_dets = [d for d in valid_dets if d.get("class_name") == behavior]
            best = max(model_dets, key=lambda d: float(d["confidence"]), default=None)
            score = float(best["confidence"]) if best else old.get("model_confidence")
            source = best.get("source") if best else ("evidence_rule" if behavior in ("eating", "turning_back") else None)
            provenance = best.get("roi_provenance") if best else None
            support, contradict = [], []
            state, reason = old["state"], old.get("decision_reason", "spatial_evidence")
            evidence = dict(ev)
            if behavior == "sleeping":
                if evidence.get("eye_closed"): support.append("ear_eye_opening_indicates_closed_eye")
                if evidence.get("head_supported_proxy"): support.append("head_supported_by_visible_hand_proxy")
                if evidence.get("head_body_tilt"): support.append("aligned_head_and_shoulder_tilt")
                evidence.update({"reading_writing_state": "unobservable_from_single_frame",
                                 "head_down_is_not_sleep": True,
                                 "eye_state_required": False})
            elif behavior == "using_phone":
                contacts = person.get("evidence", {}).get("phone_hand_contacts", [])
                foods = evidence.get("food_near_mouth", [])
                food_hands = evidence.get("fingertip_near_food", [])
                phone_support = bool(contacts)
                eating_support = bool(foods and food_hands)
                if phone_support: support.append("associated_phone_and_same_person_fingertip_contact")
                evidence["associated_phone_and_same_person_fingertip_contact"] = phone_support
                if eating_support: contradict.append("food_and_hand_near_mouth_conflict_candidate")
                # Evidence is independent: eating cues do not suppress a phone
                # when a separate phone + same-person contact is present.
                if phone_support:
                    state, reason = "detected", "independent_phone_contact_evidence"
                elif eating_support and score is not None:
                    state, reason = "suspected", "phone_model_candidate_with_eating_context_no_phone_contact"
                elif score is not None and float(score) > float(config["reporting_model_confidence_gt"]):
                    state, reason = "suspected", "high_model_score_without_verified_phone_object_contact"
                if evidence.get("associated_coco_phone_count", 0) and not phone_support:
                    contradict.append("phone_present_without_interaction")
            elif behavior == "eating":
                # No eating-trained model exists. A single frame cannot establish
                # a repeated eating action, so spatial evidence remains a candidate.
                food_mouth = evidence.get("food_near_mouth", [])
                food_hand = evidence.get("fingertip_near_food", [])
                mouth_open = evidence.get("mouth_open_ratio")
                if food_mouth and food_hand:
                    support.extend(["food_box_near_mouth", "fingertip_near_associated_food"])
                if evidence.get("drinking_not_assumed_eating"):
                    contradict.append("drinking_or_other_mouth_action_not_excluded")
                if state == "detected":
                    state, reason = "suspected", "single_frame_food_hand_geometry_cannot_confirm_eating_cycle"
                evidence.update({"eating_model_available": False,
                                 "mouth_open_is_support_not_eating_probability": mouth_open})
            elif behavior == "turning_back":
                if evidence.get("head_turn_candidate"): support.append("head_turn_candidate")
                if evidence.get("body_turn_candidate"): support.append("body_turn_candidate")
                hp = (person.get("evidence", {}).get("face") or {}).get("head_pose", {})
                yaw = hp.get("yaw_deg") if hp.get("valid") else None
                target = config.get("turn_rear_yaw_target_deg")
                evidence.update({"head_yaw_deg": yaw, "rear_direction_calibrated": target is not None,
                                 "rear_yaw_target_deg": target,
                                 "rear_direction_tolerance_deg": config.get("turn_rear_yaw_tolerance_deg")})
                if target is not None and yaw is not None and abs(float(yaw)-float(target)) <= float(config.get("turn_rear_yaw_tolerance_deg", 20.)):
                    state, reason = "suspected", "head_orientation_matches_configured_rear_candidate_waiting_for_video_persistence"
                    support.append("head_orientation_matches_calibrated_rear_direction")
                elif yaw is not None and abs(float(yaw)) >= 45.:
                    state, reason = "suspected", "head_turn_candidate_rear_direction_not_matched"
                else:
                    state, reason = "unknown", "rear_direction_or_landmarks_insufficient"
            bbox = person_box if state == "detected" else None
            decision_rows[behavior] = _decision(
                state, model_score=score, model_source=source, roi_provenance=provenance,
                supporting=support, contradicting=contradict,
                association_quality=(best.get("association_status", "associated_by_roi") if best else "person_scoped_rule"),
                reason=reason, bbox=bbox, evidence=evidence)
        person["behaviors"] = decision_rows
        person["person_bbox_xyxy_clipped"] = person_box
    # Keep all proposals that did not get a trustworthy person assignment in
    # debug/evidence; they are never promoted to person-level alerts.
    return base


class TemporalV3Fusion:
    """A conservative wrapper around timestamp V2 lifecycle, with lost-ID reset."""
    def __init__(self, config):
        from src.hybrid_v2 import TemporalV2Fusion
        self._impl = TemporalV2Fusion(config)
        self.cfg = config
        self.last_identity_box = {}
        self.identity_discontinuities = 0

    @property
    def events(self): return self._impl.events

    def update(self, track_id, timestamp, behaviors, person_box=None):
        tid = int(track_id)
        identity_reset = False
        if person_box is not None:
            box = np.asarray(person_box, dtype=float)
            prior = self.last_identity_box.get(tid)
            if prior is not None:
                old_t, old_box = prior
                # A severe same-ID jump is an identity discontinuity. Discard
                # history before feeding current evidence, avoiding inheritance.
                inter = max(0., min(box[2], old_box[2])-max(box[0], old_box[0])) * max(0., min(box[3], old_box[3])-max(box[1], old_box[1]))
                area_a = max(1., (box[2]-box[0])*(box[3]-box[1])); area_b=max(1.,(old_box[2]-old_box[0])*(old_box[3]-old_box[1]))
                iou = inter/(area_a+area_b-inter)
                if timestamp-old_t <= float(self.cfg["active_grace_sec"]) and iou < .01:
                    identity_reset = True
                    self.identity_discontinuities += 1
                    self._impl.history.pop(tid, None)
                    for key in list(self._impl.states):
                        if key[0] == tid: self._impl.states.pop(key, None)
                    for key in list(self._impl.event_open):
                        if key[0] == tid:
                            record=self._impl.event_records.pop(key,None)
                            self._impl.event_open.pop(key,None)
                            if record is not None:
                                record.update({"end_sec": float(timestamp), "duration_sec": max(0.,float(timestamp)-record["start_sec"]), "end_reason":"identity_discontinuity"})
                self.last_identity_box[tid] = (float(timestamp), box)
            else:
                self.last_identity_box[tid] = (float(timestamp), box)
        temporal_inputs = {b: dict(v) for b, v in behaviors.items()}
        # A food + hand + mouth spatial candidate is only a temporal cycle
        # candidate; V2 lifecycle confirms after distinct repeated episodes.
        eating = temporal_inputs.get("eating", {})
        eev = eating.get("evidence", {})
        if eev.get("food_near_mouth") and eev.get("fingertip_near_food"):
            eating["state"] = "detected"
            eating["decision_reason"] = "food_hand_mouth_cycle_candidate_for_temporal_confirmation"
        phone = temporal_inputs.get("using_phone", {})
        if phone.get("evidence", {}).get("associated_phone_and_same_person_fingertip_contact"):
            phone["decision_reason"] = "associated_coco_phone_and_fingertip_contact"
        out = self._impl.update(tid, timestamp, temporal_inputs)
        for item in out.values():
            evidence = item["evidence"]
            evidence["identity_box_checked"] = person_box is not None
            evidence["identity_discontinuity_reset"] = identity_reset
            if identity_reset or evidence.get("history_reset"):
                phase = "EXPIRED"
            elif evidence.get("ended_at_sec") is not None:
                phase = "LOST_EVIDENCE"
            elif item["state"] == "detected":
                phase = "CONFIRMED"
            elif "waiting_for_persistence" in item.get("decision_reason", ""):
                phase = "CANDIDATE"
            elif item["state"] == "suspected":
                phase = "SUSPECTED"
            else:
                phase = "NORMAL_OR_UNKNOWN"
            evidence["temporal_phase"] = phase
        return out

    def expire(self, timestamp): return self._impl.expire(timestamp)
