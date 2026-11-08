"""Hybrid V5: fix coarse-person gating for independently verified phone use.

V5 keeps the V1.4 behavior checkpoint and reuses V4 proposals and V3 temporal
lifecycle. It adds geometry-based recovery only when one phone object and a
uniquely owned hand provide independent evidence. It does not calibrate scores.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any

from src.hybrid_v3 import BEHAVIORS
from src.hybrid_v4 import (prepare_objects_v4, spatial_v4, TemporalV4Fusion,
                           debug_evidence_rows as _debug_evidence_rows)
import numpy as np


def _inside(point, box):
    x, y = map(float, point)
    x1, y1, x2, y2 = map(float, box)
    return x1 <= x <= x2 and y1 <= y <= y2


def prepare_objects_v5(frame, objects, *, config, device="cpu"):
    """Reuse V4 detections and recover a phone owner from unique hand contact.

    A phone remains unassigned if multiple distinct students have qualifying
    hand contact or the hand itself has ambiguous ownership. A full-body person
    box is not required when the phone center is ambiguous but contact uniquely
    identifies one student.
    """
    prepare_started = time.perf_counter()
    evidence = prepare_objects_v4(frame, objects, config=config, device=device)
    association_started = time.perf_counter()
    min_tips = int(config.get("phone_direct_contact_min_fingertips", 2))
    people = {p["detection_id"]: p for p in evidence.get("person_detections", [])}
    hands = evidence.get("hand_detections", [])
    phones = evidence.get("phone_detections", [])
    associations = {x.get("phone_detection_id"): x for x in evidence.get("associations", [])}
    repaired, pairs, interactions = [], [], []
    recovery_log = []

    for phone in phones:
        phone_id = phone.get("detection_id")
        box = phone.get("bbox_xyxy", [])
        original = dict(associations.get(phone_id, {}))
        contact_hands = []
        for hand in hands:
            tips = hand.get("fingertips_xy", [])
            inside_count = sum(_inside(pt, box) for pt in tips)
            uniquely_owned = (hand.get("person_association_status") == "ASSOCIATED_GEOMETRICALLY"
                              and hand.get("person_association") in people
                              and len(hand.get("person_candidate_ids", [hand.get("person_association")])) == 1)
            if inside_count >= min_tips and uniquely_owned:
                contact_hands.append((hand, inside_count))
        contact_owners = {hand.get("person_association") for hand, _ in contact_hands}
        original_owner = original.get("person_detection_id")
        owner = original_owner
        recovery_reason = None
        if original_owner is None and len(contact_owners) == 1:
            owner = next(iter(contact_owners))
            recovery_reason = "unique_hand_phone_contact_resolved_ambiguous_phone_center"
        elif original_owner is not None and contact_owners and contact_owners != {original_owner}:
            owner = None
            recovery_reason = "phone_center_owner_conflicts_with_hand_contact_owner"
        elif len(contact_owners) > 1:
            owner = None
            recovery_reason = "multiple_people_contact_same_phone_ambiguous"

        if owner is not None:
            record = {**original, "phone_detection_id": phone_id,
                      "person_detection_id": owner,
                      "status": "ASSOCIATED_BY_UNIQUE_HAND_CONTACT" if recovery_reason else original.get("status"),
                      "candidate_person_ids": [owner], "candidate_count": 1,
                      "original_status": original.get("status"),
                      "recovery_reason": recovery_reason}
        else:
            record = {**original, "phone_detection_id": phone_id,
                      "person_detection_id": None,
                      "status": "UNCERTAIN", "candidate_person_ids": sorted(x for x in contact_owners if x),
                      "candidate_count": len(contact_owners) or original.get("candidate_count", 0),
                      "original_status": original.get("status"),
                      "recovery_reason": recovery_reason or "no_unique_phone_owner"}
        repaired.append(record)
        recovery_log.append(record)

        qualifying = 0
        for hand in hands:
            tips = hand.get("fingertips_xy", [])
            inside_count = sum(_inside(pt, box) for pt in tips)
            is_unique_hand = (hand.get("person_association_status") == "ASSOCIATED_GEOMETRICALLY"
                              and hand.get("person_association") in people
                              and len(hand.get("person_candidate_ids", [hand.get("person_association")])) == 1)
            contact = bool(owner is not None and is_unique_hand
                           and hand.get("person_association") == owner and inside_count >= min_tips)
            if contact:
                qualifying += 1
            x1, y1, x2, y2 = map(float, box)
            diag = max(1., math.hypot(x2-x1, y2-y1))
            distances = [math.hypot(max(x1-float(x), 0., float(x)-x2),
                                    max(y1-float(y), 0., float(y)-y2))/diag for x, y in tips]
            pairs.append({"phone_detection_id": phone_id,
                          "hand_detection_id": hand.get("detection_id"),
                          "phone_person_detection_id": owner,
                          "person_detection_id": hand.get("person_association"),
                          "hand_person_status": hand.get("person_association_status"),
                          "fingertip_landmarks_inside_phone": inside_count,
                          "nearest_fingertip_to_phone_normalized": min(distances) if distances else None,
                          "contact_evidence": contact,
                          "hand_phone_status": "CONTACT_GEOMETRY" if contact else "NO_CONTACT_EVIDENCE"})
        interactions.append({"phone_detection_id": phone_id, "person_detection_id": owner,
                             "status": "USING_PHONE_EVIDENCE" if qualifying else "UNCERTAIN",
                             "qualifying_hand_contact_count": qualifying,
                             "independent_object_and_hand_sources": ["coco_phone_detector", "mediapipe_hand_landmarker"]
                             if qualifying else []})

    evidence["associations"] = repaired
    evidence["hand_phone_associations"] = pairs
    evidence["interaction_evidence"] = interactions
    evidence["phone_association_recovery"] = recovery_log
    evidence["v5_association_ms"] = round((time.perf_counter()-association_started)*1000, 3)
    evidence["v5_prepare_total_ms"] = round((time.perf_counter()-prepare_started)*1000, 3)
    return evidence


def associate_sleeping_by_upper_body(detections, object_evidence, *, config):
    """Recover only unassigned sleeping proposals from unique pose geometry.

    Candidate center must be near a nose/ear/shoulder anchor, and the best
    normalized-distance match must beat the second person by a configured gap.
    Already assigned or ROI-provenanced proposals are never reassigned.
    """
    max_distance = float(config.get("sleeping_pose_association_max_person_height", .22))
    margin = float(config.get("sleeping_pose_association_margin", .08))
    pose_people = object_evidence.get("pose_landmarks", [])
    by_person = {p.get("person_id"): p.get("landmarks", {}) for p in pose_people}
    people = {p.get("detection_id"): p for p in object_evidence.get("person_detections", [])}
    logs = []
    for det in detections:
        if det.get("class_name") != "sleeping" or det.get("person_ref") is not None:
            continue
        box = det.get("bbox_xyxy")
        if not box or det.get("source") == "person_roi" or "person_roi" in det.get("sources", []):
            continue
        cx, cy = (float(box[0])+float(box[2]))/2, (float(box[1])+float(box[3]))/2
        ranked = []
        for pid, landmarks in by_person.items():
            person = people.get(pid)
            if not person: continue
            height = max(1., float(person["bbox_xyxy"][3])-float(person["bbox_xyxy"][1]))
            anchors = [landmarks.get(name) for name in ("nose", "left_ear", "right_ear",
                       "left_shoulder", "right_shoulder")]
            anchors = [p for p in anchors if p and (p.get("visibility") or 0.) >= .45]
            if not anchors: continue
            # Use the closest visible head/shoulder anchor in person-height units.
            distance = min(math.hypot(cx-float(p["x"]), cy-float(p["y"])) for p in anchors)/height
            if distance <= max_distance:
                ranked.append((distance, pid))
        ranked.sort()
        if ranked and (len(ranked) == 1 or ranked[1][0]-ranked[0][0] >= margin):
            distance, owner = ranked[0]
            det.update({"person_ref": owner, "association_ambiguous": False,
                        "association_status": "associated_by_unique_head_upper_body_pose",
                        "pose_association_distance_person_height": round(distance, 4)})
            logs.append({"detection_id": det.get("detection_id"), "person_ref": owner,
                         "status": "RECOVERED_UNIQUE_POSE", "distance_person_height": round(distance, 4)})
        else:
            logs.append({"detection_id": det.get("detection_id"), "person_ref": None,
                         "status": "AMBIGUOUS_OR_NO_POSE_MATCH",
                         "candidate_person_ids": [pid for _, pid in ranked]})
    return logs


def spatial_v5(frame, detections, object_evidence, face_extractor, config,
               spatial_fusion_fn, spatial_config):
    pose_logs = associate_sleeping_by_upper_body(detections, object_evidence, config=config)
    effective_config = dict(config)
    if effective_config.get("classroom_rear_yaw_deg") is not None:
        effective_config["turn_rear_yaw_target_deg"] = float(effective_config["classroom_rear_yaw_deg"])
        effective_config["turn_rear_yaw_tolerance_deg"] = float(
            effective_config.get("classroom_rear_yaw_tolerance_deg", 20.))
    result = spatial_v4(frame, detections, object_evidence, face_extractor, effective_config,
                        spatial_fusion_fn, spatial_config)
    assoc = {x.get("phone_detection_id"): x for x in object_evidence.get("associations", [])}
    pairs = object_evidence.get("hand_phone_associations", [])
    for person in result.get("people", []):
        pid = person.get("person_ref")
        dec = person.get("behaviors", {}).get("using_phone", {})
        qualifying = []
        for pair in pairs:
            if (pair.get("phone_person_detection_id") == pid
                    and pair.get("person_detection_id") == pid
                    and pair.get("hand_person_status") == "ASSOCIATED_GEOMETRICALLY"
                    and pair.get("contact_evidence")
                    and int(pair.get("fingertip_landmarks_inside_phone", 0))
                    >= int(config.get("phone_direct_contact_min_fingertips", 2))):
                phone_assoc = assoc.get(pair.get("phone_detection_id"), {})
                if (phone_assoc.get("person_detection_id") == pid
                        and phone_assoc.get("candidate_count") == 1
                        and phone_assoc.get("status") in ("ASSOCIATED_GEOMETRICALLY",
                            "ASSOCIATED_BY_UNIQUE_HAND_CONTACT")):
                    qualifying.append((pair, phone_assoc))
        # V4's generic coarse-person gate rejected a direct phone + same-person
        # hand contact. Permit this independent two-model evidence path; the
        # association still has to resolve uniquely to exactly one person.
        if qualifying:
            pair, phone_assoc = qualifying[0]
            dec.update({"state": "detected", "status": "IMAGE_ONLY", "decision": "detected",
                        "decision_reason": "independent_phone_object_unique_person_and_hand_contact",
                        "reason": "independent_phone_object_unique_person_and_hand_contact",
                        "bounding_box": person.get("person_bbox_xyxy_clipped") or person.get("bbox_xyxy"),
                        "supporting_evidence": ["independent_coco_phone_object",
                            "unique_phone_person_geometric_association",
                            "same_person_multi_fingertip_contact"]})
            ev = dec.setdefault("evidence", {})
            ev.update({"phone_association_status": phone_assoc.get("status"),
                       "phone_candidate_person_count": phone_assoc.get("candidate_count"),
                       "contact_fingertips_inside_phone": pair.get("fingertip_landmarks_inside_phone"),
                       "phone_detection_id": pair.get("phone_detection_id"),
                       "hand_detection_id": pair.get("hand_detection_id"),
                       "coarse_person_box_did_not_block_independent_interaction":
                           person.get("person_detection_quality", {}).get("quality") == "coarse",
                       "evidence_sources": ["behavior_yolo_v1_4", "coco_phone", "mediapipe_hands"]})
            dec["contradicting_evidence"] = [x for x in dec.get("contradicting_evidence", [])
                                               if x != "person_box_exceeds_configured_geometry_limits"]
        person.setdefault("evidence", {})["sleeping_pose_association_audit"] = pose_logs
    result["association_audit"] = {"sleeping_pose_recovery": pose_logs,
                                   "phone_recovery": object_evidence.get("phone_association_recovery", [])}
    return result


def debug_evidence_rows_v5(people, behavior_colors):
    rows = _debug_evidence_rows(people, behavior_colors)
    by_id = {p.get("person_id", p.get("person_ref")): p for p in people}
    for row in rows:
        lines = row.get("lines", [])
        pid = lines[0].split()[0] if lines else None
        person = by_id.get(pid, {})
        evidence = person.get("evidence", {})
        phones = [x.get("detection_id") for x in evidence.get("associated_phone_boxes", [])]
        contacts = [x.get("phone_detection_id") for x in evidence.get("phone_hand_contacts", [])]
        q = person.get("person_detection_quality", {}).get("quality", "unknown")
        row["lines"].append(f"person_box={q} phone_boxes={phones or 'none'} contacts={contacts or 'none'}")
    return rows


def debug_sidecar_v5(frame, rows, *, panel_width=520, max_rows=28):
    """Add V5 evidence at the right, leaving source pixels unchanged."""
    import cv2
    h, w = frame.shape[:2]
    panel = np.full((h, panel_width, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "HYBRID V5 EVIDENCE", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, "model score is raw output, not probability", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, .30, (180, 180, 180), 1, cv2.LINE_AA)
    y = 64
    for row in rows[:max_rows]:
        color = row.get("color", (225, 225, 225))
        for line in row.get("lines", []):
            if y > h-6: break
            cv2.putText(panel, str(line)[:95], (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .29, color, 1, cv2.LINE_AA)
            y += 15
        y += 3
        if y > h-6: break
    cv2.putText(panel, "Full evidence: JSON/CSV sidecar", (12, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, .30, (170, 170, 170), 1, cv2.LINE_AA)
    return np.concatenate([frame, panel], axis=1)


class TemporalV5Fusion(TemporalV4Fusion):
    """V4 time lifecycle; phone contact evidence is retained per track/frame."""
    def update(self, track_id, timestamp, behaviors, person_box=None):
        out = super().update(track_id, timestamp, behaviors, person_box)
        for behavior in BEHAVIORS:
            item = out[behavior]
            item.setdefault("evidence", {})["temporal_interaction_duration_sec"] = item.get("evidence", {}).get("confirmation_duration_sec") if behavior == "using_phone" else None
            if behavior == "using_phone":
                item["evidence"]["temporal_phone_contact_observed"] = bool(
                    item["evidence"].get("associated_phone_and_same_person_fingertip_contact"))
        return out
