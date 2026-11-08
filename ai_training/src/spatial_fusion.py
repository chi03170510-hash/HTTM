"""Single-frame multimodal fusion shared by image and video inference."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import numpy as np


@dataclass
class SpatialConfig:
    sleeping_detect: float = .75
    sleeping_suspect: float = .30
    phone_detect: float = .75
    phone_suspect: float = .25
    sleeping_ear_closed: float = .18
    sleeping_pitch_support_deg: float = 35.0
    eating_mouth_open_ratio: float = .15
    mouth_hand_distance_ratio: float = .12
    head_turn_suspect_deg: float = 45.0
    rear_yaw_target_deg: float | None = None
    rear_yaw_tolerance_deg: float = 20.0


def _iou(a, b):
    ax1,ay1,ax2,ay2=map(float,a); bx1,by1,bx2,by2=map(float,b)
    inter=max(0.,min(ax2,bx2)-max(ax1,bx1))*max(0.,min(ay2,by2)-max(ay1,by1))
    union=max(0.,ax2-ax1)*max(0.,ay2-ay1)+max(0.,bx2-bx1)*max(0.,by2-by1)-inter
    return inter/union if union>0 else 0.


def _label(state, reason, score=None, **evidence):
    # IMAGE states are deliberately separate from video CONFIRMED state.
    return {"state":state,"status":"IMAGE_ONLY" if state=="detected" else state.upper(),
            "score":score,"confidence_yolo":score,"reason":reason,"evidence":evidence}


def spatial_fusion(frame_bgr: np.ndarray, yolo_detections: list[dict[str,Any]],
                   object_evidence: dict[str,Any], face_extractor,
                   config: SpatialConfig | None = None) -> dict[str,Any]:
    """Fuse detections/landmarks for one still image or one sampled video frame.

    Does no temporal confirmation. `object_evidence` is supplied by the existing
    UsingPhonePrototype (COCO phone + MediaPipe Pose/Hands), so hand/phone
    ownership follows that module's geometric association rules.
    """
    c=config or SpatialConfig(); h,w=frame_bgr.shape[:2]
    people=object_evidence.get("person_detections",[])
    poses={p.get("person_id"):p.get("landmarks",{}) for p in object_evidence.get("pose_landmarks",[])}
    hands=object_evidence.get("hand_detections",[])
    phones=object_evidence.get("phone_detections",[])
    foods=object_evidence.get("food_detections",[])
    phone_by_owner={}
    for assoc in object_evidence.get("hand_phone_associations",[]):
        if assoc.get("contact_evidence"):
            owner=assoc.get("phone_person_detection_id")
            if owner: phone_by_owner.setdefault(owner,[]).append(assoc)
    phone_boxes_by_owner={}
    phone_assoc_records=object_evidence.get("associations",[])
    phone_assoc_by_id={a.get("phone_detection_id"):a.get("person_detection_id") for a in phone_assoc_records}
    for ph in phones:
        owner=phone_assoc_by_id.get(ph.get("detection_id"))
        if owner: phone_boxes_by_owner.setdefault(owner,[]).append(ph)
    near_phone_by_owner={}
    for pair in object_evidence.get("hand_phone_associations",[]):
        owner=pair.get("phone_person_detection_id")
        near=pair.get("nearest_fingertip_to_phone_normalized")
        if owner and pair.get("hand_person_status")=="ASSOCIATED_GEOMETRICALLY" and near is not None and near<=.25:
            near_phone_by_owner.setdefault(owner,[]).append(pair)
    food_assoc={a.get("food_detection_id"):a.get("person_detection_id") for a in object_evidence.get("food_person_associations",[])}
    food_by_owner={}
    for food in foods:
        owner=food_assoc.get(food.get("detection_id"))
        if owner:food_by_owner.setdefault(owner,[]).append(food)

    out_people=[]; assigned_yolo=set(); yolo_owner={}
    for i,d in enumerate(yolo_detections):
        if d.get("association_ambiguous"):
            continue
        if d.get("person_ref") is not None and any(p.get("detection_id")==d.get("person_ref") for p in people):
            yolo_owner[i]=d.get("person_ref")
            continue
        db=d.get("bbox_xyxy",[])
        if len(db)!=4 or not people: continue
        overlaps=[_iou(p["bbox_xyxy"],db) for p in people]
        owner_i=int(np.argmax(overlaps))
        if overlaps[owner_i]>=.15:yolo_owner[i]=people[owner_i]["detection_id"]
    for person in people:
        pid=person["detection_id"]; box=person["bbox_xyxy"]
        matched=[]
        for i,d in enumerate(yolo_detections):
            if yolo_owner.get(i)==pid:
                matched.append((i,d));assigned_yolo.add(i)
        scores={}
        for _,d in matched:
            name=str(d.get("class_name","")).strip().lower()
            if name in ("sleeping","using_phone"):
                scores[name]=max(scores.get(name,0.),float(d["confidence"]))

        x1,y1,x2,y2=[int(round(v)) for v in box]
        x1=max(0,min(w-1,x1));x2=max(x1+1,min(w,x2));y1=max(0,min(h-1,y1));y2=max(y1+1,min(h,y2))
        crop=frame_bgr[y1:y2,x1:x2]
        face=face_extractor.extract(crop) if crop.size and face_extractor is not None else None
        if face and face.get("mouth_center_xy"):
            face["mouth_center_xy_frame"]=[face["mouth_center_xy"][0]+x1,face["mouth_center_xy"][1]+y1]
        hp=face.get("head_pose",{"valid":False}) if face else {"valid":False}
        ear=(float(np.mean([face.get("ear_left"),face.get("ear_right")]))
             if face and face.get("eyes_valid") else None)
        pose=poses.get(pid,{})
        visible_pose={k:v for k,v in pose.items() if (v.get("visibility") or 0)>=.35}
        pose_valid=bool(visible_pose)
        person_hands=[hand for hand in hands if hand.get("person_association_status")=="ASSOCIATED_GEOMETRICALLY" and hand.get("person_association")==pid]
        wrists=[pose[k] for k in ("left_wrist","right_wrist") if k in visible_pose]
        hands_visible=bool(wrists or person_hands)
        phone_contacts=phone_by_owner.get(pid,[])
        phone_near_contacts=near_phone_by_owner.get(pid,[])
        phone_objects=phone_boxes_by_owner.get(pid,[])
        food_objects=food_by_owner.get(pid,[])
        mouth=face.get("mouth_center_xy_frame") if face else None
        hand_near_mouth=False
        if mouth:
            pw=max(1.,x2-x1);ph=max(1.,y2-y1)
            hand_near_mouth=any(math.hypot((pt["x"]-mouth[0])/pw,(pt["y"]-mouth[1])/ph)<=c.mouth_hand_distance_ratio for pt in wrists)
            if not hand_near_mouth:
                hand_near_mouth=any(math.hypot((hand.get("wrist_xy",[1e9,1e9])[0]-mouth[0])/pw,
                                                (hand.get("wrist_xy",[1e9,1e9])[1]-mouth[1])/ph)<=c.mouth_hand_distance_ratio for hand in person_hands)
        mouth_open=bool(face and face.get("mouth_open_ratio") is not None and face["mouth_open_ratio"]>=c.eating_mouth_open_ratio)

        sleep=scores.get("sleeping"); sleep_support=bool(hp.get("valid") and ear is not None and
            abs(float(hp.get("pitch_deg") or 0))>=c.sleeping_pitch_support_deg and ear<=c.sleeping_ear_closed)
        if sleep is not None and sleep>c.sleeping_detect:
            sleeping=_label("detected","yolo_v1_4_confidence_strictly_above_0_75",sleep,
                pose_support=sleep_support,head_down_alone_used=False,face_valid=bool(hp.get("valid")),eye_valid=bool(face and face.get("eyes_valid")))
        elif sleep is not None and sleep>=c.sleeping_suspect:
            sleeping=_label("suspected","yolo_confidence_not_strictly_above_0_75; posture_is_context_only",sleep,pose_support=sleep_support,head_down_alone_used=False)
        elif sleep is None:
            sleeping=_label("not_detected" if (face and face.get("eyes_valid")) else "unknown",
                "no_sleeping_detection; single_frame_cannot_assess_persistence",None,
                face_valid=bool(hp.get("valid")),eye_valid=bool(face and face.get("eyes_valid")),pose_valid=pose_valid,ear=ear,
                ear_semantics="eye_opening_measurement_not_sleep_probability",head_down_alone_used=False)
        else:
            sleeping=_label("not_detected","sleeping_confidence_below_suspect_threshold",sleep,
                face_valid=bool(hp.get("valid")),eye_valid=bool(face and face.get("eyes_valid")),pose_valid=pose_valid)

        phone_score=scores.get("using_phone"); interaction=bool(phone_contacts)
        phone_uncertain=any(a.get("status")=="UNCERTAIN" and pid in a.get("candidate_person_ids",[])
                            for a in phone_assoc_records)
        if phone_score is not None and phone_score>c.phone_detect:
            phone=_label("detected","yolo_v1_4_confidence_strictly_above_0_75",phone_score,
                coco_phone_count=len(phone_objects),contact_count=len(phone_contacts),hands_visible=hands_visible)
        elif phone_score is not None and phone_score>=c.phone_suspect:
            phone=_label("suspected","yolo_candidate_without_phone_hand_interaction_confirmation",phone_score,
                coco_phone_count=len(phone_objects),contact=False,hands_visible=hands_visible)
        elif interaction or (phone_objects and phone_near_contacts):
            phone=_label("suspected","COCO_phone_hand_contact_is_rule_candidate_only; strict_YOLO_gate_not_met",phone_score,
                coco_phone_count=len(phone_objects),near_hand_pairs=len(phone_near_contacts),hands_visible=hands_visible)
        elif phone_objects and not hands_visible:
            phone=_label("unknown","phone_associated_with_person_but_hand_interaction_not_observable",phone_score,
                coco_phone_count=len(phone_objects),hands_visible=False)
        elif phone_uncertain:
            phone=_label("unknown","COCO_phone_person_association_ambiguous",phone_score,hands_visible=hands_visible)
        else:
            phone=_label("not_detected","no_yolo_phone_candidate_and_no_interacting_COCO_phone",phone_score,
                coco_phone_count=len(phone_objects),hands_visible=hands_visible,phone_on_desk_alone_used=False)

        yaw=hp.get("yaw_deg") if hp.get("valid") else None
        yaw_abs=abs(float(yaw)) if yaw is not None else None
        if yaw_abs is not None and yaw_abs>=c.head_turn_suspect_deg:
            if c.rear_yaw_target_deg is not None and abs(float(yaw)-c.rear_yaw_target_deg)<=c.rear_yaw_tolerance_deg:
                turn=_label("suspected","rear_direction_target_requires_video_persistence",None,head_yaw_deg=yaw)
            else:
                turn=_label("suspected","head_turn_only_rear_direction_not_identified",None,head_yaw_deg=yaw)
        else:
            turn=_label("unknown","single_frame_or_rear_direction_reference_insufficient",None,
                face_valid=bool(hp.get("valid")),pose_valid=pose_valid,head_yaw_deg=yaw)

        food_at_person=food_objects
        if hand_near_mouth and mouth_open and food_at_person:
            eating=_label("suspected","single_image_food_hand_mouth_is_rule_candidate_not_video_confirmation",None,
                hand_near_mouth=True,mouth_open_ratio=face.get("mouth_open_ratio"),food_objects=food_at_person)
        elif hand_near_mouth and mouth_open:
            eating=_label("suspected","hand_near_open_mouth_without_confirmed_food_context",None,
                hand_near_mouth=True,mouth_open_ratio=face.get("mouth_open_ratio"),chin_rest_not_used_alone=True)
        else:
            eating=_label("unknown","no_single_frame_eating_specific_sequence_or_food_interaction",None,
                hand_near_mouth=hand_near_mouth,mouth_open_ratio=face.get("mouth_open_ratio") if face else None,
                food_objects=food_at_person,chin_rest_not_used_alone=True)

        out_people.append({"person_ref":pid,"bbox_xyxy":box,"yolo_detections":[d for _,d in matched],
            "behaviors":{"sleeping":sleeping,"using_phone":phone,"turning_back":turn,"eating":eating},
            "evidence_validity":{"pose":pose_valid,"face":bool(hp.get("valid")),"eyes":bool(face and face.get("eyes_valid")),
                "hands":hands_visible,"phone_detector":True,"food_detector":"COCO subset"},
            "evidence":{"face":face,"pose_landmarks":pose,"associated_hands":person_hands,
                "associated_phone_boxes":phone_objects,"phone_hand_contacts":phone_contacts,
                "phone_near_hand_pairs":phone_near_contacts,"associated_food_boxes":food_objects}})
    unassigned=[]
    for i,d in enumerate(yolo_detections):
        if i in assigned_yolo:continue
        name=d.get("class_name");score=float(d.get("confidence",0.))
        threshold=c.sleeping_detect if name=="sleeping" else c.phone_detect
        low=c.sleeping_suspect if name=="sleeping" else c.phone_suspect
        state="unknown" if d.get("association_ambiguous") else "detected" if score>threshold else "suspected" if score>=low else "not_detected"
        unassigned.append({**d,"spatial_state":state,"person_ref":None,
            "association_status":"ambiguous" if d.get("association_ambiguous") else "unknown",
            "reason":"multiple_person_boxes_are_equally_plausible" if d.get("association_ambiguous") else "raw_YOLO_detection_preserved_without_person_or_landmark_association"})
    return {"people":out_people,"phone_detections":phones,"food_detections":foods,
        "raw_yolo_detections":yolo_detections,
        "unassigned_yolo_detections":unassigned,
        "object_evidence":object_evidence}
