"""Hybrid V6 behavior-specific decisions built on the existing V3/V4 pipeline.

The module deliberately keeps raw model scores separate from rule decisions.
It reuses V4 object/person proposals and V5 temporal turning-back lifecycle,
without editing either version.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any

from src.hybrid_v3 import BEHAVIORS
from src.hybrid_v4 import prepare_objects_v4, debug_evidence_rows
from src.hybrid_v5 import TemporalV5Fusion, debug_sidecar_v5
from src.hybrid_v3 import spatial_v3
from src.hybrid_v4 import _unique_center_owner


def _center_distance(a, b):
    ax=(float(a[0])+float(a[2]))/2; ay=(float(a[1])+float(a[3]))/2
    bx=(float(b[0])+float(b[2]))/2; by=(float(b[1])+float(b[3]))/2
    return math.hypot(ax-bx, ay-by)


def _same_box_object(a, b, min_iou=.55):
    return _box_iou(a,b) >= float(min_iou)


def _crop_person_object_proposals(frame, objects, people, config, device):
    """Re-run the existing COCO checkpoint on padded person crops only."""
    import cv2
    crops=[]; metadata=[]; h,w=frame.shape[:2]
    pad_x=float(config.get("object_crop_padding_x",.16)); pad_y=float(config.get("object_crop_padding_y",.12))
    target=int(config.get("object_crop_min_side",480)); max_scale=float(config.get("object_crop_max_upscale",2.0))
    for person in people:
        b=person.get("bbox_xyxy")
        if not b or len(b)!=4: continue
        x1,y1,x2,y2=map(float,b); pw=x2-x1; ph=y2-y1
        if pw<24 or ph<24: continue
        ix1=max(0,int(x1-pad_x*pw)); iy1=max(0,int(y1-pad_y*ph))
        ix2=min(w,int(x2+pad_x*pw)); iy2=min(h,int(y2+pad_y*ph))
        crop=frame[iy1:iy2,ix1:ix2]
        if crop.size==0:continue
        scale=min(max_scale,max(1.,target/max(1,min(crop.shape[:2]))))
        if scale>1.01:crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        crops.append(crop);metadata.append({"person_detection_id":person.get("detection_id"),"origin_xy":[ix1,iy1],"scale":scale,"crop_xyxy_frame":[ix1,iy1,ix2,iy2]})
    if not crops:return []
    classes=[i for i,n in objects.class_names.items() if n=="cell phone" or n in objects.food_classes]
    if not classes:return []
    results=objects.model.predict(source=crops,classes=classes,conf=float(config.get("object_crop_conf",.08)),
        iou=float(config.get("object_crop_iou",.5)),imgsz=int(config.get("object_crop_imgsz",640)),
        batch=int(config.get("object_crop_batch",8)),device=device,verbose=False)
    output=[];serial=0
    for meta,result in zip(metadata,results):
        if result.boxes is None:continue
        ox,oy=meta["origin_xy"];scale=float(meta["scale"])
        for box,score,cid in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
            label=objects.class_names.get(int(cid),"")
            kind="phone" if label=="cell phone" else "food" if label in objects.food_classes else None
            if kind is None:continue
            local=list(map(float,box));mapped=[ox+local[0]/scale,oy+local[1]/scale,ox+local[2]/scale,oy+local[3]/scale]
            mapped=[max(0.,min(float(w-1),mapped[0])),max(0.,min(float(h-1),mapped[1])),max(0.,min(float(w-1),mapped[2])),max(0.,min(float(h-1),mapped[3]))]
            if mapped[2]-mapped[0]<3 or mapped[3]-mapped[1]<3:continue
            if kind=="phone":
                ratio=(mapped[2]-mapped[0])/max(1.,mapped[3]-mapped[1])
                if float(score)<float(config.get("object_crop_phone_conf",.15)) or ratio>float(config.get("object_crop_phone_max_aspect",3.0)):
                    continue
            output.append({"detection_id":f"crop_{kind}_{serial}","class_id":int(cid),"class_name":label,
                "bbox_xyxy":[round(v,2) for v in mapped],"confidence":round(float(score),4),"source":"person_crop_coco",
                "person_crop_ref":meta["person_detection_id"],"crop_xyxy_frame":meta["crop_xyxy_frame"],
                "crop_scale":round(scale,3),"bbox_xyxy_crop_scaled":[round(float(v),2) for v in local]});serial+=1
    return output


def prepare_objects_v6_1(frame, objects, *, config, device="cpu"):
    """V6 object evidence plus selective, source-coordinate-audited crop inference."""
    evidence=prepare_objects_v6(frame,objects,config=config,device=device)
    evidence["raw_fullframe_phone_detections"]=[dict(x) for x in evidence.get("phone_detections",[])]
    evidence["raw_fullframe_food_detections"]=[dict(x) for x in evidence.get("food_detections",[])]
    crop=_crop_person_object_proposals(frame,objects,evidence.get("person_detections",[]),config,device)
    evidence["raw_crop_object_detections"]=crop
    for kind,key in (("phone","phone_detections"),("food","food_detections")):
        raw=evidence.get(key,[]);candidates=raw+[x for x in crop if ("phone" if x["class_name"]=="cell phone" else "food")==kind]
        kept=[]
        for det in sorted(candidates,key=lambda d:float(d.get("confidence",0.)),reverse=True):
            old=next((x for x in kept if x.get("class_name")==det.get("class_name") and _same_box_object(x["bbox_xyxy"],det["bbox_xyxy"],config.get("object_duplicate_iou",.55))),None)
            if old is None:kept.append(dict(det))
            else:
                old.setdefault("duplicate_proposals",[]).append({"detection_id":det.get("detection_id"),"source":det.get("source"),"confidence":det.get("confidence"),"bbox_xyxy":det.get("bbox_xyxy")})
                old.setdefault("sources",[old.get("source","full_frame")]);old["sources"].append(det.get("source","full_frame"))
        evidence[key]=kept
    assocs=[]
    for phone in evidence.get("phone_detections",[]):
        owner,status,candidates=_unique_center_owner(phone["bbox_xyxy"],evidence.get("person_detections",[]),near_ratio=0.)
        if owner is None and phone.get("source")=="person_crop_coco":
            matching=[p for p in evidence.get("person_detections",[]) if p.get("detection_id")==phone.get("person_crop_ref")]
            if len(matching)==1:owner,status,candidates=_unique_center_owner(phone["bbox_xyxy"],matching,near_ratio=0.)
        assocs.append({"phone_detection_id":phone.get("detection_id"),"person_detection_id":owner,"status":status,
            "candidate_person_ids":candidates,"candidate_count":len(candidates),"confidence":phone.get("confidence")})
    evidence["associations"]=assocs
    evidence["v6_phone_person_associations"]=[{**a,"normalized_distance":0. if a.get("person_detection_id") else None,
        "phone_center_inside_original_person_box":bool(a.get("person_detection_id")),
        "phone_center_inside_expanded_interaction_region":bool(a.get("person_detection_id"))} for a in assocs]
    foods=[]
    for food in evidence.get("food_detections",[]):
        owner,status,candidates=_unique_center_owner(food["bbox_xyxy"],evidence.get("person_detections",[]),near_ratio=0.)
        if owner is None and food.get("source")=="person_crop_coco":
            matching=[p for p in evidence.get("person_detections",[]) if p.get("detection_id")==food.get("person_crop_ref")]
            if len(matching)==1:owner,status,candidates=_unique_center_owner(food["bbox_xyxy"],matching,near_ratio=0.)
        foods.append({"food_detection_id":food.get("detection_id"),"person_detection_id":owner,"status":status,
            "candidate_person_ids":candidates,"candidate_count":len(candidates)})
    evidence["food_person_associations"]=foods
    evidence["v6_1_object_crop_audit"]={"crop_count":len({x.get("person_crop_ref") for x in crop}),"crop_raw_detection_count":len(crop),
        "phone_final_count":len(evidence.get("phone_detections",[])),"food_final_count":len(evidence.get("food_detections",[])),
        "classes_used":["cell phone",*sorted(objects.food_classes)],"checkpoint":"same pretrained COCO checkpoint; person crops only"}
    return evidence


def deduplicate_sleeping_people(people, config):
    """Merge only near-identical person boxes AND strongly overlapping sleep boxes."""
    candidates=[]
    for p in people:
        d=p.get("behaviors",{}).get("sleeping",{});box=d.get("evidence",{}).get("sleeping_yolo_box")
        if d.get("state") not in ("detected","suspected") or not box:continue
        candidates.append((p,d,box))
    # Greedy representative matching avoids single-link transitive merges: a
    # broad box cannot connect two boxes that do not themselves match it.
    ordered=sorted(candidates,key=lambda row:float(row[1].get("model_confidence") or 0.),reverse=True)
    groups=[]
    for row in ordered:
        p,d,b=row;pb=p.get("person_bbox_xyxy_clipped") or p.get("bbox_xyxy");target=None
        for group in groups:
            rp,rd,rb=group[0];rpb=rp.get("person_bbox_xyxy_clipped") or rp.get("bbox_xyxy")
            if pb and rpb and _box_iou(pb,rpb)>=float(config.get("sleeping_duplicate_person_iou",.55)) and _box_iou(b,rb)>=float(config.get("sleeping_duplicate_behavior_iou",.60)):
                target=group;break
        if target is None:groups.append([row])
        else:target.append(row)
    removed=[]
    for group in groups:
        if len(group)<2:continue
        winner=max(group,key=lambda row:float(row[1].get("model_confidence") or 0.))
        for p,d,b in group:
            if p is winner[0]:continue
            old=d.get("state")
            d.update({"state":"unknown","status":"UNKNOWN","decision":"unknown","decision_reason":"duplicate_sleeping_box_same_person_proposal_cluster",
                "reason":"duplicate_sleeping_box_same_person_proposal_cluster","bounding_box":None,"supporting_evidence":[],
                "evidence":{**d.get("evidence",{}),"duplicate_of_person_ref":winner[0].get("person_ref"),"duplicate_proposal_box":b,
                    "duplicate_proposal_previous_state":old,"duplicate_detection_retained_in_raw_json":True}})
            removed.append({"person_ref":p.get("person_ref"),"duplicate_of_person_ref":winner[0].get("person_ref"),"bbox_xyxy":b,
                "person_box_iou":_box_iou(p.get("person_bbox_xyxy_clipped") or p.get("bbox_xyxy"),winner[0].get("person_bbox_xyxy_clipped") or winner[0].get("bbox_xyxy")),
                "behavior_box_iou":_box_iou(b,winner[2])})
    return removed


def spatial_v6_1(frame, detections, object_evidence, face_extractor, config, spatial_fusion_fn, spatial_config):
    """Image-only V6.1 behavior decisions; no temporal state is invoked."""
    out=spatial_v6(frame,detections,object_evidence,face_extractor,config,spatial_fusion_fn,spatial_config)
    assoc_rows=object_evidence.get("v6_phone_person_associations",[])
    by_owner=defaultdict(list)
    for a in assoc_rows:
        if a.get("person_detection_id") is not None:by_owner[a["person_detection_id"]].append(a)
    phone_by_id={p.get("detection_id"):p for p in object_evidence.get("phone_detections",[])}
    all_phones=object_evidence.get("phone_detections",[])
    for person in out.get("people",[]):
        pid=person.get("person_ref");det=person["behaviors"]["using_phone"]
        person_dets=[x for x in person.get("yolo_detections",[]) if x.get("class_name")=="using_phone" and x.get("bbox_xyxy")]
        min_score=float(config.get("candidate_model_confidence_min",.25))
        candidate=max((x for x in person_dets if float(x.get("confidence",0.))>=min_score),key=lambda x:float(x.get("confidence",0.)),default=None)
        matches=by_owner.get(pid,[])
        ambiguous=[a for a in assoc_rows if pid in a.get("candidate_person_ids",[]) and a.get("status") in ("AMBIGUOUS","UNCERTAIN")]
        plausible_matches=[];implausible_matches=[]
        if candidate and matches:
            pbox=person.get("person_bbox_xyxy_clipped") or person.get("bbox_xyxy")
            for a in matches:
                phone_box=phone_by_id.get(a.get("phone_detection_id"),{});valid,geometry=phone_object_geometry_plausible(phone_box.get("bbox_xyxy"),pbox,config)
                item=(a,phone_box,geometry)
                (plausible_matches if valid else implausible_matches).append(item)
        if candidate and plausible_matches:
            _,best_phone,best_geometry=max(plausible_matches,key=lambda x:float(x[1].get("confidence",0.)))
            _set_decision(det,"detected","YOLO_PHONE_AND_SAME_PERSON_COCO_OBJECT",bbox=best_phone.get("bbox_xyxy"),
                supporting=["behavior_yolo_using_phone_candidate","unique_same_person_coco_phone_object"],extra={
                "reason_code":"PHONE_OBJECT_ASSOCIATED","phone_association_status":"ASSOCIATED_GEOMETRICALLY",
                "phone_detection_ids":[a.get("phone_detection_id") for a,_,_ in plausible_matches],"phone_boxes":[p.get("bbox_xyxy") for _,p,_ in plausible_matches],
                "phone_object_geometry":best_geometry,"hand_or_wrist_required":False,"phone_surface_rule_applied":False,"raw_yolo_score_is_not_calibrated_probability":True})
        elif candidate and implausible_matches:
            _set_decision(det,"suspected","PHONE_OBJECT_GEOMETRY_IMPLAUSIBLE",missing=["phone_object_scale_consistent_with_handheld_device"],extra={
                "reason_code":"PHONE_OBJECT_GEOMETRY_IMPLAUSIBLE","rejected_phone_objects":[{"phone_detection_id":a.get("phone_detection_id"),
                    "bbox_xyxy":p.get("bbox_xyxy"),"geometry":g} for a,p,g in implausible_matches],"raw_yolo_score_is_not_calibrated_probability":True})
        elif candidate and ambiguous:
            _set_decision(det,"suspected","PHONE_ASSOCIATION_AMBIGUOUS",missing=["unique_phone_person_association"],extra={
                "reason_code":"PHONE_ASSOCIATION_AMBIGUOUS","association_candidates":[a.get("candidate_person_ids") for a in ambiguous],"raw_yolo_score_is_not_calibrated_probability":True})
        elif candidate and not all_phones:
            _set_decision(det,"suspected","PHONE_OBJECT_NOT_DETECTED",missing=["coco_cell_phone_object"],extra={
                "reason_code":"PHONE_OBJECT_NOT_DETECTED","raw_yolo_score_is_not_calibrated_probability":True})
        elif candidate:
            other=[a for a in assoc_rows if a.get("person_detection_id") not in (None,pid)]
            code="PHONE_ASSOCIATION_FAILED" if not other else "PHONE_ASSOCIATED_TO_OTHER_PERSON"
            _set_decision(det,"suspected",code,missing=["phone_uniquely_associated_with_candidate_person"],extra={
                "reason_code":code,"phone_association_results":assoc_rows,"raw_yolo_score_is_not_calibrated_probability":True})
        elif person_dets:
            _set_decision(det,"unknown","BEHAVIOR_CANDIDATE_BELOW_THRESHOLD",missing=["minimum_candidate_score"],extra={
                "reason_code":"BEHAVIOR_CANDIDATE_BELOW_THRESHOLD","raw_candidates":[{"confidence":x.get("confidence"),"bbox_xyxy":x.get("bbox_xyxy")} for x in person_dets],
                "candidate_model_confidence_min":min_score})
        else:det.setdefault("evidence",{})["reason_code"]="BEHAVIOR_MODEL_MISS_OR_NO_CANDIDATE"
        sleep=person["behaviors"]["sleeping"]
        sleep_det=max((x for x in person.get("yolo_detections",[]) if x.get("class_name")=="sleeping" and x.get("bbox_xyxy")),
            key=lambda x:float(x.get("confidence",0.)),default=None)
        if sleep_det and candidate:
            if det.get("state")=="detected":
                _set_decision(sleep,"suspected","LABEL_CONFLICT_PHONE_HAS_INDEPENDENT_OBJECT_EVIDENCE",missing=["sleep_specific_evidence"],extra={
                    "reason_code":"LABEL_CONFLICT_PHONE_HAS_INDEPENDENT_OBJECT_EVIDENCE","competing_using_phone_yolo_score":float(candidate.get("confidence",0.)),
                    "sleeping_yolo_score":float(sleep_det.get("confidence",0.)),"resolved_behavior":"using_phone","phone_independently_associated":True})
            else:
                for d in (sleep,det):_set_decision(d,"suspected","LABEL_CONFLICT_UNRESOLVED",missing=["independent_behavior_specific_evidence"],extra={
                    "reason_code":"LABEL_CONFLICT_UNRESOLVED","sleeping_yolo_score":float(sleep_det.get("confidence",0.)),"using_phone_yolo_score":float(candidate.get("confidence",0.))})
    # Recompute eating with a face mouth point, then a pose-derived mouth proxy;
    # proxy evidence remains suspected. Last fallback is a tight head-region center.
    food_owner={x.get("food_detection_id"):x.get("person_detection_id") for x in object_evidence.get("food_person_associations",[])}
    food_rows=object_evidence.get("food_detections",[])
    for person in out.get("people",[]):
        pid=person.get("person_ref");ev=person.get("evidence",{});face=ev.get("face") or {};pose=ev.get("pose_landmarks") or {}
        mouth=face.get("mouth_center_xy_frame");mouth_valid=mouth is not None;mouth_source="face_landmark"
        ear_l,ear_r=pose.get("left_ear") or {},pose.get("right_ear") or {}
        ear_span=0.
        if float(ear_l.get("visibility") or 0.)>=.45 and float(ear_r.get("visibility") or 0.)>=.45:
            ear_span=math.hypot(float(ear_l["x"])-float(ear_r["x"]),float(ear_l["y"])-float(ear_r["y"]))
        head_height=max(24.,ear_span*1.6) if ear_span else max(24.,.22*(float(person["bbox_xyxy"][3])-float(person["bbox_xyxy"][1])))
        if mouth is None:
            nose=pose.get("nose") or {}
            if float(nose.get("visibility") or 0.)>=.5 and ear_span:
                mouth=[float(nose["x"]),float(nose["y"])+.35*ear_span];mouth_source="pose_nose_mouth_proxy"
            else:
                b=person.get("person_bbox_xyxy_clipped") or person["bbox_xyxy"];bw=float(b[2])-float(b[0]);bh=float(b[3])-float(b[1])
                mouth=[float(b[0])+.5*bw,float(b[1])+.20*bh];mouth_source="tight_person_head_region_fallback"
        state,reason,linked,associated=assess_food_head_v6_1(food_rows,food_owner,pid,mouth,head_height,mouth_valid,config)
        eating=person["behaviors"]["eating"]
        food_box=next((f.get("bbox_xyxy") for f in food_rows if any(m.get("food_detection_id")==f.get("detection_id") for m in linked)),None)
        if state=="detected":_set_decision(eating,"detected",reason,bbox=food_box,supporting=["unique_same_person_food_object","food_box_near_visible_mouth"],
            extra={"reason_code":"FOOD_MOUTH_ASSOCIATED","food_mouth_matches":linked,"mouth_source":mouth_source,"head_height_px":head_height,
                "object_confidences_are_not_behavior_probability":True})
        elif state=="suspected":_set_decision(eating,"suspected",reason,missing=["visible_mouth_landmark"],extra={
            "reason_code":"UNCERTAIN_EVIDENCE","food_mouth_matches":linked,"mouth_source":mouth_source,"head_height_px":head_height})
        elif state=="not_detected":_set_decision(eating,"not_detected",reason,missing=[],extra={
            "reason_code":"FOOD_MOUTH_DISTANCE_FAILED","associated_food_boxes":associated,"mouth_source":mouth_source,"head_height_px":head_height})
        else:_set_decision(eating,"unknown",reason,missing=["supported_food_object"],extra={
            "reason_code":"FOOD_OBJECT_NOT_DETECTED","mouth_source":mouth_source,"head_height_px":head_height,
            "supported_food_classes":config["food_classes"]})
    out["v6_1_sleeping_duplicate_removals"]=deduplicate_sleeping_people(out.get("people",[]),config)
    out["v6_1_object_crop_audit"]=object_evidence.get("v6_1_object_crop_audit",{})
    out["v6_1_phone_associations"]=assoc_rows
    out["v6_1_food_associations"]=object_evidence.get("food_person_associations",[])
    return out


def _point_box_distance(point, box):
    x, y = map(float, point)
    x1, y1, x2, y2 = map(float, box)
    return math.hypot(max(x1-x, 0., x-x2), max(y1-y, 0., y-y2))


def _box_iou(a,b):
    x1=max(float(a[0]),float(b[0]));y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]));y2=min(float(a[3]),float(b[3]))
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,float(a[2])-float(a[0]))*max(0.,float(a[3])-float(a[1]))
    bb=max(0.,float(b[2])-float(b[0]))*max(0.,float(b[3])-float(b[1]))
    return inter/(aa+bb-inter) if aa+bb>inter else 0.


def _phone_owner(phone, people, config):
    """Assign each phone to at most one nearby person using normalized geometry."""
    box = list(map(float, phone["bbox_xyxy"]))
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    ranked = []
    max_dist = float(config["phone_association_max_person_height"])
    for person in people:
        pb = list(map(float, person["bbox_xyxy"]))
        height = max(1., pb[3]-pb[1])
        # Interaction area includes a small margin around the torso and below
        # the detected person for phones used near the lap/under a desk.
        expanded = [pb[0]-.12*(pb[2]-pb[0]), pb[1]-.05*height,
                    pb[2]+.12*(pb[2]-pb[0]), pb[3]+.22*height]
        inside = expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]
        core_inside=pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]
        distance = _point_box_distance((cx, cy), expanded)/height
        if inside or distance <= max_dist:
            ranked.append((0 if core_inside else 1, 0. if core_inside else distance,
                           person.get("detection_id"), core_inside, inside))
    ranked.sort(key=lambda x:(x[0],x[1]))
    if not ranked:
        return {"status":"UNASSOCIATED", "person_detection_id":None, "candidate_person_ids":[], "normalized_distance":None}
    margin = float(config["phone_association_ambiguity_margin"])
    best_core=ranked[0][3]
    competing=[r for r in ranked[1:] if r[0]==ranked[0][0]]
    if best_core and not any(r[3] for r in competing):
        competing=[]
    if competing and competing[0][1]-ranked[0][1] < margin:
        return {"status":"AMBIGUOUS", "person_detection_id":None,
                "candidate_person_ids":[ranked[0][2],competing[0][2]], "normalized_distance":ranked[0][1]}
    return {"status":"ASSOCIATED_GEOMETRICALLY", "person_detection_id":ranked[0][2],
            "candidate_person_ids":[ranked[0][2]], "normalized_distance":ranked[0][1],
            "phone_center_inside_original_person_box":ranked[0][3],
            "phone_center_inside_expanded_interaction_region":ranked[0][4]}


def decide_sleeping_candidate(score, box, config):
    if not box or score is None:
        return "not_detected","no_sleeping_box_from_behavior_detector"
    if float(score)>=float(config["sleeping_yolo_detect_threshold"]):
        return "detected","yolo_sleeping_candidate_threshold_met"
    return "suspected","yolo_sleeping_candidate_below_detection_threshold"


def decide_phone_candidate_v6_1(score, association_status, phone_found, config):
    """A behavior candidate is confirmed only with a unique same-person phone."""
    if score is None:return "unknown","BEHAVIOR_MODEL_MISS"
    if association_status=="ASSOCIATED_GEOMETRICALLY" and phone_found:
        return "detected","YOLO_PHONE_AND_SAME_PERSON_COCO_OBJECT"
    if association_status in ("AMBIGUOUS","UNCERTAIN"):
        return "suspected","PHONE_ASSOCIATION_AMBIGUOUS"
    if not phone_found:
        return "suspected","PHONE_OBJECT_NOT_DETECTED"
    return "suspected","PHONE_ASSOCIATION_FAILED"


def phone_object_geometry_plausible(phone_box, person_box, config):
    if not phone_box or not person_box:return False,{"reason":"missing_box"}
    pw=max(1.,float(person_box[2])-float(person_box[0]));ph=max(1.,float(person_box[3])-float(person_box[1]))
    ow=max(1.,float(phone_box[2])-float(phone_box[0]));oh=max(1.,float(phone_box[3])-float(phone_box[1]))
    metrics={"area_ratio":ow*oh/(pw*ph),"height_ratio":oh/ph,"width_ratio":ow/pw,
        "limits":{"area_ratio":float(config.get("phone_object_max_person_area_ratio",.08)),
            "height_ratio":float(config.get("phone_object_max_person_height_ratio",.25)),
            "width_ratio":float(config.get("phone_object_max_person_width_ratio",.40))}}
    valid=(metrics["area_ratio"]<=metrics["limits"]["area_ratio"] and metrics["height_ratio"]<=metrics["limits"]["height_ratio"]
        and metrics["width_ratio"]<=metrics["limits"]["width_ratio"])
    return valid,metrics


def assess_food_head_v6_1(foods, food_owner, person_id, mouth, head_height, mouth_valid, config):
    """Require a uniquely associated COCO food box and tight mouth/head proximity."""
    supported=set(config["food_classes"]);associated=[];unique=[]
    for food in foods:
        if food_owner.get(food.get("detection_id"))!=person_id or food.get("class_name") not in supported:continue
        associated.append(food)
    for food in sorted(associated,key=lambda f:float(f.get("confidence",0.)),reverse=True):
        if any(_box_iou(food.get("bbox_xyxy",[]),old.get("bbox_xyxy",[]))>=float(config.get("food_duplicate_iou",.7)) for old in unique):continue
        unique.append(food)
    linked=[]
    for food in unique:
        box=food.get("bbox_xyxy")
        if not box:continue
        dist=_point_box_distance(mouth,box)/max(1.,float(head_height))
        threshold=float(config.get("food_mouth_distance_head_height",.22) if mouth_valid else config.get("food_head_proxy_distance_head_height",.55))
        if dist<=threshold:
            linked.append({"food_detection_id":food.get("detection_id"),"class_name":food.get("class_name"),
                "object_confidence":food.get("confidence"),"mouth_distance_head_height":dist,"proximity_threshold_head_height":threshold})
    if linked and mouth_valid:return "detected","associated_food_box_near_visible_mouth",linked,associated
    if linked:return "suspected","associated_food_near_pose_or_head_proxy",linked,associated
    if associated:return "not_detected","associated_food_not_near_head_or_mouth",[],associated
    return "unknown","no_associated_supported_food_object",[],[]


def phone_surface_candidate(phone, person_box, pose_landmarks, config):
    """Flag a flat device on the desk/lower body absent a visible nearby wrist."""
    b=list(map(float,phone.get("bbox_xyxy",[])))
    if len(b)!=4:return False
    x1,y1,x2,y2=b;pw=max(1.,x2-x1);ph=max(1.,y2-y1)
    px=(x1+x2)/2;py=(y1+y2)/2
    pb=list(map(float,person_box));height=max(1.,pb[3]-pb[1])
    low=(py-pb[1])/height>=float(config["phone_surface_low_body_y_ratio"])
    flat=pw/ph>=float(config["phone_surface_flat_aspect_ratio"])
    if not (low and flat):return False
    nose=(pose_landmarks or {}).get("nose") or {}
    if float(nose.get("visibility") or 0.)>=float(config["phone_surface_wrist_visibility"]):
        if math.hypot(px-float(nose.get("x",px)),py-float(nose.get("y",py)))/height <= float(config["phone_surface_head_distance_person_height"]):
            return False
    radius=float(config["phone_surface_wrist_distance_person_height"])*height
    for name in ("left_wrist","right_wrist"):
        lm=(pose_landmarks or {}).get(name) or {}
        if float(lm.get("visibility") or 0.)<float(config["phone_surface_wrist_visibility"]):continue
        if _point_box_distance((lm.get("x",-1),lm.get("y",-1)),b)<=radius:return False
    return True


def phone_has_nearby_wrist(phone, person_box, pose_landmarks, config):
    b=phone.get("bbox_xyxy",[])
    if len(b)!=4:return False
    height=max(1.,float(person_box[3])-float(person_box[1]))
    radius=float(config["phone_interaction_wrist_distance_person_height"])*height
    for name in ("left_wrist","right_wrist"):
        lm=(pose_landmarks or {}).get(name) or {}
        if float(lm.get("visibility") or 0.) < float(config["phone_surface_wrist_visibility"]):continue
        if _point_box_distance((lm.get("x",-1),lm.get("y",-1)),b)<=radius:return True
    return False


def decide_phone_candidate(score, associated_phone, association_ambiguous, config, surface_conflict=False,
                           sleeping_conflict=False):
    if score is None:
        return "unknown","no_yolo_using_phone_candidate"
    if float(score)<float(config["candidate_model_confidence_min"]):
        return "unknown","yolo_using_phone_score_below_candidate_threshold"
    if associated_phone and surface_conflict:
        return "suspected","flat_low_phone_without_nearby_wrist_may_be_resting_on_surface"
    if associated_phone and sleeping_conflict:
        return "suspected","phone_near_high_sleeping_candidate_without_nearby_wrist_interaction"
    if associated_phone:
        return "detected","yolo_using_phone_with_uniquely_associated_coco_phone"
    return "suspected","yolo_using_phone_candidate_without_unique_phone_association"


def assess_food_head(foods, food_owner, person_id, mouth, height, mouth_valid, config):
    """Return supported food-to-head evidence; drink/utensil classes never qualify."""
    supported=set(config["food_classes"]);associated=[];unique=[]
    for food in foods:
        if food_owner.get(food.get("detection_id"))!=person_id or food.get("class_name") not in supported:continue
        associated.append(food)
    for food in sorted(associated,key=lambda f:float(f.get("confidence",0.)),reverse=True):
        if any(_box_iou(food.get("bbox_xyxy",[]),old.get("bbox_xyxy",[]))>=float(config["food_duplicate_iou"]) for old in unique):continue
        unique.append(food)
    linked=[]
    for food in unique:
        fb=food.get("bbox_xyxy")
        if not fb:continue
        dist=_point_box_distance(mouth,fb)/max(1.,float(height))
        if dist<=float(config["food_mouth_distance_person_height"]):
            linked.append({"food_detection_id":food.get("detection_id"),"class_name":food.get("class_name"),
                "object_confidence":food.get("confidence"),"mouth_distance_person_height":dist})
    if linked and mouth_valid:return "detected","associated_food_box_near_visible_mouth",linked,associated
    if linked:return "suspected","associated_food_near_estimated_head_region",linked,associated
    if associated:return "not_detected","associated_food_not_near_head_or_mouth",[],associated
    return "unknown","no_associated_supported_food_object",[],[]


def prepare_objects_v6(frame, objects, *, config, device="cpu"):
    evidence = prepare_objects_v4(frame, objects, config=config, device=device)
    people = evidence.get("person_detections", [])
    used = set(); associations=[]
    for phone in evidence.get("phone_detections", []):
        assoc = _phone_owner(phone, [p for p in people if p.get("detection_id") not in used], config)
        assoc["phone_detection_id"] = phone.get("detection_id")
        assoc["confidence"] = phone.get("confidence")
        if assoc.get("person_detection_id") is not None:
            used.add(assoc["person_detection_id"])
        associations.append(assoc)
    evidence["v6_phone_person_associations"] = associations
    return evidence


def _set_decision(dec, state, reason, *, bbox=None, supporting=None, missing=None, extra=None):
    dec.update({"state":state, "status":"IMAGE_ONLY" if state=="detected" else state.upper(),
                "decision":state, "decision_reason":reason, "reason":reason,
                "bounding_box":bbox if state=="detected" else None,
                "supporting_evidence":supporting or [], "contradicting_evidence":[],
                "evidence":{**dec.get("evidence",{}), "missing_evidence":missing or [], **(extra or {})}})


def spatial_v6(frame, detections, object_evidence, face_extractor, config,
               spatial_fusion_fn, spatial_config):
    """Run behavior-specific spatial decisions; preserve unassigned sleeping boxes."""
    decision_config=dict(config)
    if decision_config.get("classroom_rear_yaw_deg") is not None:
        decision_config["turn_rear_yaw_target_deg"]=float(decision_config["classroom_rear_yaw_deg"])
        decision_config["turn_rear_yaw_tolerance_deg"]=float(decision_config.get("classroom_rear_yaw_tolerance_deg",20.))
    base = spatial_v3(frame, detections, object_evidence, face_extractor, decision_config,
                      spatial_fusion_fn, spatial_config)
    people = {p.get("person_ref"):p for p in base.get("people",[])}
    associations = {a.get("phone_detection_id"):a for a in object_evidence.get("v6_phone_person_associations",[])}
    phones = {p.get("detection_id"):p for p in object_evidence.get("phone_detections",[])}
    foods = object_evidence.get("food_detections",[])
    food_assoc = object_evidence.get("food_person_associations",[])
    food_owner = {x.get("food_detection_id"):x.get("person_detection_id") for x in food_assoc}
    unassigned_sleep=[]
    for person in base.get("people",[]):
        pid=person.get("person_ref"); box=person.get("person_bbox_xyxy_clipped") or person.get("bbox_xyxy")
        detections_for_person=person.get("yolo_detections",[])
        evidence=person.get("evidence",{}); face=evidence.get("face") or {}
        # SLEEPING: the behavior detector is the only required decision source.
        sleep_candidates=[d for d in detections_for_person if d.get("class_name")=="sleeping" and d.get("bbox_xyxy")]
        sleep_best=max(sleep_candidates,key=lambda d:float(d.get("confidence",0)),default=None)
        sleep=person["behaviors"]["sleeping"]
        sleep["model_confidence"]=float(sleep_best["confidence"]) if sleep_best else None
        sleep_state,sleep_reason=decide_sleeping_candidate(float(sleep_best["confidence"]) if sleep_best else None,
                                                           sleep_best.get("bbox_xyxy") if sleep_best else None,config)
        if sleep_state=="detected":
            _set_decision(sleep,"detected","yolo_sleeping_candidate_threshold_met",bbox=sleep_best["bbox_xyxy"],
                supporting=["behavior_yolo_sleeping_candidate"],
                extra={"sleeping_yolo_box":sleep_best["bbox_xyxy"],"sleeping_yolo_source":sleep_best.get("source"),
                       "sleeping_yolo_confidence":float(sleep_best["confidence"]),
                       "sleeping_yolo_threshold":float(config["sleeping_yolo_detect_threshold"]),
                       "pose_or_face_required":False,"raw_yolo_score_is_not_calibrated_probability":True})
        elif sleep_state=="suspected":
            _set_decision(sleep,"suspected","yolo_sleeping_candidate_below_detection_threshold",
                missing=["configured_sleeping_yolo_threshold"],extra={"sleeping_yolo_box":sleep_best["bbox_xyxy"],
                "sleeping_yolo_confidence":float(sleep_best["confidence"]),
                "sleeping_yolo_threshold":float(config["sleeping_yolo_detect_threshold"]),"pose_or_face_required":False})
        else:
            _set_decision(sleep,"not_detected","no_sleeping_box_from_behavior_detector",missing=[])

        # PHONE: YOLO supplies the behavior candidate. COCO supplies an
        # independent object check; hands/face are optional and never gates.
        phone_dets=[d for d in detections_for_person if d.get("class_name")=="using_phone"]
        phone_best=max(phone_dets,key=lambda d:float(d.get("confidence",0)),default=None)
        phone_score=float(phone_best["confidence"]) if phone_best else None
        phone_matches=[]
        if phone_best:
            for phone_id, assoc in associations.items():
                if assoc.get("person_detection_id")==pid and assoc.get("status")=="ASSOCIATED_GEOMETRICALLY":
                    phone_matches.append({**phones.get(phone_id,{}),"association":assoc})
        phone=person["behaviors"]["using_phone"];phone["model_confidence"]=phone_score
        pose_landmarks=evidence.get("pose_landmarks",{})
        surface_matches=[m for m in phone_matches if phone_surface_candidate(m,box,pose_landmarks,config)]
        sleeping_conflict=(person["behaviors"]["sleeping"].get("state")=="detected"
            and phone_score is not None and phone_score<float(config["phone_sleeping_conflict_score_threshold"])
            and phone_matches and not any(phone_has_nearby_wrist(m,box,pose_landmarks,config) for m in phone_matches))
        phone_state,phone_reason=decide_phone_candidate(phone_score,bool(phone_matches),
            any(pid in a.get("candidate_person_ids",[]) for a in associations.values()),config,
            surface_conflict=bool(phone_matches and len(surface_matches)==len(phone_matches)),
            sleeping_conflict=sleeping_conflict)
        if phone_state=="detected":
            _set_decision(phone,"detected","yolo_using_phone_with_uniquely_associated_coco_phone",bbox=box,
                supporting=["behavior_yolo_using_phone_candidate","unique_coco_phone_person_association"],
                extra={"phone_detection_ids":[x.get("detection_id") for x in phone_matches],
                       "phone_associations":[x["association"] for x in phone_matches],
                       "phone_object_confidences":[x.get("confidence") for x in phone_matches],
                       "hand_or_fingertip_required":False,"raw_yolo_score_is_not_calibrated_probability":True})
        elif phone_state=="suspected":
            ambiguous=any(pid in a.get("candidate_person_ids",[]) for a in associations.values())
            is_surface=bool(surface_matches and len(surface_matches)==len(phone_matches))
            reason=phone_reason if is_surface or sleeping_conflict else "yolo_using_phone_candidate_without_unique_phone_association"
            missing=(["visible_wrist_or_other_interaction_evidence_for_flat_low_phone"] if is_surface else
                     ["visible_wrist_or_other_interaction_evidence_while_sleeping_candidate"] if sleeping_conflict else
                     ["unique_coco_phone_person_association"])
            _set_decision(phone,"suspected",reason,missing=missing,
                extra={"phone_association_ambiguous":ambiguous,"phone_surface_placement_candidates":surface_matches,
                       "phone_associations":[x["association"] for x in phone_matches],
                       "sleeping_phone_conflict":sleeping_conflict,
                       "hand_or_fingertip_required":False,
                       "raw_yolo_score_is_not_calibrated_probability":True})
        elif phone_best:
            _set_decision(phone,"unknown","yolo_using_phone_score_below_candidate_threshold",missing=["usable_behavior_candidate"],
                extra={"candidate_model_confidence_min":float(config["candidate_model_confidence_min"]),
                       "raw_yolo_score_is_not_calibrated_probability":True})
        else:
            _set_decision(phone,"unknown","no_yolo_using_phone_candidate",missing=["behavior_yolo_candidate"])

        # EATING: only configured edible COCO classes. A visible face gives a
        # mouth point; otherwise use a tight upper-head fallback. A food object
        # merely present on the desk is not considered an eating action.
        x1,y1,x2,y2=map(float,box);height=max(1.,y2-y1);width=max(1.,x2-x1)
        mouth=face.get("mouth_center_xy_frame")
        mouth_valid=mouth is not None
        if not mouth_valid: mouth=(x1+.5*width,y1+.19*height)
        eat_state,eat_reason,linked,associated_food=assess_food_head(foods,food_owner,pid,mouth,height,mouth_valid,config)
        eating=person["behaviors"]["eating"]
        if eat_state=="detected":
            _set_decision(eating,"detected","associated_food_box_near_visible_mouth",bbox=box,
                supporting=["eligible_food_object_associated_with_person","food_box_near_visible_mouth"],
                extra={"food_mouth_matches":linked,"mouth_source":"face_landmark",
                    "behavior_confidence":None,"object_confidences_are_not_behavior_probability":True})
        elif eat_state=="suspected":
            _set_decision(eating,"suspected","associated_food_near_estimated_head_region",missing=["visible_mouth_landmark"],
                extra={"food_mouth_matches":linked,"mouth_source":"tight_person_box_head_fallback",
                    "behavior_confidence":None,"object_confidences_are_not_behavior_probability":True})
        elif eat_state=="not_detected":
            _set_decision(eating,"not_detected","associated_food_not_near_head_or_mouth",missing=[],extra={
                "associated_food_boxes":associated_food,
                "food_detector_miss_possible":True,"supported_food_classes":config["food_classes"]})
        else:
            _set_decision(eating,"unknown","no_associated_supported_food_object",missing=["supported_food_object"],extra={
                "food_detector_miss_possible":True,"supported_food_classes":config["food_classes"],
                "excluded_drink_or_utensil_only_evidence":True})

        # V6 leaves V5 turning-back decision and evidence untouched.
        turning=dict(person["behaviors"]["turning_back"])
        if decision_config.get("turn_rear_yaw_target_deg") is None:
            _set_decision(turning,"unknown","rear_direction_uncalibrated",missing=["camera_relative_classroom_rear_calibration"],
                extra={"rear_direction_calibrated":False})
        person["behaviors"]["turning_back"]=turning

    # Keep valid sleeping candidates even when a student/person association is
    # unavailable; never borrow a neighboring student's ID or merge boxes.
    assigned={id(d) for p in base.get("people",[]) for d in p.get("yolo_detections",[])}
    for det in detections:
        if det.get("class_name")!="sleeping" or id(det) in assigned or not det.get("bbox_xyxy"):continue
        if float(det.get("confidence",0)) < float(config["sleeping_yolo_detect_threshold"]): continue
        unassigned_sleep.append({"behavior":"sleeping","state":"detected","decision_reason":"yolo_sleeping_candidate_unassigned_person",
            "model_confidence":float(det["confidence"]),"bbox_xyxy":det["bbox_xyxy"],"person_ref":None,
            "detection_id":det.get("detection_id"),"source":det.get("source"),"association_status":det.get("association_status","unassigned")})
    base["unassigned_sleeping_candidates"]=unassigned_sleep
    base["v6_object_associations"]={"phones":object_evidence.get("v6_phone_person_associations",[]),
        "foods":object_evidence.get("food_person_associations",[])}
    return base


class TemporalV6Fusion:
    """Light 2-of-3 smoothing for YOLO/object decisions; V5 turning logic kept."""
    def __init__(self, config):
        self.cfg=config;self.turning=TemporalV5Fusion(config);self.hist=defaultdict(lambda:defaultdict(lambda:deque(maxlen=3)))
        self.states={};self._events=[];self.event_open={};self.last_end={}
    @property
    def events(self):return self._events+self.turning.events
    @property
    def identity_discontinuities(self):return self.turning.identity_discontinuities
    def expire(self,t):self.turning.expire(t)
    def update(self,track_id,timestamp,behaviors,person_box=None):
        tid=int(track_id);turn_out=self.turning.update(tid,timestamp,behaviors,person_box);out={}
        for b in BEHAVIORS:
            src=dict(behaviors[b]);state=src.get("state","unknown");ev=dict(src.get("evidence",{}))
            if b=="turning_back":out[b]=turn_out[b];continue
            hist=self.hist[tid][b];hist.append({"t":float(timestamp),"positive":state=="detected","src":src})
            positives=[x for x in hist if x["positive"]]
            stable=len(positives)>=int(self.cfg["video_smoothing_positive_samples"])
            smoothed="detected" if stable else ("suspected" if state=="detected" else state)
            reason=src.get("decision_reason","spatial_decision") if stable else ("waiting_for_light_temporal_smoothing" if state=="detected" else src.get("decision_reason"))
            key=(tid,b);prev=self.states.get(key,"NORMAL");status="CONFIRMED" if smoothed=="detected" else "SUSPECTED" if smoothed=="suspected" else "NORMAL"
            event=False
            if status=="CONFIRMED" and prev!="CONFIRMED":
                cooldown_until=self.last_end.get(key,-1e9)+float(self.cfg["cooldown_sec"])
                if timestamp>=cooldown_until:
                    self.event_open[key]=float(timestamp);self._events.append({"track_id":tid,"behavior":b,"start_sec":float(timestamp),"end_sec":None,"reason":reason});event=True
            elif prev=="CONFIRMED" and status!="CONFIRMED":
                for item in reversed(self._events):
                    if item["track_id"]==tid and item["behavior"]==b and item["end_sec"] is None:
                        item.update(end_sec=float(timestamp),duration_sec=max(0.,timestamp-item["start_sec"]));break
                self.last_end[key]=float(timestamp);self.event_open.pop(key,None)
            self.states[key]=status
            out[b]={**src,"state":smoothed,"status":status,"decision":smoothed,"decision_reason":reason,
                "evidence":{**ev,"spatial_state":state,"timestamp_sec":round(float(timestamp),3),
                    "light_temporal_positive_samples":len(positives),"light_temporal_window_samples":len(hist),
                    "confirmation_event":event,"cooldown_until_sec":(self.last_end[key]+float(self.cfg["cooldown_sec"]) if key in self.last_end else None)}}
        return out


def debug_evidence_rows_v6(people, colors):
    rows=debug_evidence_rows(people,colors)
    by_id={p.get("person_id",p.get("person_ref")):p for p in people}
    for row in rows:
        lines=row.get("lines",[])
        if not lines:continue
        pieces=lines[0].split();pid=pieces[0] if pieces else None
        behavior=pieces[1] if len(pieces)>1 else None
        person=by_id.get(pid,{})
        decision=person.get("behaviors",{}).get(behavior,{})
        evidence=decision.get("evidence",{})
        if behavior=="using_phone":
            assocs=evidence.get("phone_associations",[])
            summary=[f"{x.get('phone_detection_id')}:{x.get('status')} d={x.get('normalized_distance')}" for x in assocs]
            row["lines"].append("phone match="+(",".join(summary) if summary else "none; unique object required"))
        elif behavior=="eating":
            matches=evidence.get("food_mouth_matches",[])
            summary=[f"{x.get('class_name')}:{x.get('food_detection_id')} d={x.get('mouth_distance_person_height')}" for x in matches]
            row["lines"].append("food-mouth="+(",".join(summary) if summary else "none"))
        elif behavior=="sleeping":
            row["lines"].append(f"YOLO box={evidence.get('sleeping_yolo_box')} threshold={evidence.get('sleeping_yolo_threshold','config')}")
    return rows


def debug_sidecar_v6(frame, rows, *, panel_width=520, max_rows=28):
    import cv2
    import numpy as np
    h,w=frame.shape[:2]
    panel=np.full((h,panel_width,3),28,dtype=np.uint8)
    cv2.putText(panel,"HYBRID V6 EVIDENCE",(12,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(panel,"YOLO scores are raw; rules are separate",(12,42),cv2.FONT_HERSHEY_SIMPLEX,.30,(180,180,180),1,cv2.LINE_AA)
    y=64
    for row in rows[:max_rows]:
        for line in row.get("lines",[]):
            if y>h-6:break
            cv2.putText(panel,str(line)[:95],(12,y),cv2.FONT_HERSHEY_SIMPLEX,.29,row.get("color",(225,225,225)),1,cv2.LINE_AA);y+=15
        y+=3
        if y>h-6:break
    return np.concatenate([frame,panel],axis=1)
