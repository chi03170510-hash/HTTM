"""Run image-only Hybrid V7 with immutable V6.1 comparison evidence."""
from __future__ import annotations
import argparse, json, sys, time, hashlib
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]; AI=ROOT/"ai_training"
sys.path.insert(0,str(AI))
from src.face_features import FaceFeatureExtractor
from src.spatial_fusion import spatial_fusion, SpatialConfig
from src.hybrid_v3 import BEHAVIORS, infer_v3
from src.hybrid_v7 import (prepare_objects_v7, spatial_v7, add_sleeping_highres_proposals_v7)
from src.hybrid_v6_1 import (debug_evidence_rows_v6, debug_sidecar_v6)
from src.object_evidence import UsingPhonePrototype

IMAGES=AI/"test_images"; WEIGHTS=AI/"models/best_behavior_v2.pt"
CONFIG=AI/"config/hybrid_v7.json"; FACE=AI/"models/face_landmarker.task"
EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def box_iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0.

def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def draw_text(img,text,xy,color,scale=.46,thickness=1):
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    x=max(2,min(img.shape[1]-tw-3,int(xy[0])));y=max(th+2,min(img.shape[0]-3,int(xy[1])))
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+2,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)

def draw_clean_label(img,text,box,color,occupied):
    x1,y1,x2,y2=map(int,map(round,box));scale=.46;thickness=1
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    h,w=img.shape[:2];choices=[(x1,y1-th-7),(x1,y2+th+6),(x2-tw,y1-th-7),(x2-tw,y2+th+6)]
    selected=None
    for x,y in choices:
        x=max(2,min(w-tw-3,x));baseline=max(th+3,min(h-3,y));r=(x-2,baseline-th-3,x+tw+2,baseline+3)
        if not any(not (r[2]<q[0] or q[2]<r[0] or r[3]<q[1] or q[3]<r[1]) for q in occupied):selected=(x,baseline,r);break
    if selected is None:
        # Search alternate rows before falling back; clamping a stack near the
        # bottom previously placed several labels at the same baseline.
        ys=sorted(range(th+5,max(th+6,h-4),th+7),key=lambda y:abs(y-(y1-th-7)))
        xs=[max(2,min(w-tw-3,x1)),max(2,min(w-tw-3,x2-tw)),2]
        for y in ys:
          for x in xs:
            r=(x-2,y-th-3,x+tw+2,y+3)
            if not any(not (r[2]<q[0] or q[2]<r[0] or r[3]<q[1] or q[3]<r[1]) for q in occupied):selected=(x,y,r);break
          if selected is not None:break
        if selected is None:
          x=2;baseline=max(th+5,min(h-4,th+5+len(occupied)*(th+7)));selected=(x,baseline,(x-2,baseline-th-3,x+tw+2,baseline+3))
    x,y,r=selected;occupied.append(r);draw_text(img,text,(x,y),color,scale,thickness)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images",type=Path,default=IMAGES);ap.add_argument("--weights",type=Path,default=WEIGHTS)
    ap.add_argument("--face-model",type=Path,default=FACE);ap.add_argument("--config",type=Path,default=CONFIG)
    ap.add_argument("--output-dir",type=Path,default=AI/"runs"/f"hybrid_v7_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--device",default="0" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--imgsz",type=int,default=640);ap.add_argument("--iou",type=float,default=.50)
    a=ap.parse_args()
    for f in (a.images,a.weights,a.face_model,a.config):
        if not f.exists():raise FileNotFoundError(f)
    files=sorted(x for x in a.images.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)
    if not files:raise RuntimeError(f"No supported images in {a.images}")
    out=a.output_dir
    if out.exists() and any((out/name).exists() for name in ("images_clean","images_debug","predictions.json","summary.json")):
        out=out.with_name(out.name+"_"+time.strftime("%Y%m%d_%H%M%S"))
    clean_dir=out/"images_clean";debug_dir=out/"images_debug"
    clean_dir.mkdir(parents=True,exist_ok=False);debug_dir.mkdir(parents=True,exist_ok=False)
    checkpoint_hash_before=hashlib.sha256(a.weights.read_bytes()).hexdigest()
    config=json.loads(a.config.read_text(encoding="utf-8"));
    behavior=YOLO(str(a.weights));names={int(k):str(v).strip().lower() for k,v in behavior.names.items()}
    if set(names.values())!={"sleeping","using_phone"}:raise RuntimeError(f"Unexpected v1.4 class mapping {names}")
    objects=UsingPhonePrototype(conf=.12,hand_model_path=AI/"models"/"disabled_by_v6.task",device=a.device)
    ok,msg=FaceFeatureExtractor.availability(a.face_model)
    face=FaceFeatureExtractor(model_path=a.face_model) if ok else None
    if not ok:print("FaceLandmarker fallback:",msg,flush=True)
    spatial_cfg=SpatialConfig(**json.loads((AI/"config/hybrid_behavior_v1_4.json").read_text(encoding="utf-8"))["spatial"])
    rows=[];counts={b:{s:0 for s in ("detected","suspected","not_detected","unknown")} for b in BEHAVIORS}
    model_counts={b:{"raw_boxes":0,"gt_075":0,"roi_added":0} for b in ("sleeping","using_phone")}
    started=time.perf_counter()
    try:
      for ix,path in enumerate(files,1):
        image=cv2.imread(str(path))
        if image is None:
            rows.append({"image":str(path.resolve()),"ok":False,"error":"cv2.imread failed"});continue
        obj=prepare_objects_v7(image,objects,config=config,device=a.device)
        inf=infer_v3(behavior,image,obj,names,imgsz=a.imgsz,conf=float(config["inference_conf"]),
            iou=a.iou,device=a.device,roi_enabled=bool(config["person_roi_enabled"]),
            padding=float(config["person_roi_padding"]),dedupe_iou=float(config["dedupe_iou"]))
        highres_raw,highres_audit=add_sleeping_highres_proposals_v7(image,behavior,inf["detections"],obj,names,config,a.device)
        spatial=spatial_v7(image,inf["detections"],obj,face,config,spatial_fusion,spatial_cfg,objects,a.device)
        owner_by_key={}
        for person in spatial["people"]:
            for det in person["yolo_detections"]:owner_by_key[id(det)]=person
            for b in BEHAVIORS:counts[b][person["behaviors"][b]["state"]]+=1
        for b in model_counts:
            model_counts[b]["raw_boxes"]+=sum(d["class_name"]==b for d in inf["baseline_full_frame"])
            model_counts[b]["gt_075"]+=sum(d["class_name"]==b and d["confidence"]>.75 for d in inf["baseline_full_frame"])
            model_counts[b]["roi_added"]+=sum(d["class_name"]==b and d["source"]=="person_roi" for d in inf["detections"])
        clean=image.copy();debug=image.copy();h,w=image.shape[:2];clean_entries=[]
        # CLEAN: confirmed behavior uses the associated person's box, never a head/object box.
        colors={"sleeping":(40,40,240),"using_phone":(0,150,255),"turning_back":(200,80,200),"eating":(40,180,40)}
        for person in spatial["people"]:
          for behavior_name in BEHAVIORS:
            dec=person["behaviors"][behavior_name]
            if dec["state"]!="detected":continue
            box=dec.get("bounding_box") or person.get("person_bbox_xyxy_clipped")
            if behavior_name=="sleeping":box=dec.get("evidence",{}).get("sleeping_yolo_box") or box
            elif behavior_name=="using_phone":box=(dec.get("evidence",{}).get("phone_boxes") or [box])[0]
            elif behavior_name=="eating":
              food_ids={m.get("food_detection_id") for m in dec.get("evidence",{}).get("food_mouth_matches",[])}
              food=next((x for x in spatial.get("food_detections",[]) if x.get("detection_id") in food_ids),None)
              if food:box=food.get("bbox_xyxy")
            if not box:continue
            x1,y1,x2,y2=map(int,map(round,box));color=colors[behavior_name]
            cv2.rectangle(clean,(x1,y1),(x2,y2),color,2)
            score=dec.get("model_confidence")
            evidence_labels={
              "associated_phone_and_same_person_fingertip_contact":"phone+hand",
              "ear_eye_opening_indicates_closed_eye":"closed-eye",
              "head_supported_by_visible_hand_proxy":"head-support",
              "aligned_head_and_shoulder_tilt":"head/shoulder",
              "head_orientation_matches_calibrated_rear_direction":"rear-yaw",
              "food_box_near_mouth":"food-mouth",
              "fingertip_near_associated_food":"hand-food",
            }
            labels=[evidence_labels[x] for x in dec.get("supporting_evidence",[]) if x in evidence_labels]
            if behavior_name=="eating" and dec.get("state")=="detected":source="food-mouth"
            else:source="+".join(labels[:2]) if labels else ("YOLO" if behavior_name in ("sleeping","using_phone") else "rule")
            label=f"{behavior_name} DETECTED {source}"+(f" yolo={score:.2f}" if score is not None else "")
            clean_entries.append({"person_ref":person["person_ref"],"behavior":behavior_name,"label":label,"box":list(box),"color":color})
        # DEBUG stays selective and readable; full low-score evidence remains JSON.
        if config.get("draw_person_boxes_debug"):
          for person in spatial["people"]:
            x1,y1,x2,y2=map(int,map(round,person["bbox_xyxy"]))
            cv2.rectangle(debug,(x1,y1),(x2,y2),(120,120,120),1)
            draw_text(debug,str(person["person_ref"]),(x1,min(h-4,y2+13)),(180,180,180),.35,1)
        # Collapse all overlapping full-frame/ROI hypotheses to one best row
        # per student and behavior. The detailed hypotheses remain in JSON.
        best_by_person={}
        for det in inf["detections"]:
          if not det.get("bbox_xyxy"):
            continue
          if det["confidence"]<.12:continue
          person=next((p for p in spatial["people"] if det in p["yolo_detections"]),None)
          key=(person["person_ref"] if person else "unassigned",det["class_name"])
          if key not in best_by_person or det["confidence"]>best_by_person[key][0]["confidence"]:
            best_by_person[key]=(det,person)
        draw_rows=sorted(best_by_person.values(),key=lambda item:item[0]["confidence"],reverse=True)[:int(config["max_debug_behavior_boxes"])]
        panel=[]
        for det,person in draw_rows:
          state=person["behaviors"][det["class_name"]]["state"] if person else "unknown"
          x1,y1,x2,y2=map(int,map(round,det["bbox_xyxy"]));color=colors[det["class_name"]]
          cv2.rectangle(debug,(x1,y1),(x2,y2),color,1)
          who=person["person_ref"] if person else "unassigned"
          panel.append((f"{who} {det['class_name']} {det['confidence']:.2f} {state}",color))
        shown={(p["person_ref"],b) for p in spatial["people"] for b in BEHAVIORS
               if any(d["class_name"]==b for d in p["yolo_detections"])}
        for person in spatial["people"]:
          for behavior_name in BEHAVIORS:
            decision=person["behaviors"][behavior_name]
            if (person["person_ref"],behavior_name) in shown or decision["state"] not in ("detected","suspected"):continue
            panel.append((f"{person['person_ref']} {behavior_name}:{decision['state']} rule",colors[behavior_name]))
        if config.get("draw_object_boxes_debug"):
          for kind,items,color in (("phone",spatial["phone_detections"],(255,170,0)),("food",spatial["food_detections"],(40,220,40))):
            groups=[]
            for d in sorted(items,key=lambda x:float(x.get("confidence",0)),reverse=True):
              group=next((g for g in groups if box_iou(g["bbox_xyxy"],d["bbox_xyxy"])>.70),None) if kind=="food" else None
              if group:
                group["class_names"].add(str(d.get("class_name")));group["members"].append(d)
              else:groups.append({"bbox_xyxy":d["bbox_xyxy"],"class_names":{str(d.get("class_name"))},"members":[d]})
            for group in groups:
              x1,y1,x2,y2=map(int,map(round,group["bbox_xyxy"]));cv2.rectangle(debug,(x1,y1),(x2,y2),color,1)
              labels="/".join(sorted(group["class_names"]))
              score=max(float(x.get("confidence",0)) for x in group["members"])
              if kind!="food":draw_text(debug,f"{kind}:{labels} {score:.2f}",(x1,max(12,y1-2)),color,.33,1)
        if config.get("draw_landmarks_debug"):
          for person in spatial["people"]:
            for name,pt in person["evidence"].get("pose_landmarks",{}).items():
              if (pt.get("visibility") or 0)>=.45:
                x,y=int(pt["x"]),int(pt["y"]);cv2.circle(debug,(x,y),2,(255,0,255),-1)
            for hand in person["evidence"].get("associated_hands",[]):
              for x,y in hand.get("fingertips_xy",[]):cv2.circle(debug,(int(x),int(y)),3,(255,255,0),-1)
            mouth=(person["evidence"].get("face") or {}).get("mouth_center_xy_frame")
            if mouth:cv2.circle(debug,(int(mouth[0]),int(mouth[1])),4,(0,255,255),1)
        for candidate in spatial.get("unassigned_sleeping_candidates",[]):
          x1,y1,x2,y2=map(int,map(round,candidate["bbox_xyxy"]));cv2.rectangle(clean,(x1,y1),(x2,y2),colors["sleeping"],2)
          clean_entries.append({"person_ref":"unassigned","behavior":"sleeping candidate","label":f"sleeping candidate unassigned yolo={candidate['model_confidence']:.2f}","box":candidate["bbox_xyxy"],"color":colors["sleeping"]})
          cv2.rectangle(debug,(x1,y1),(x2,y2),colors["sleeping"],1)
        for person in spatial["people"]:
          for behavior_name in ("sleeping","using_phone","eating"):
            dec=person["behaviors"][behavior_name]
            if dec["state"]!="detected":continue
            final_box=dec.get("evidence",{}).get("sleeping_yolo_box") if behavior_name=="sleeping" else dec.get("bounding_box")
            if behavior_name=="using_phone":final_box=(dec.get("evidence",{}).get("phone_boxes") or [final_box])[0]
            if behavior_name=="eating":
              food_ids={m.get("food_detection_id") for m in dec.get("evidence",{}).get("food_mouth_matches",[])}
              food=next((x for x in spatial.get("food_detections",[]) if x.get("detection_id") in food_ids),None)
              if food:final_box=food.get("bbox_xyxy")
            if final_box:
              x1,y1,x2,y2=map(int,map(round,final_box));cv2.rectangle(debug,(x1,y1,x2,y2),colors[behavior_name],3)
              draw_text(debug,f"FINAL {person['person_ref']} {behavior_name}",(x1,max(16,y1-5)),colors[behavior_name],.42,2)
        # Keep descriptive labels in a side panel so overlapping student boxes
        # never cause text to cover one another or the original scene.
        panel_w=300;panel=np.full((h,panel_w,3),245,dtype=np.uint8)
        cv2.putText(panel,"FINAL IMAGE DECISIONS",(12,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,25,25),1,cv2.LINE_AA)
        y=48
        for index,entry in enumerate(clean_entries,1):
          if y>h-20:break
          cv2.putText(panel,f"#{index} {entry['person_ref']} {entry['label'][:31]}",(12,y),cv2.FONT_HERSHEY_SIMPLEX,.31,entry["color"],1,cv2.LINE_AA)
          y+=14
          b=entry["box"]
          cv2.putText(panel,f"box={','.join(str(round(float(v),1)) for v in b)}",(20,y),cv2.FONT_HERSHEY_SIMPLEX,.27,(60,60,60),1,cv2.LINE_AA)
          y+=21
        clean=np.concatenate([clean,panel],axis=1)
        debug=debug_sidecar_v6(debug,debug_evidence_rows_v6(spatial["people"],colors))
        clean_path=clean_dir/(path.stem+".jpg");debug_path=debug_dir/(path.stem+".jpg")
        if not cv2.imwrite(str(clean_path),clean) or not cv2.imwrite(str(debug_path),debug):raise OSError(f"Image write failed: {path}")
        baseline={"detections":inf["baseline_full_frame"],"strict_gt_075_counts":{
            b:sum(d["class_name"]==b and d["confidence"]>.75 for d in inf["baseline_full_frame"]) for b in ("sleeping","using_phone")}}
        rows.append({"image":str(path.resolve()),"ok":True,"clean_image":str(clean_path.resolve()),"debug_image":str(debug_path.resolve()),
          "baseline_v1_4":baseline,"v7":{"person_count":len(spatial["people"]),"person_detection_audit":obj.get("person_detection_audit"),"phone_association_recovery":obj.get("phone_association_recovery"),"roi_person_count":inf["roi_person_count"],
          "deduped_count":inf["deduped_count"],"detections":inf["detections"],"raw_sleeping_highres_proposals":highres_raw,"sleeping_highres_audit":highres_audit,"roi_raw_detections":inf["roi_raw"],
          "people":spatial["people"],"unassigned_yolo_detections":spatial["unassigned_yolo_detections"],
          "unassigned_sleeping_candidates":spatial.get("unassigned_sleeping_candidates",[]),
          "phone_detections":spatial["phone_detections"],"food_detections":spatial["food_detections"],
          "person_detections":obj.get("person_detections",[]),"v6_1_phone_associations":spatial.get("v6_1_phone_associations",[]),
          "v6_1_food_associations":spatial.get("v6_1_food_associations",[]),"v6_1_sleeping_duplicate_removals":spatial.get("v6_1_sleeping_duplicate_removals",[]),
          "object_crop_audit":obj.get("v6_1_object_crop_audit"),"raw_mouth_food_detections":spatial.get("raw_mouth_food_detections",[]),"mouth_food_crop_audit":spatial.get("mouth_food_crop_audit"),"v7_sleep_phone_conflicts":spatial.get("v7_sleep_phone_conflicts",[]),"v7_sleeping_duplicate_removals":spatial.get("v7_sleeping_duplicate_removals",[]),"raw_fullframe_phone_detections":obj.get("raw_fullframe_phone_detections"),
          "raw_fullframe_food_detections":obj.get("raw_fullframe_food_detections"),"raw_crop_object_detections":obj.get("raw_crop_object_detections"),
          "v6_object_associations":spatial.get("v6_object_associations",{})}})
        print(f"V7 image {ix}/{len(files)}: {path.name}; people={len(spatial['people'])}; raw={len(inf['baseline_full_frame'])}; ROI={len(inf['roi_raw'])}; crop_objects={len(obj.get('raw_crop_object_detections',[]))}",flush=True)
    finally:
      if face is not None:face.close()
      objects.close()
    elapsed=time.perf_counter()-started
    checkpoint_hash_after=hashlib.sha256(a.weights.read_bytes()).hexdigest()
    if checkpoint_hash_after!=checkpoint_hash_before:raise RuntimeError("Behavior checkpoint SHA-256 changed during image run")
    summary={"version":"hybrid_v7","mode":"IMAGE spatial evidence only; temporal rules disabled","checkpoint":str(a.weights.resolve()),
      "checkpoint_sha256_before":checkpoint_hash_before,"checkpoint_sha256_after":checkpoint_hash_after,
      "class_mapping":names,"input_dir":str(a.images.resolve()),"total_images":len(files),
      "processed_successfully":sum(x.get("ok",False) for x in rows),"processing_seconds":round(elapsed,3),
      "baseline_v1_4_full_frame_counts":model_counts,"v7_person_spatial_state_counts":counts,
      "person_roi_enabled":bool(config["person_roi_enabled"]),"config":config,"ground_truth_available":False,
      "face_landmarker":msg,"pose_landmarker":objects.pose_version,"hand_landmarker":objects.hand_version,
      "limits":["Image decisions do not use temporal persistence.","COCO and MediaPipe are fallible; absent object evidence does not prove absence.",
        "Eating evidence without food-to-mouth support remains suspected/unknown.","Turning back remains unknown without camera rear-direction calibration.",
        "No ground truth; no accuracy, precision, recall, or mAP is reported."]}
    (out/"predictions.json").write_text(json.dumps({"summary":summary,"images":rows},ensure_ascii=False,indent=2,default=json_default),encoding="utf-8")
    raw_rows=[]
    for row in rows:
      if not row.get("ok"):continue
      v=row["v7"]
      raw_rows.append({"image":row["image"],"baseline_full_frame_behavior_detections":row["baseline_v1_4"]["detections"],
        "behavior_detections_after_fullframe_roi_mapping":v["detections"],"raw_roi_detections_before_dedup":v["roi_raw_detections"],"raw_sleeping_highres_proposals":v.get("raw_sleeping_highres_proposals",[]),"sleeping_highres_audit":v.get("sleeping_highres_audit"),
        "person_detections":v.get("person_detections",[]),"phone_detections_fullframe":v.get("raw_fullframe_phone_detections",[]),
        "food_detections_fullframe":v.get("raw_fullframe_food_detections",[]),"person_crop_object_detections":v.get("raw_crop_object_detections",[]),"mouth_focused_food_detections":v.get("raw_mouth_food_detections",[]),"mouth_food_crop_audit":v.get("mouth_food_crop_audit"),
        "unassigned_sleeping_candidates":v.get("unassigned_sleeping_candidates",[]),
        "final_phone_detections":v.get("phone_detections",[]),"final_food_detections":v.get("food_detections",[]),
        "phone_associations":v.get("v6_1_phone_associations",[]),"food_associations":v.get("v6_1_food_associations",[]),
        "sleeping_duplicate_removals":v.get("v6_1_sleeping_duplicate_removals",[]),
        "person_decisions":[{"person_ref":p["person_ref"],"person_box":p.get("person_bbox_xyxy_clipped"),"behaviors":p["behaviors"]} for p in v.get("people",[])]})
    (out/"raw_detections.json").write_text(json.dumps({"checkpoint_sha256":checkpoint_hash_before,"images":raw_rows},ensure_ascii=False,indent=2,default=json_default),encoding="utf-8")
    import csv
    with (out/"predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
      writer=csv.DictWriter(f,fieldnames=["image","person_id","behavior","state","model_confidence","supporting_evidence","missing_evidence","reason"]);writer.writeheader()
      for row in rows:
       if not row.get("ok"):continue
       for person in row["v7"]["people"]:
        for behavior,dec in person["behaviors"].items():writer.writerow({"image":Path(row["image"]).name,"person_id":person["person_id"],"behavior":behavior,"state":dec["state"],"model_confidence":dec.get("model_confidence"),"supporting_evidence":";".join(map(str,dec.get("supporting_evidence",[]))),"missing_evidence":";".join(map(str,dec.get("evidence",{}).get("missing_evidence",[]))),"reason":dec.get("decision_reason")})
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=json_default),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2));print("OUTPUT_DIR="+str(out.resolve()))
    return out,summary

if __name__=="__main__":main()
