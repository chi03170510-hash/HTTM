import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))

from src.face_features import FaceFeatureExtractor
from src.spatial_fusion import spatial_fusion, SpatialConfig
from src.hybrid_v3 import infer_v3
from src.hybrid_v7 import (
    prepare_objects_v7,
    spatial_v7,
    add_sleeping_highres_proposals_v7,
)
from src.object_evidence import UsingPhonePrototype
from src.behavior_checkpoint import behavior_checkpoint_path


def process_image(image_path, frame_id, device):
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    config = json.loads(
        (AI / "config/hybrid_v7.json").read_text(encoding="utf-8")
    )

    model = YOLO(str(behavior_checkpoint_path()))

    names = {
        int(k): str(v).strip().lower()
        for k, v in model.names.items()
    }

    if set(names.values()) != {"sleeping", "using_phone"}:
        raise RuntimeError(f"Unexpected classes: {names}")

    objects = UsingPhonePrototype(
        conf=0.12,
        hand_model_path=AI / "models/disabled_by_v6.task",
        device=device,
    )

    face_path = AI / "models/face_landmarker.task"
    available, message = FaceFeatureExtractor.availability(face_path)

    face = (
        FaceFeatureExtractor(model_path=face_path)
        if available
        else None
    )

    spatial_config = SpatialConfig(
        **json.loads(
            (AI / "config/hybrid_behavior_v1_4.json").read_text(
                encoding="utf-8"
            )
        )["spatial"]
    )

    obj = prepare_objects_v7(
        frame, objects, config=config, device=device
    )

    inf = infer_v3(
        model,
        frame,
        obj,
        names,
        imgsz=640,
        conf=float(config["inference_conf"]),
        iou=0.50,
        device=device,
        roi_enabled=bool(config["person_roi_enabled"]),
        padding=float(config["person_roi_padding"]),
        dedupe_iou=float(config["dedupe_iou"]),
    )

    highres_raw, highres_audit = add_sleeping_highres_proposals_v7(
        frame,
        model,
        inf["detections"],
        obj,
        names,
        config,
        device,
    )

    spatial = spatial_v7(
        frame,
        inf["detections"],
        obj,
        face,
        config,
        spatial_fusion,
        spatial_config,
        objects,
        device,
    )

    people = []

    for person in spatial["people"]:
        people.append({
            "person_ref": person["person_ref"],
            "person_bbox_xyxy_clipped": person.get(
                "person_bbox_xyxy_clipped"
            ),
            "bbox_xyxy": person.get("bbox_xyxy"),
            "behaviors": person["behaviors"],
        })

    return {
        "frame_id": frame_id,
        "image": str(image_path.resolve()),
        "mode": "IMAGE_ONLY",
        "person_count": len(people),
        "people": people,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    result = process_image(
        args.image, args.frame_id, args.device
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Hybrid V7 worker: OK")
    print("Frame ID:", result["frame_id"])
    print("People:", result["person_count"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
