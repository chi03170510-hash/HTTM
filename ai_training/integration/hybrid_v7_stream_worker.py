import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO
from person_tracker_diagnostics import DiagnosticConfig, write_detection_sidecar
from person_confidence_trial import filter_person_outputs, person_trial_min_conf_from_environment
from bbox_visual_metadata import enabled as visual_enabled, write_source

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


class HybridV7Stream:
    # Baseline also applies to test harnesses constructing workers via __new__.
    trial_min_conf = None
    visualization_enabled = False

    def __init__(self, device="cpu"):
        self.device = device
        self.objects = None
        self.face = None
        self.diagnostic_config = DiagnosticConfig.from_environment(AI.parent)
        self._diagnostic_frames = 0
        self.visualization_enabled = visual_enabled()

        try:
            self.trial_min_conf = person_trial_min_conf_from_environment()
            if self.trial_min_conf is not None:
                print("[PERSON CONFIDENCE TRIAL] enabled min_conf=0.35 (worker output only)",
                      file=sys.stderr, flush=True)
            self.config = json.loads(
                (AI / "config/hybrid_v7.json").read_text(
                    encoding="utf-8"
                )
            )

            checkpoint = behavior_checkpoint_path()
            self.model = YOLO(str(checkpoint))

            self.names = {
                int(k): str(v).strip().lower()
                for k, v in self.model.names.items()
            }

            if set(self.names.values()) != {
                "sleeping", "using_phone"
            }:
                raise RuntimeError(
                    f"Unexpected classes: {self.names}"
                )

            print(f"[BEHAVIOR MODEL] {checkpoint}; names={self.names}",
                  file=sys.stderr, flush=True)

            self.objects = UsingPhonePrototype(
                conf=0.12,
                hand_model_path=AI / "models/disabled_by_v6.task",
                device=device,
            )

            face_path = AI / "models/face_landmarker.task"
            available, message = (
                FaceFeatureExtractor.availability(face_path)
            )

            self.face = (
                FaceFeatureExtractor(model_path=face_path)
                if available
                else None
            )

            self.spatial_config = SpatialConfig(
                **json.loads(
                    (AI / "config/hybrid_behavior_v1_4.json")
                    .read_text(encoding="utf-8")
                )["spatial"]
            )

        except Exception:
            self.close()
            raise

    def process(self, image_path, frame_id):
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        obj = prepare_objects_v7(
            frame,
            self.objects,
            config=self.config,
            device=self.device,
        )

        inf = infer_v3(
            self.model,
            frame,
            obj,
            self.names,
            imgsz=640,
            conf=float(self.config["inference_conf"]),
            iou=0.50,
            device=self.device,
            roi_enabled=bool(
                self.config["person_roi_enabled"]
            ),
            padding=float(
                self.config["person_roi_padding"]
            ),
            dedupe_iou=float(
                self.config["dedupe_iou"]
            ),
        )

        highres_raw, highres_audit = (
            add_sleeping_highres_proposals_v7(
                frame,
                self.model,
                inf["detections"],
                obj,
                self.names,
                self.config,
                self.device,
            )
        )

        spatial = spatial_v7(
            frame,
            inf["detections"],
            obj,
            self.face,
            self.config,
            spatial_fusion,
            self.spatial_config,
            self.objects,
            self.device,
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

        if self.trial_min_conf is not None:
            before_count = len(people)
            people, rejected = filter_person_outputs(
                people, obj.get("person_detections", []), self.trial_min_conf,
            )
            print("[PERSON CONFIDENCE TRIAL] " + json.dumps({
                "frame_id": frame_id, "min_conf": self.trial_min_conf,
                "before": before_count, "after": len(people), "rejected": rejected,
            }, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)

        if self.diagnostic_config.enabled and self._diagnostic_frames < self.diagnostic_config.max_frames:
            self._diagnostic_frames += 1
            write_detection_sidecar(
                self.diagnostic_config, image_path, frame_id, frame.shape,
                people, obj.get("person_detections", []),
            )

        if self.visualization_enabled:
            write_source(image_path, frame_id, frame.shape, "hybrid", detections=inf["detections"],
                         objects=obj, people=people)

        return {
            "frame_id": frame_id,
            "image": str(image_path.resolve()),
            "mode": "IMAGE_ONLY",
            "person_count": len(people),
            "people": people,
        }

    def close(self):
        if self.face is not None:
            self.face.close()
            self.face = None

        if self.objects is not None:
            self.objects.close()
            self.objects = None


def main():
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    worker = None

    try:
        print("Loading Hybrid V7...", file=sys.stderr)

        worker = HybridV7Stream(device="cpu")

        print(
            "Hybrid V7 model loaded once.",
            file=sys.stderr,
        )

        for line in sys.stdin:
            if not line.strip():
                continue

            frame_id = None

            try:
                request = json.loads(line)
                frame_id = request["frame_id"]
                image_path = Path(request["image"])

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                result = worker.process(
                    image_path, frame_id
                )

            except Exception as exc:
                result = {
                    "frame_id": frame_id,
                    "error": str(exc),
                }

            protocol_stdout.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            protocol_stdout.flush()

    finally:
        if worker is not None:
            worker.close()

        sys.stdout = protocol_stdout


if __name__ == "__main__":
    main()
