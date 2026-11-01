"""YOLO behavior detector with explicit V2/V1 experimental checkpoint selection."""
from pathlib import Path
from ultralytics import YOLO

MODEL_DIR = Path(__file__).resolve().parent / "models" / "experimental"
CHECKPOINTS = {
    "v2": MODEL_DIR / "three_behavior_v2_best.pt",
    "v1": MODEL_DIR / "three_behavior_v1_last.pt",
}
EXPECTED_CLASSES = {0: "sleeping", 1: "using_phone", 2: "raising_hand"}


class BehaviorDetector:
    """Detect sleeping, using_phone, and raising_hand from a BGR frame.

    The default experimental model is V2. Pass ``model_version="v1"`` or an
    explicit ``model_path`` to run the V1 checkpoint for comparison.
    """

    def __init__(self, model_path=None, sleeping_conf=0.40, phone_conf=0.40,
                 raising_hand_conf=0.40, imgsz=640, device=None, model_version="v2"):
        version = str(model_version).lower()
        if version not in CHECKPOINTS:
            raise ValueError(f"Unknown model_version={model_version!r}; choose one of {tuple(CHECKPOINTS)}")
        self.model_path = Path(model_path) if model_path is not None else CHECKPOINTS[version]
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Behavior checkpoint not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        names = self.model.names
        name_items = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_names = {int(class_id): str(name).strip().lower() for class_id, name in name_items}
        if len(set(self.class_names.values())) != len(self.class_names):
            raise RuntimeError(f"Checkpoint has duplicate class names: {self.class_names}")
        if any(name not in EXPECTED_CLASSES.values() for name in self.class_names.values()):
            raise RuntimeError(f"Unsupported behavior class mapping: {self.class_names}")
        legacy_two_class = {0: "sleeping", 1: "using_phone"}
        if self.class_names not in (EXPECTED_CLASSES, legacy_two_class):
            raise RuntimeError(f"Checkpoint class IDs do not match HTTM mapping: {self.class_names}")

        self.thresholds = {
            "sleeping": float(sleeping_conf),
            "using_phone": float(phone_conf),
            "raising_hand": float(raising_hand_conf),
        }
        self.imgsz = imgsz
        self.device = device

    def predict(self, frame):
        """Predict per-frame detections from an OpenCV BGR ndarray.

        Each result is a model detection, not a ground-truth behavior event.
        ``track_id`` remains None because this per-frame module does not track.
        """
        if frame is None or getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty OpenCV BGR image with shape HxWx3")
        result = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=0.10, iou=0.70,
            device=self.device, verbose=False
        )[0]
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])
            if confidence < self.thresholds[class_name]:
                continue
            bbox_xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": bbox_xyxy,
                "track_id": None,
                "status": "MODEL_DETECTION",
            })
        return detections

    def detect(self, frame):
        """Compatibility adapter for the legacy detector interface."""
        return [
            {"class": item["class_name"], "class_id": item["class_id"],
             "confidence": item["confidence"],
             "bbox": [int(v) for v in item["bbox_xyxy"]]}
            for item in self.predict(frame)
        ]

    def reset(self):
        """Reset per-video state. This detector is stateless and assigns no track IDs."""
        return None
