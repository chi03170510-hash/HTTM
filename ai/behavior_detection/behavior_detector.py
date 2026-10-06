from ultralytics import YOLO


class BehaviorDetector:
    """
    Detect student abnormal behaviors.

    Classes:
        0: sleeping
        1: using_phone
    """

    def __init__(
        self,
        model_path,
        sleeping_conf=0.40,
        phone_conf=0.40,
        imgsz=640,
        device=None
    ):
        self.model = YOLO(model_path)

        self.thresholds = {
            "sleeping": sleeping_conf,
            "using_phone": phone_conf
        }

        self.imgsz = imgsz
        self.device = device

    def detect(self, frame):
        """
        Input:
            frame: OpenCV BGR frame (numpy.ndarray)

        Output:
            List[dict]
        """

        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,

            # Lấy candidate từ mức thấp hơn,
            # sau đó lọc riêng theo từng class
            conf=0.10,

            iou=0.70,
            device=self.device,
            verbose=False
        )[0]

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = self.model.names[class_id]

            # Chỉ lấy 2 behavior được hỗ trợ
            if class_name not in self.thresholds:
                continue

            threshold = self.thresholds[class_name]

            if confidence < threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 4),

                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            }

            detections.append(detection)

        return detections