import json
from pathlib import Path

import cv2

AI = Path(__file__).resolve().parents[1]
RUN = AI / "runs/ch09_integration"

face = json.loads(
    (RUN / "face_result.json").read_text(encoding="utf-8")
)

hybrid = json.loads(
    (RUN / "hybrid_result.json").read_text(encoding="utf-8")
)

if face["frame_id"] != hybrid["frame_id"]:
    raise RuntimeError("Frame ID mismatch")

if Path(face["image"]).resolve() != Path(hybrid["image"]).resolve():
    raise RuntimeError("Image mismatch")

image = cv2.imread(face["image"])

if image is None:
    raise RuntimeError("Cannot read image")

for i, person in enumerate(hybrid["people"]):
    box = person.get("person_bbox_xyxy_clipped")

    if box is None:
        continue

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image, (x1, y1), (x2, y2), (255, 0, 0), 2
    )

    cv2.putText(
        image, f"P{i}", (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 0, 0), 2
    )

for i, face_item in enumerate(face["faces"]):
    x1, y1, x2, y2 = map(
        int, face_item["face_bbox"]
    )

    cv2.rectangle(
        image, (x1, y1), (x2, y2), (0, 255, 0), 2
    )

    cv2.putText(
        image, f"F{i}", (x1, min(image.shape[0] - 5, y2 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2
    )

output = RUN / "association_overlay.jpg"

if not cv2.imwrite(str(output), image):
    raise RuntimeError("Cannot save overlay")

print("Overlay saved:", output)
