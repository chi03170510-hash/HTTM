import json
from pathlib import Path

from person_face_association import associate_faces_to_people
from person_face_association_v2 import associate_v2

run = Path("ai_training/runs/ch09_integration")

face = json.loads(
    (run / "face_result.json").read_text(encoding="utf-8")
)

hybrid = json.loads(
    (run / "hybrid_result.json").read_text(encoding="utf-8")
)

assert face["frame_id"] == hybrid["frame_id"]
assert Path(face["image"]).resolve() == Path(hybrid["image"]).resolve()

tracks = [
    (
        f"static_face_{i}",
        item["face_bbox"],
        item["student_id"],
        item["score"],
    )
    for i, item in enumerate(face["faces"])
]

v1 = associate_faces_to_people(tracks, hybrid["people"])
v2 = associate_v2(tracks, hybrid["people"])

print("V1 matched:", len(v1))
for item in v1:
    print(item["track_id"], "->", item["person_ref"])

print("V2 matched:", len(v2))
for item in v2:
    print(item["track_id"], "->", item["person_ref"])
