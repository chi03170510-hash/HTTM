import json
from pathlib import Path

from person_face_association import associate_faces_to_people
from person_face_association_v2 import associate_v2
from person_face_association_v3 import associate_v3

run = Path("ai_training/runs/ch09_integration")

face = json.loads(
    (run / "face_result_02.json").read_text(encoding="utf-8")
)

hybrid = json.loads(
    (run / "hybrid_result_02.json").read_text(encoding="utf-8")
)

assert face["frame_id"] == hybrid["frame_id"] == 2
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

results = {
    "V1": associate_faces_to_people(tracks, hybrid["people"]),
    "V2": associate_v2(tracks, hybrid["people"]),
    "V3": associate_v3(tracks, hybrid["people"]),
}

for version, matches in results.items():
    print(f"\n{version} matched: {len(matches)}/{len(tracks)}")

    for item in sorted(matches, key=lambda x: str(x["track_id"])):
        print(item["track_id"], "->", item["person_ref"])

    matched_ids = {item["track_id"] for item in matches}

    for track_id, *_ in tracks:
        if track_id not in matched_ids:
            print(track_id, "-> unmatched")
