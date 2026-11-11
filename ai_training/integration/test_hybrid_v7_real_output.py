import json
from pathlib import Path

from person_face_association import associate_faces_to_people


ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "ai_training" / "runs" / "ch09_hybrid_test" / "predictions.json"


def valid_box(box):
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(v, (int, float)) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def find_unique_face(people):
    for target in people:
        box = target.get("person_bbox_xyxy_clipped") or target.get("bbox_xyxy")

        if not valid_box(box):
            continue

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        for xr in (0.25, 0.5, 0.75):
            for yr in (0.15, 0.3, 0.45):
                cx = x1 + xr * width
                cy = y1 + yr * height

                matches = []

                for person in people:
                    pb = person.get("person_bbox_xyxy_clipped") or person.get("bbox_xyxy")

                    if not valid_box(pb):
                        continue

                    a, b, c, d = pb

                    if a <= cx <= c and b <= cy <= b + 0.6 * (d - b):
                        matches.append(person)

                if len(matches) == 1 and matches[0] is target:
                    face_box = [cx - 2, cy - 2, cx + 2, cy + 2]
                    return target, face_box

    return None, None


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_matches = 0

    for image in data["images"]:
        people = image["v7"]["people"]
        target, face_box = find_unique_face(people)

        print(f"\nIMAGE: {Path(image['image']).name}")
        print(f"People: {len(people)}")

        if target is None:
            print("SKIP: No unambiguous synthetic face position")
            continue

        face_tracks = [
            (101, face_box, "SV_TEST_001", 0.95)
        ]

        results = associate_faces_to_people(face_tracks, people)

        assert len(results) == 1, "Expected exactly one match"

        result = results[0]

        assert result["person_ref"] == target["person_ref"]
        assert result["track_id"] == 101
        assert result["student_id"] == "SV_TEST_001"
        assert result["behaviors"] == target.get("behaviors", {})
        assert result["association_status"] == "matched"

        print("PASS")
        print("Matched person:", result["person_ref"])
        print("Student:", result["student_id"])
        print("Behaviors:", {
            k: v.get("state")
            for k, v in result["behaviors"].items()
        })

        total_matches += 1

    print(f"\nTOTAL MATCHED IMAGES: {total_matches}/{len(data['images'])}")

    assert total_matches > 0, "No association was tested"


if __name__ == "__main__":
    main()
