import argparse
import json
from pathlib import Path

from person_face_association import associate_faces_to_people


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    face_data = json.loads(args.face.read_text(encoding="utf-8"))
    hybrid_data = json.loads(args.hybrid.read_text(encoding="utf-8"))

    if face_data["frame_id"] != hybrid_data["frame_id"]:
        raise RuntimeError("Frame ID mismatch")

    if Path(face_data["image"]).resolve() != Path(
        hybrid_data["image"]
    ).resolve():
        raise RuntimeError("Image path mismatch")

    face_tracks = []

    for index, face in enumerate(face_data["faces"]):
        track_id = face.get("track_id")

        # Temporary ID for static-image testing only.
        if track_id is None:
            track_id = f"static_face_{index}"

        face_tracks.append((
            track_id,
            face["face_bbox"],
            face["student_id"],
            face["score"],
        ))

    matches = associate_faces_to_people(
        face_tracks,
        hybrid_data["people"],
    )

    result = {
        "frame_id": face_data["frame_id"],
        "image": face_data["image"],
        "face_count": len(face_tracks),
        "person_count": len(hybrid_data["people"]),
        "matched_count": len(matches),
        "matches": matches,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Merge completed")
    print("Faces:", result["face_count"])
    print("People:", result["person_count"])
    print("Matched:", result["matched_count"])

    for match in matches:
        print(
            match["track_id"],
            "->",
            match["person_ref"],
            "|",
            match["student_id"],
        )


if __name__ == "__main__":
    main()
