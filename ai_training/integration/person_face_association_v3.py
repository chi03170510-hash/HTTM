from person_face_association import associate_faces_to_people
from person_face_association_v2 import associate_v2


def associate_v3(face_tracks, people):
    v1 = associate_faces_to_people(face_tracks, people)
    v2 = associate_v2(face_tracks, people)

    by_face_v1 = {x["track_id"]: x for x in v1}
    by_face_v2 = {x["track_id"]: x for x in v2}

    proposals = []

    for track_id in set(by_face_v1) | set(by_face_v2):
        a = by_face_v1.get(track_id)
        b = by_face_v2.get(track_id)

        if a is not None and b is not None:
            if a["person_ref"] != b["person_ref"]:
                continue
            selected = a
        else:
            selected = a if a is not None else b

        proposals.append(selected)

    # Reject duplicate person assignments.
    counts = {}

    for item in proposals:
        ref = item["person_ref"]
        counts[ref] = counts.get(ref, 0) + 1

    return [
        item for item in proposals
        if counts[item["person_ref"]] == 1
    ]
