import math


def valid_box(box):
    try:
        if box is None or len(box) != 4:
            return False
        x1, y1, x2, y2 = map(float, box)
        return (
            all(math.isfinite(v) for v in (x1, y1, x2, y2))
            and x2 > x1
            and y2 > y1
        )
    except (TypeError, ValueError):
        return False


def area(box):
    return (
        (float(box[2]) - float(box[0]))
        * (float(box[3]) - float(box[1]))
    )


def face_overlap(face_box, person_box):
    x1 = max(face_box[0], person_box[0])
    y1 = max(face_box[1], person_box[1])
    x2 = min(face_box[2], person_box[2])
    y2 = min(face_box[3], person_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    return intersection / area(face_box)


def associate_v2(face_tracks, people):
    proposals = []

    for track_id, face_box, student_id, score in face_tracks:
        if not valid_box(face_box):
            continue

        candidates = []

        for person in people:
            box = (
                person.get("person_bbox_xyxy_clipped")
                or person.get("bbox_xyxy")
            )

            if not valid_box(box):
                continue

            overlap = face_overlap(face_box, box)

            if overlap >= 0.95:
                candidates.append({
                    "person": person,
                    "box": box,
                    "overlap": overlap,
                    "area": area(box),
                })

        candidates.sort(key=lambda x: x["area"])

        if not candidates:
            continue

        if len(candidates) > 1:
            smallest = candidates[0]["area"]
            second = candidates[1]["area"]

            if second / smallest < 2.0:
                continue

        selected = candidates[0]

        proposals.append({
            "track_id": track_id,
            "student_id": student_id,
            "face_score": score,
            "face_bbox": face_box,
            "person_ref": selected["person"]["person_ref"],
            "person_bbox": selected["box"],
            "behaviors": selected["person"]["behaviors"],
            "association_status": "matched",
        })

    counts = {}

    for proposal in proposals:
        ref = proposal["person_ref"]
        counts[ref] = counts.get(ref, 0) + 1

    return [
        proposal
        for proposal in proposals
        if counts[proposal["person_ref"]] == 1
    ]
