import math

from person_face_association_v3 import associate_v3


def associate_v4(face_tracks, people, max_distance=0.30, min_margin=0.15):
    base_matches = associate_v3(face_tracks, people)

    matched_faces = {m["track_id"] for m in base_matches}
    used_people = {m["person_ref"] for m in base_matches}

    proposals = []

    for track_id, face_box, student_id, score in face_tracks:
        if track_id in matched_faces:
            continue

        if face_box is None or len(face_box) != 4:
            continue

        fx1, fy1, fx2, fy2 = map(float, face_box)

        if not all(math.isfinite(v) for v in (fx1, fy1, fx2, fy2)):
            continue

        if fx2 <= fx1 or fy2 <= fy1:
            continue

        cx = (fx1 + fx2) / 2
        cy = (fy1 + fy2) / 2

        candidates = []

        for person in people:
            box = (
                person.get("person_bbox_xyxy_clipped")
                or person.get("bbox_xyxy")
            )

            if box is None or len(box) != 4:
                continue

            try:
                x1, y1, x2, y2 = map(float, box)
            except (TypeError, ValueError):
                continue

            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            rx = (cx - x1) / (x2 - x1)
            ry = (cy - y1) / (y2 - y1)

            if not (0 <= rx <= 1 and 0 <= ry <= 0.6):
                continue

            distance = math.hypot(rx - 0.5, ry - 0.25)

            candidates.append((distance, person, box))

        candidates.sort(key=lambda item: item[0])

        if not candidates:
            continue

        best_distance, best_person, best_box = candidates[0]

        if best_distance > max_distance:
            continue

        if len(candidates) > 1:
            margin = candidates[1][0] - best_distance
            if margin < min_margin:
                continue

        if best_person["person_ref"] in used_people:
            continue

        proposals.append({
            "track_id": track_id,
            "student_id": student_id,
            "face_score": score,
            "face_bbox": face_box,
            "person_ref": best_person["person_ref"],
            "person_bbox": best_box,
            "behaviors": best_person.get("behaviors", {}),
            "association_status": "matched_v4_heuristic",
            "association_distance": round(best_distance, 4),
        })

    # Reject multiple new faces proposing the same person.
    counts = {}

    for item in proposals:
        ref = item["person_ref"]
        counts[ref] = counts.get(ref, 0) + 1

    accepted = [
        item for item in proposals
        if counts[item["person_ref"]] == 1
    ]

    return base_matches + accepted
