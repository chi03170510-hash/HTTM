from person_face_association import associate_faces_to_people


def make_person(ref, box):
    return {
        "person_ref": ref,
        "bbox_xyxy": box,
        "behaviors": {
            "using_phone": {"state": "detected"}
        },
    }


def test_single_person():
    faces = [(1, (40, 20, 80, 70), "SV001", 0.82)]
    people = [make_person("P1", (20, 10, 120, 220))]

    result = associate_faces_to_people(faces, people)

    assert len(result) == 1
    assert result[0]["student_id"] == "SV001"
    assert result[0]["person_ref"] == "P1"


def test_two_people():
    faces = [
        (1, (40, 20, 80, 70), "SV001", 0.82),
        (2, (240, 20, 280, 70), "SV002", 0.78),
    ]
    people = [
        make_person("P1", (20, 10, 120, 220)),
        make_person("P2", (220, 10, 320, 220)),
    ]

    result = associate_faces_to_people(faces, people)

    assert len(result) == 2
    assert {r["student_id"] for r in result} == {"SV001", "SV002"}


def test_ambiguous_people():
    faces = [(1, (90, 20, 110, 70), "SV001", 0.82)]
    people = [
        make_person("P1", (20, 10, 150, 220)),
        make_person("P2", (70, 10, 200, 220)),
    ]

    result = associate_faces_to_people(faces, people)

    assert result == []


def test_two_faces_one_person():
    faces = [
        (1, (40, 20, 80, 70), "SV001", 0.82),
        (2, (70, 20, 100, 70), "SV002", 0.78),
    ]
    people = [make_person("P1", (20, 10, 150, 220))]

    result = associate_faces_to_people(faces, people)

    assert result == []


def test_invalid_box():
    faces = [(1, (40, 20, 40, 70), "SV001", 0.82)]
    people = [make_person("P1", (20, 10, 120, 220))]

    assert associate_faces_to_people(faces, people) == []


if __name__ == "__main__":
    test_single_person()
    test_two_people()
    test_ambiguous_people()
    test_two_faces_one_person()
    test_invalid_box()
    print("PASS: 5/5 association tests")
