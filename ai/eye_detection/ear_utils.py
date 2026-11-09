import math


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    return distance

def get_point(face_landmarks, index, width, height):
    landmark = face_landmarks[index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (x, y)

def get_eye_points(face_landmarks, eye_indices, width, height):
    eye_points = []

    for index in eye_indices:
        point = get_point(
            face_landmarks,
            index,
            width,
            height
        )

        eye_points.append(point)

    return eye_points

def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear