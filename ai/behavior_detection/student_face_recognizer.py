
import sqlite3
from pathlib import Path

import numpy as np


# ============================================
# CONFIG
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "students_faces.db"

EMBEDDING_DIM = 512

# CH-20E: Experimental threshold
# Must be validated with more people before
# using for real student identification.
MATCH_THRESHOLD = 0.80

UNCERTAIN_THRESHOLD = 0.70


# ============================================
# STUDENT FACE RECOGNIZER
# ============================================

class StudentFaceRecognizer:

    def __init__(
        self,
        db_path=DB_PATH,
        match_threshold=MATCH_THRESHOLD,
        uncertain_threshold=UNCERTAIN_THRESHOLD
    ):

        self.db_path = Path(db_path)

        self.match_threshold = float(
            match_threshold
        )

        self.uncertain_threshold = float(
            uncertain_threshold
        )

        if not (
            -1.0 <= self.uncertain_threshold
            < self.match_threshold <= 1.0
        ):
            raise ValueError(
                "Invalid recognition thresholds"
            )

        self.students = {}
        self.embeddings = []

        self.load_database()

    def load_database(self):

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Face database not found: {self.db_path}"
            )

        self.students.clear()
        self.embeddings.clear()

        with sqlite3.connect(self.db_path) as conn:

            student_rows = conn.execute(
                """
                SELECT student_id, student_name
                FROM students
                """
            ).fetchall()

            embedding_rows = conn.execute(
                """
                SELECT student_id, embedding
                FROM face_embeddings
                """
            ).fetchall()

        for student_id, student_name in student_rows:

            self.students[student_id] = (
                student_name
            )

        for student_id, blob in embedding_rows:

            if student_id not in self.students:
                continue

            if len(blob) != EMBEDDING_DIM * 4:

                print(
                    "Skipping invalid embedding "
                    f"for {student_id}: "
                    f"{len(blob)} bytes"
                )

                continue

            vector = np.frombuffer(
                blob,
                dtype=np.float32
            ).copy()

            if (
                vector.size != EMBEDDING_DIM
                or not np.all(np.isfinite(vector))
            ):
                continue

            norm = float(
                np.linalg.norm(vector)
            )

            if norm <= 1e-8:
                continue

            vector /= norm

            self.embeddings.append({
                "student_id": student_id,
                "vector": vector
            })

        print(
            f"Students loaded: {len(self.students)}"
        )

        print(
            f"Valid embeddings: {len(self.embeddings)}"
        )

        print(
            "Recognition thresholds: "
            f"MATCH={self.match_threshold:.2f}, "
            f"UNCERTAIN={self.uncertain_threshold:.2f}"
        )

    def recognize(self, face_embedding):

        result = {
            "student_id": None,
            "student_name": None,
            "similarity": 0.0,
            "status": "UNKNOWN"
        }

        if face_embedding is None:
            return result

        vector = np.asarray(
            face_embedding,
            dtype=np.float32
        ).reshape(-1)

        if (
            vector.size != EMBEDDING_DIM
            or not np.all(np.isfinite(vector))
        ):
            return result

        norm = float(
            np.linalg.norm(vector)
        )

        if norm <= 1e-8:
            return result

        vector = vector / norm

        if not self.embeddings:
            return result

        scores = {}

        for item in self.embeddings:

            student_id = item["student_id"]

            similarity = float(
                np.dot(
                    vector,
                    item["vector"]
                )
            )

            scores[student_id] = max(
                scores.get(student_id, -1.0),
                similarity
            )

        best_student_id = max(
            scores,
            key=scores.get
        )

        best_similarity = scores[
            best_student_id
        ]

        result["similarity"] = (
            best_similarity
        )

        if best_similarity >= self.match_threshold:

            result["student_id"] = (
                best_student_id
            )

            result["student_name"] = (
                self.students[best_student_id]
            )

            result["status"] = "MATCH"

        elif (
            best_similarity
            >= self.uncertain_threshold
        ):

            result["status"] = "UNCERTAIN"

        return result


# ============================================
# TEST DATABASE LOADING
# ============================================

if __name__ == "__main__":

    recognizer = StudentFaceRecognizer()

    print()
    print("CH-20E recognizer initialized")
    print("Ready for camera recognition test")
