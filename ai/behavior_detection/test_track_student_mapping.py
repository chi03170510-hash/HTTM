
import sqlite3
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "students_faces.db"


# ==========================================
# LOAD STUDENTS FROM SQLITE
# ==========================================
def load_students():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True
    )

    try:
        rows = conn.execute("""
            SELECT student_id, student_name
            FROM students
            ORDER BY student_id
        """).fetchall()
    finally:
        conn.close()

    return {
        student_id: student_name
        for student_id, student_name in rows
    }


# ==========================================
# TRACK-STUDENT MAPPING
# ==========================================
class TrackStudentMapper:
    def __init__(self, students):
        self.students = students
        self.mapping = {}
        self.active_tracks = set()

    def add_track(self, track_id):
        if track_id in self.active_tracks:
            print(f"Track {track_id} already active")
            return

        self.active_tracks.add(track_id)
        print(f"Created Track {track_id}")

    def assign_student(self, track_id, student_id):
        if track_id not in self.active_tracks:
            print("ERROR: Track does not exist")
            return False

        if student_id not in self.students:
            print("ERROR: Student ID not found in SQLite")
            return False

        for other_track, assigned_id in self.mapping.items():
            if (
                other_track != track_id
                and assigned_id == student_id
            ):
                print(
                    f"ERROR: Student {student_id} "
                    f"already assigned to Track {other_track}"
                )
                return False

        self.mapping[track_id] = student_id

        print(
            f"Assigned Track {track_id} "
            f"-> Student {student_id}"
        )

        return True

    def remove_track(self, track_id):
        self.active_tracks.discard(track_id)

        if track_id in self.mapping:
            del self.mapping[track_id]

        print(f"Removed Track {track_id}")

    def show_mapping(self):
        print("\nACTIVE TRACK-STUDENT MAPPING")
        print("=" * 50)

        if not self.active_tracks:
            print("No active tracks")
            return

        for track_id in sorted(self.active_tracks):
            student_id = self.mapping.get(track_id)

            if student_id is None:
                print(
                    f"Track {track_id} -> UNASSIGNED"
                )
            else:
                student_name = self.students[student_id]

                print(
                    f"Track {track_id} -> "
                    f"Student {student_id} "
                    f"({student_name})"
                )


# ==========================================
# TEST SCENARIOS
# ==========================================
def main():
    print("=" * 50)
    print("HTTM - CH-12 TRACK-STUDENT MAPPING")
    print("=" * 50)

    try:
        students = load_students()
    except (sqlite3.Error, OSError) as error:
        print(f"ERROR: {error}")
        return

    print(f"\nStudents loaded: {len(students)}")

    for student_id, name in students.items():
        print(f"Student {student_id}: {name}")

    mapper = TrackStudentMapper(students)

    print("\nTEST 1: Create tracks")
    mapper.add_track(1)
    mapper.add_track(2)
    mapper.add_track(3)
    mapper.show_mapping()

    print("\nTEST 2: Assign registered student")
    mapper.assign_student(1, "001")
    mapper.show_mapping()

    print("\nTEST 3: Prevent duplicate assignment")
    mapper.assign_student(2, "001")
    mapper.show_mapping()

    print("\nTEST 4: Reject unknown student")
    mapper.assign_student(3, "999")
    mapper.show_mapping()

    print("\nTEST 5: Remove Track 1")
    mapper.remove_track(1)
    mapper.show_mapping()

    print("\nTEST 6: Reassign Student 001")
    mapper.assign_student(2, "001")
    mapper.show_mapping()

    print("\nCH-12 test completed")
    print("No database records were modified.")


if __name__ == "__main__":
    main()
