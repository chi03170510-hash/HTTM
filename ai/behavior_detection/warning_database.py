
import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "behavior_warnings.db"


class WarningDatabase:

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.create_tables()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS behavior_warnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    student_id TEXT,
                    behavior TEXT NOT NULL,
                    duration REAL NOT NULL,
                    warning_time TEXT NOT NULL,
                    image_path TEXT
                )
            """)

    def add_warning(
        self,
        track_id,
        behavior,
        duration,
        image_path=None,
        student_id=None
    ):
        warning_time = datetime.now().isoformat(
            timespec="seconds"
        )

        with self.connect() as conn:
            cursor = conn.execute("""
                INSERT INTO behavior_warnings (
                    track_id,
                    student_id,
                    behavior,
                    duration,
                    warning_time,
                    image_path
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                student_id,
                behavior,
                duration,
                warning_time,
                str(image_path) if image_path else None
            ))

            return cursor.lastrowid

    def get_all_warnings(self):
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT *
                FROM behavior_warnings
                ORDER BY id DESC
            """).fetchall()

            return [dict(row) for row in rows]


if __name__ == "__main__":
    database = WarningDatabase()
    print("Database initialized:", DB_PATH)
