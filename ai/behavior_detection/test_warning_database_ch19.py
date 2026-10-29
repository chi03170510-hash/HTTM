
from warning_database import WarningDatabase


def main():
    database = WarningDatabase()
    warnings = database.get_all_warnings()

    print(f"Total warnings: {len(warnings)}")

    for warning in warnings:
        print("-" * 50)
        print("ID:", warning["id"])
        print("Track ID:", warning["track_id"])
        print("Student ID:", warning["student_id"])
        print("Behavior:", warning["behavior"])
        print("Duration:", warning["duration"])
        print("Time:", warning["warning_time"])
        print("Image:", warning["image_path"])


if __name__ == "__main__":
    main()
