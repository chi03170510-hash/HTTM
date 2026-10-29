
import sqlite3
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "students_faces.db"

STUDENT_ID = "001"


def main():
    if not DB_PATH.exists():
        print("[ERROR] Database not found")
        return

    conn = sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True
    )

    try:
        rows = conn.execute(
            """
            SELECT id, embedding
            FROM face_embeddings
            WHERE student_id = ?
            ORDER BY id
            """,
            (STUDENT_ID,)
        ).fetchall()
    finally:
        conn.close()

    sample_ids = []
    embeddings = []

    for sample_id, blob in rows:
        vector = np.frombuffer(
            blob, dtype=np.float32
        ).copy()

        if (
            vector.shape != (512,)
            or not np.all(np.isfinite(vector))
        ):
            print(f"[WARNING] Invalid sample {sample_id}")
            continue

        norm = np.linalg.norm(vector)

        if norm <= 1e-8:
            print(f"[WARNING] Zero sample {sample_id}")
            continue

        sample_ids.append(sample_id)
        embeddings.append(vector / norm)

    if len(embeddings) < 2:
        print("[ERROR] Need at least 2 valid samples")
        return

    matrix = np.stack(embeddings)
    similarity = matrix @ matrix.T

    n = len(matrix)

    # Exclude self-comparison from averages
    average_scores = (
        similarity.sum(axis=1) - 1
    ) / (n - 1)

    print("\nHTTM - EMBEDDING QUALITY REPORT")
    print("=" * 45)

    for i, score in enumerate(average_scores):
        print(
            f"Sample ID {sample_ids[i]} | "
            f"Average similarity: {score:.4f}"
        )

    weakest_index = int(np.argmin(average_scores))

    print("\nLowest average similarity:")
    print("Sample ID:", sample_ids[weakest_index])
    print(
        "Average:",
        f"{average_scores[weakest_index]:.4f}"
    )

    # Find lowest similarity pair
    upper = np.triu_indices(n, k=1)
    pair_scores = similarity[upper]

    weakest_pair_index = int(np.argmin(pair_scores))

    i = upper[0][weakest_pair_index]
    j = upper[1][weakest_pair_index]

    print("\nLowest similarity pair:")
    print("Sample A:", sample_ids[i])
    print("Sample B:", sample_ids[j])
    print(
        "Similarity:",
        f"{similarity[i, j]:.4f}"
    )

    print("\nAnalysis completed")
    print("No database records were modified.")


if __name__ == "__main__":
    main()
