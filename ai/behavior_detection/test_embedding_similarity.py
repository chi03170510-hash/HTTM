
import sqlite3
import logging
import numpy as np
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "students_faces.db"

EMBEDDING_DIM = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("HTTM-Embedding-Test")


# ==========================================
# COSINE SIMILARITY
# ==========================================
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero-length embedding")

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


# ==========================================
# LOAD DATABASE
# ==========================================
def load_embeddings():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    # Read-only connection
    db_uri = DB_PATH.resolve().as_uri() + "?mode=ro"

    conn = sqlite3.connect(
        db_uri,
        uri=True
    )

    try:
        rows = conn.execute("""
            SELECT
                s.student_id,
                f.embedding
            FROM face_embeddings f
            JOIN students s
                ON s.student_id = f.student_id
            ORDER BY s.student_id, f.id
        """).fetchall()
    finally:
        conn.close()

    students = {}

    for student_id, blob in rows:
        embedding = np.frombuffer(
            blob,
            dtype=np.float32
        )

        if embedding.shape != (EMBEDDING_DIM,):
            logger.warning(
                "Invalid embedding for student %s",
                student_id
            )
            continue

        if not np.all(np.isfinite(embedding)):
            logger.warning(
                "Non-finite embedding for student %s",
                student_id
            )
            continue

        if np.linalg.norm(embedding) == 0:
            logger.warning(
                "Zero embedding for student %s",
                student_id
            )
            continue

        students.setdefault(
            student_id, []
        ).append(embedding.copy())

    return students


# ==========================================
# ANALYZE EMBEDDINGS
# ==========================================
def analyze_student(student_id, embeddings):
    count = len(embeddings)

    logger.info(
        "Student %s | Samples: %d",
        student_id,
        count
    )

    if count < 2:
        logger.warning(
            "Need at least 2 samples for comparison"
        )
        return

    similarities = []

    for i in range(count):
        for j in range(i + 1, count):
            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            similarities.append(similarity)

    scores = np.array(similarities)

    logger.info(
        "Pairs compared: %d",
        len(scores)
    )

    logger.info(
        "Min similarity: %.4f",
        np.min(scores)
    )

    logger.info(
        "Mean similarity: %.4f",
        np.mean(scores)
    )

    logger.info(
        "Max similarity: %.4f",
        np.max(scores)
    )

    logger.info(
        "Std deviation: %.4f",
        np.std(scores)
    )


# ==========================================
# MAIN
# ==========================================
def main():
    logger.info("=" * 45)
    logger.info("HTTM - EMBEDDING QUALITY TEST")
    logger.info("=" * 45)

    try:
        students = load_embeddings()

        logger.info(
            "Students loaded: %d",
            len(students)
        )

        total_samples = sum(
            len(samples)
            for samples in students.values()
        )

        logger.info(
            "Valid embeddings: %d",
            total_samples
        )

        for student_id, embeddings in students.items():
            analyze_student(
                student_id,
                embeddings
            )

        logger.info("Analysis completed")

    except Exception:
        logger.exception(
            "Embedding analysis failed"
        )


if __name__ == "__main__":
    main()
