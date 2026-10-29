
import csv
import logging
from datetime import datetime
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "unknown_test.log"
CSV_PATH = LOG_DIR / "unknown_test_results.csv"

# Example thresholds for testing control flow only.
# NOT calibrated for real face recognition.
MATCH_THRESHOLD = 0.85
UNKNOWN_THRESHOLD = 0.65

TEST_SCORES = [
    0.95,
    0.89,
    0.85,
    0.82,
    0.74,
    0.65,
    0.60,
    0.40,
    -0.10,
    None,
    float("nan"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_PATH,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HTTM-Unknown-Test")


# ==========================================
# CLASSIFY SIMILARITY
# ==========================================
def classify_similarity(score):
    if score is None:
        return "INVALID"

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "INVALID"

    if not (-1.0 <= score <= 1.0):
        return "INVALID"

    if score >= MATCH_THRESHOLD:
        return "MATCH"

    if score < UNKNOWN_THRESHOLD:
        return "UNKNOWN"

    return "UNCERTAIN"


# ==========================================
# MAIN
# ==========================================
def main():
    logger.info("=" * 50)
    logger.info("HTTM - UNKNOWN HANDLING TEST")
    logger.info("=" * 50)

    results = []
    counts = {
        "MATCH": 0,
        "UNCERTAIN": 0,
        "UNKNOWN": 0,
        "INVALID": 0
    }

    for index, score in enumerate(TEST_SCORES, start=1):
        status = classify_similarity(score)
        counts[status] += 1

        logger.info(
            "Test %02d | Score: %s | Status: %s",
            index,
            score,
            status
        )

        results.append({
            "test_id": index,
            "similarity": score,
            "status": status,
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            )
        })

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_id",
                "similarity",
                "status",
                "timestamp"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    logger.info("-" * 50)

    for status, count in counts.items():
        logger.info("%s: %d", status, count)

    logger.info("CSV saved: %s", CSV_PATH)
    logger.info("Log saved: %s", LOG_PATH)
    logger.info("Test completed")


if __name__ == "__main__":
    main()
