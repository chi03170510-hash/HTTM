import csv
import statistics


CSV_PATH = "data/ear_test.csv"

open_ears = []
closed_ears = []


with open(CSV_PATH, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        state = row["state"]
        avg_ear = float(row["avg_ear"])

        if state == "OPEN":
            open_ears.append(avg_ear)

        elif state == "CLOSED":
            closed_ears.append(avg_ear)


print("===== OPEN =====")
print("Count:", len(open_ears))
print("Mean:", statistics.mean(open_ears))
print("Min:", min(open_ears))
print("Max:", max(open_ears))
print("Std:", statistics.stdev(open_ears))


print("\n===== CLOSED =====")
print("Count:", len(closed_ears))
print("Mean:", statistics.mean(closed_ears))
print("Min:", min(closed_ears))
print("Max:", max(closed_ears))
print("Std:", statistics.stdev(closed_ears))

print("\n===== THRESHOLD TEST =====")

thresholds = [
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13
]

for threshold in thresholds:

    correct_open = sum(
        1 for ear in open_ears
        if ear >= threshold
    )

    correct_closed = sum(
        1 for ear in closed_ears
        if ear < threshold
    )

    total_correct = (
        correct_open +
        correct_closed
    )

    total_samples = (
        len(open_ears) +
        len(closed_ears)
    )

    accuracy = total_correct / total_samples

    print(
        f"Threshold {threshold:.2f}"
        f" -> Accuracy: {accuracy:.4f}"
    )

print("\n===== CONFUSION MATRIX TEST =====")

for threshold in thresholds:

    # CLOSED được quy ước là Positive

    # TP:
    # Thực tế CLOSED, dự đoán CLOSED
    tp = sum(
        1 for ear in closed_ears
        if ear < threshold
    )

    # FN:
    # Thực tế CLOSED, nhưng dự đoán OPEN
    fn = sum(
        1 for ear in closed_ears
        if ear >= threshold
    )

    # TN:
    # Thực tế OPEN, dự đoán OPEN
    tn = sum(
        1 for ear in open_ears
        if ear >= threshold
    )

    # FP:
    # Thực tế OPEN, nhưng dự đoán CLOSED
    fp = sum(
        1 for ear in open_ears
        if ear < threshold
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(
        f"Threshold {threshold:.2f}"
        f" -> TP={tp}, TN={tn}, FP={fp}, FN={fn}"
        f" | Precision={precision:.4f}"
        f" Recall={recall:.4f}"
        f" F1={f1:.4f}"
    )