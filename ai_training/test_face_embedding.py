import cv2
import time
import numpy as np
from insightface.app import FaceAnalysis

CAMERA_INDEX = 2

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(320, 320))

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open iPhone camera")

print("Camera ready. Press Q to quit.")

last_inference = 0.0
last_boxes = []
last_latency = 0.0
last_embedding_shape = None
last_embedding_norm = None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Cannot read camera frame")
            break

        now = time.perf_counter()

        if now - last_inference >= 0.5:
            start = time.perf_counter()

            faces = app.get(frame)
            last_boxes = []
            last_embedding_shape = None
            last_embedding_norm = None

            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding

                if embedding is not None:
                    last_boxes.append(tuple(bbox))
                    last_embedding_shape = embedding.shape
                    last_embedding_norm = float(
                        np.linalg.norm(embedding)
                    )

            last_latency = (
                time.perf_counter() - start
            ) * 1000

            last_inference = time.perf_counter()

        for x1, y1, x2, y2 in last_boxes:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Embedding OK",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        cv2.putText(
            frame,
            f"Faces: {len(last_boxes)}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"Inference: {last_latency:.0f} ms",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow("HTTM CH06 Embedding Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if last_embedding_shape is not None:
            print(
                f"\rEmbedding shape: {last_embedding_shape} "
                f"| Norm: {last_embedding_norm:.3f} "
                f"| Latency: {last_latency:.0f} ms",
                end="",
                flush=True
            )

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest finished.")
