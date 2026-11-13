from pathlib import Path
import re

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "face_enrollment"

VALID_ID = re.compile(r"^SV[0-9]{3,8}$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_student_id(student_id):
    student_id = student_id.strip().upper()

    if not VALID_ID.fullmatch(student_id):
        raise ValueError(
            "Student ID must have format SV001, SV002, ..."
        )

    return student_id


def load_face_model():
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    return app


def extract_embedding(app, image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    faces = app.get(image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly 1 face, found {len(faces)}"
        )

    embedding = faces[0].embedding

    if embedding is None:
        raise ValueError("Face embedding unavailable")

    embedding = np.asarray(
        embedding, dtype=np.float32
    )

    if embedding.shape != (512,):
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError("Embedding contains invalid values")

    norm = float(np.linalg.norm(embedding))

    if norm <= 1e-8:
        raise ValueError("Embedding norm is zero")

    return embedding / norm


def save_embeddings(student_id, embeddings):
    student_id = validate_student_id(student_id)

    matrix = np.asarray(
        embeddings, dtype=np.float32
    )

    if matrix.ndim != 2 or matrix.shape[1] != 512:
        raise ValueError("Expected embedding matrix Nx512")

    if matrix.shape[0] == 0:
        raise ValueError("No embeddings to save")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid embedding values")

    student_dir = DATA_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    output_path = student_dir / "embeddings.npz"

    if output_path.exists():
        raise FileExistsError(
            f"Enrollment already exists: {output_path}"
        )

    np.savez_compressed(
        output_path,
        student_id=np.array(student_id),
        embeddings=matrix
    )

    return output_path
