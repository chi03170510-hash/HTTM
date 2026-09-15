import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_NAME = os.getenv("DB_NAME", "httm")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

raw_model_path = os.getenv("MODEL_PATH", "../ai/eye_detection/models/face_landmarker.task")
if os.path.isabs(raw_model_path):
    MODEL_PATH = raw_model_path
else:
    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, raw_model_path))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "violations")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.10))