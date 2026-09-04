from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = BASE_DIR / "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.50
