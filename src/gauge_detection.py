from pathlib import Path
from ultralytics import YOLO


def detect_gauges(image, model_path: str):
    """Run YOLO detection on the input image."""

    if not Path(model_path).exists():
        return {
            "status": "model_missing",
            "message": f"YOLO model not found: {model_path}",
            "detections": []
        }

    # Load YOLO model
    model = YOLO(model_path)

    # Run detection
    results = model(image, verbose=False)

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

    return {
        "status": "success",
        "detections": detections
    }