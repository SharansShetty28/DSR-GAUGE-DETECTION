import json
import sys

from config import YOLO_MODEL_PATH
from src.preprocessing import preprocess_image
from src.gauge_detection import detect_gauges
from src.gauge_classification import classify_gauge
from src.ocr import read_text_and_numbers
from src.analog_reader import read_analog_gauge
from src.vision_llm import interpret_gauge
from src.validation import validate_result


def run_pipeline(image_path: str):
    image = preprocess_image(image_path)

    detection = detect_gauges(image, str(YOLO_MODEL_PATH))
    interpretation = interpret_gauge(image_path)

    result = {
        "equipment": interpretation["equipment"],
        "parameter": interpretation["parameter"],
        "value": None,
        "unit": interpretation["unit"],
        "gauge_type": "unknown",
        "confidence": 0.0,
        "detection": detection,
    }

    # Once YOLO returns a crop, we will process the crop here.
    if detection.get("detections"):
        gauge_crop = detection["detections"][0].get("crop")
        gauge_type = classify_gauge(gauge_crop)
        result["gauge_type"] = gauge_type["type"]

        if gauge_type["type"] == "digital":
            ocr_result = read_text_and_numbers(gauge_crop)
            result["value"] = ocr_result["numbers"][0] if ocr_result["numbers"] else None
            result["confidence"] = ocr_result["confidence"]

        elif gauge_type["type"] == "analog":
            analog_result = read_analog_gauge(gauge_crop)
            result["value"] = analog_result["value"]
            result["unit"] = analog_result["unit"] or result["unit"]
            result["confidence"] = analog_result["confidence"]

    return validate_result(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/main.py path/to/gauge_image.jpg")
        raise SystemExit(1)

    output = run_pipeline(sys.argv[1])
    print(json.dumps(output, indent=2))
