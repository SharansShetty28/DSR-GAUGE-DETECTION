import os
import re
import cv2

from analog_reader import read_analog_gauge


RAW_DIR = "data/raw"


for filename in sorted(os.listdir(RAW_DIR)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(RAW_DIR, filename)

    print("\n" + "=" * 60)
    print("Processing:", filename)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image")
        continue

    # Remove file extension
    gauge_name = os.path.splitext(filename)[0]

    # Remove leading number such as:
    # (1)
    # (14)
    # (15)
    # (26.1)
    gauge_name = re.sub(
        r"^\(\d+(?:\.\d+)?\)",
        "",
        gauge_name
    )

    # Remove duplicate "(1)" at the end
    gauge_name = re.sub(
        r"\s+\(1\)$",
        "",
        gauge_name
    )

    gauge_name = gauge_name.strip()

    result = read_analog_gauge(
        image,
        gauge_name
    )

    print("Gauge name:", gauge_name)
    print("Value:", result.get("value"))
    print("Unit:", result.get("unit"))
    print("Needle angle:", result.get("needle_angle"))
    print("Confidence:", result.get("confidence"))

    if result.get("error"):
        print("Error:", result.get("error"))