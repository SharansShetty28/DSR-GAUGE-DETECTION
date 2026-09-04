import os
import cv2
import math

from gauge_detector import detect_gauge
from needle_detector import detect_needle


RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed/needle_detection"

os.makedirs(OUTPUT_DIR, exist_ok=True)


for filename in sorted(os.listdir(RAW_DIR)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(RAW_DIR, filename)

    print("\n" + "=" * 70)
    print("Processing:", filename)

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Could not load image")
        continue

    print("Image size:", image.shape)

    # ---------------------------------------
    # STEP 1: Detect gauge automatically
    # ---------------------------------------

    gauge = detect_gauge(image)

    print("\nGauge:")
    print(gauge)

    if not gauge["gauge_detected"]:
        print("❌ Gauge not detected")
        continue

    center = gauge["center"]
    radius = gauge["radius"]

    # ---------------------------------------
    # STEP 2: Detect needle automatically
    # ---------------------------------------

    needle = detect_needle(
        image,
        center,
        radius
    )

    print("\nNeedle:")
    print(needle)

    # ---------------------------------------
    # Draw result
    # ---------------------------------------

    output = image.copy()

    center_x, center_y = center

    # Draw gauge circle
    cv2.circle(
        output,
        (center_x, center_y),
        radius,
        (0, 255, 0),
        8
    )

    # Draw gauge center
    cv2.circle(
        output,
        (center_x, center_y),
        15,
        (0, 0, 255),
        -1
    )

    # Draw needle
    if needle["needle_detected"]:

        angle = needle["needle_angle"]

        theta = math.radians(angle)

        needle_length = int(
            radius * 0.72
        )

        end_x = int(
            center_x +
            needle_length *
            math.cos(theta)
        )

        end_y = int(
            center_y +
            needle_length *
            math.sin(theta)
        )

        cv2.line(
            output,
            (center_x, center_y),
            (end_x, end_y),
            (255, 0, 0),
            10
        )

        print("✅ Needle detected")

    else:

        print("❌ Needle not detected")

    # ---------------------------------------
    # Save result
    # ---------------------------------------

    output_filename = (
        os.path.splitext(filename)[0]
        + "_needle.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    cv2.imwrite(
        output_path,
        output
    )

    print("✅ Saved:", output_path)