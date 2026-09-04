import os
import cv2
import math

from gauge_detector import detect_gauge
from scale_detector import detect_scale


RAW_DIR = "data/raw"


for filename in sorted(os.listdir(RAW_DIR)):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    print("\n" + "=" * 70)
    print("Processing:", filename)

    image_path = os.path.join(
        RAW_DIR,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Could not load image")
        continue

    print("Image size:", image.shape)

    # ---------------------------------------
    # STEP 1: Detect gauge
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
    # STEP 2: Detect MAJOR scale lines
    # ---------------------------------------

    scale = detect_scale(
        image,
        center,
        radius
    )

    print("\nScale:")
    print(scale)

    # IMPORTANT:
    # New detector returns "major_angles"
    major_angles = scale["major_angles"]

    if major_angles:

        print(
            "✅ Major scale lines detected:",
            len(major_angles)
        )

        print(
            "Major angles:",
            major_angles
        )

    else:

        print("❌ No major scale lines detected")

    # ---------------------------------------
    # STEP 3: Draw MAJOR scale lines
    # ---------------------------------------

    output = image.copy()

    center_x, center_y = center

    for angle in major_angles:

        theta = math.radians(angle)

        # -----------------------------------
        # Major tick position
        # -----------------------------------

        inner_r = int(radius * 0.68)
        outer_r = int(radius * 0.88)

        x1 = int(
            center_x +
            inner_r * math.cos(theta)
        )

        y1 = int(
            center_y +
            inner_r * math.sin(theta)
        )

        x2 = int(
            center_x +
            outer_r * math.cos(theta)
        )

        y2 = int(
            center_y +
            outer_r * math.sin(theta)
        )

        # -----------------------------------
        # Draw detected major line
        # Yellow
        # -----------------------------------

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            6
        )

    # ---------------------------------------
    # STEP 4: Draw gauge boundary
    # ---------------------------------------

    cv2.circle(
        output,
        (center_x, center_y),
        radius,
        (0, 255, 0),
        5
    )

    # ---------------------------------------
    # STEP 5: Draw gauge center
    # ---------------------------------------

    cv2.circle(
        output,
        (center_x, center_y),
        12,
        (0, 0, 255),
        -1
    )

    # ---------------------------------------
    # STEP 6: Save result
    # ---------------------------------------

    output_dir = "data/processed/scale_detection"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_name = (
        os.path.splitext(filename)[0]
        + "_scale.jpg"
    )

    output_path = os.path.join(
        output_dir,
        output_name
    )

    cv2.imwrite(
        output_path,
        output
    )

    print(
        "✅ Saved:",
        output_path
    )