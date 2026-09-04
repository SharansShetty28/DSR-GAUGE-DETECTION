import os
import cv2

from gauge_detector import detect_gauge


RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed/gauge_detection"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------
# Process every image
# ---------------------------------------

for filename in sorted(os.listdir(RAW_DIR)):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(
        RAW_DIR,
        filename
    )

    print("\n" + "=" * 70)
    print("Processing:", filename)

    image = cv2.imread(
        image_path
    )

    if image is None:
        print("❌ Could not load image")
        continue

    print(
        "Image size:",
        image.shape
    )

    # -----------------------------------
    # Automatic gauge detection
    # -----------------------------------

    result = detect_gauge(
        image
    )

    print(
        "Gauge detected:",
        result["gauge_detected"]
    )

    print(
        "Center:",
        result["center"]
    )

    print(
        "Radius:",
        result["radius"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    # -----------------------------------
    # Draw detection
    # -----------------------------------

    if result["gauge_detected"]:

        x, y = result["center"]
        radius = result["radius"]

        output = image.copy()

        # Draw gauge circle
        cv2.circle(
            output,
            (x, y),
            radius,
            (0, 255, 0),
            8
        )

        # Draw center
        cv2.circle(
            output,
            (x, y),
            15,
            (0, 0, 255),
            -1
        )

        # --------------------------------
        # Create safe output filename
        # --------------------------------

        output_filename = (
            os.path.splitext(filename)[0]
            + "_detected.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_filename
        )

        cv2.imwrite(
            output_path,
            output
        )

        print(
            "✅ Saved:",
            output_path
        )

    else:

        print(
            "❌ Gauge not detected"
        )