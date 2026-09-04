import cv2

from gauge_detector import detect_gauge


image_path = "data/raw/(15)LO Inlet Press.jpg"

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not load image")
    exit()

print("✅ Image loaded:", image.shape)

result = detect_gauge(image)

print("\n🔍 Gauge detection result:")
print(result)


# ---------------------------------------
# Draw detected gauge
# ---------------------------------------

if result["gauge_detected"]:

    x, y = result["center"]
    radius = result["radius"]

    output = image.copy()

    # Draw detected gauge
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

    output_path = (
        "data/processed/"
        "automatic_gauge_detection.jpg"
    )

    cv2.imwrite(
        output_path,
        output
    )

    print("\n✅ Debug image saved to:")
    print(output_path)

else:

    print("\n❌ Gauge was not detected.")