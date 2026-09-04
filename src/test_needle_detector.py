import cv2
import math

from gauge_detector import detect_gauge
from needle_detector import detect_needle


image_path = "data/raw/(15)LO Inlet Press.jpg"

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not load image")
    exit()

print("✅ Image loaded:", image.shape)


# ---------------------------------------
# STEP 1: Detect gauge automatically
# ---------------------------------------

gauge = detect_gauge(image)

print("\n🔵 Gauge detection:")
print(gauge)

if not gauge["gauge_detected"]:
    print("❌ Gauge not detected")
    exit()


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

print("\n🟢 Needle detection:")
print(needle)


# ---------------------------------------
# Draw result
# ---------------------------------------

output = image.copy()

x, y = center

# Gauge circle
cv2.circle(
    output,
    (x, y),
    radius,
    (0, 255, 0),
    8
)

# Gauge center
cv2.circle(
    output,
    (x, y),
    15,
    (0, 0, 255),
    -1
)


if needle["needle_detected"]:

    angle = needle["needle_angle"]

    theta = math.radians(angle)

    needle_length = int(
        radius * 0.72
    )

    end_x = int(
        x +
        needle_length *
        math.cos(theta)
    )

    end_y = int(
        y +
        needle_length *
        math.sin(theta)
    )

    # Draw needle
    cv2.line(
        output,
        (x, y),
        (end_x, end_y),
        (255, 0, 0),
        10
    )

    print("\n✅ Needle detected successfully")

else:

    print("\n❌ Needle not detected")


# ---------------------------------------
# Save
# ---------------------------------------

output_path = (
    "data/processed/"
    "automatic_needle_detection.jpg"
)

cv2.imwrite(
    output_path,
    output
)

print("\n✅ Result saved to:")
print(output_path)