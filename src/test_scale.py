import cv2
import numpy as np

from analog_reader import read_analog_gauge
from scale_detector import detect_scale


IMAGE_PATH = "data/raw/(15)LO Inlet Press.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Could not load image")
    exit()

print(f"✅ Image loaded: {image.shape}")

# Detect gauge and needle
result = read_analog_gauge(
    image,
    gauge_name="LO Inlet Press"
)

print("\n🔎 Gauge result:")
print(result)

center = result["gauge_center"]
radius = result["gauge_radius"]

# Detect scale
scale = detect_scale(
    image,
    center,
    radius
)

print("\n📏 Scale detection result:")
print(scale)

# --------------------------------
# Draw detected tick angles
# --------------------------------

debug = image.copy()

center_x, center_y = center

for angle in scale["tick_angles"]:

    theta = np.deg2rad(angle)

    x1 = int(
        center_x +
        radius * 0.70 * np.cos(theta)
    )

    y1 = int(
        center_y +
        radius * 0.70 * np.sin(theta)
    )

    x2 = int(
        center_x +
        radius * 0.95 * np.cos(theta)
    )

    y2 = int(
        center_y +
        radius * 0.95 * np.sin(theta)
    )

    cv2.line(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        5
    )

# Draw center
cv2.circle(
    debug,
    (center_x, center_y),
    10,
    (255, 0, 0),
    -1
)

output_path = "data/processed/scale_detection.jpg"

cv2.imwrite(
    output_path,
    debug
)

print(f"\n✅ Debug image saved to:")
print(output_path)