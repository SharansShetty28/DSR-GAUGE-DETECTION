import cv2
import math
import os

from gauge_detector import detect_gauge
from scale_detector import detect_scale


IMAGE = "data/raw/(26.1)Generator Clr CW InTemp.jpg"


image = cv2.imread(IMAGE)

if image is None:
    print("❌ Image not found")
    exit()


gauge = detect_gauge(image)

print("Gauge:")
print(gauge)

if not gauge["gauge_detected"]:
    print("❌ Gauge not detected")
    exit()


center = gauge["center"]
radius = gauge["radius"]

print("\nCenter:", center)
print("Radius:", radius)


scale = detect_scale(
    image,
    center,
    radius
)

print("\nScale:")
print(scale)


output = image.copy()

cx, cy = center

# Draw gauge center
cv2.circle(
    output,
    (int(cx), int(cy)),
    12,
    (0, 0, 255),
    -1
)

# Draw gauge boundary
cv2.circle(
    output,
    (int(cx), int(cy)),
    int(radius),
    (0, 255, 0),
    5
)


# Draw detected major angles
for angle in scale["major_angles"]:

    theta = math.radians(angle)

    inner_r = radius * 0.67
    outer_r = radius * 0.91

    x1 = int(
        cx + inner_r * math.cos(theta)
    )

    y1 = int(
        cy + inner_r * math.sin(theta)
    )

    x2 = int(
        cx + outer_r * math.cos(theta)
    )

    y2 = int(
        cy + outer_r * math.sin(theta)
    )

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        8
    )


os.makedirs(
    "data/processed/diagnostic",
    exist_ok=True
)

output_path = (
    "data/processed/diagnostic/"
    "generator_scale_debug.jpg"
)

cv2.imwrite(
    output_path,
    output
)

print("\n✅ Saved:")
print(output_path)