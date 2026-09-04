import cv2
import math
from analog_reader import read_analog_gauge


image_path = "data/raw/(14)AE LO Cooler Inlet Temp.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Could not load image")
    exit()

print("Image loaded:", image.shape)

result = read_analog_gauge(
    image,
    gauge_name="AE LO Cooler Inlet Temp"
)

print("\nGauge detection result:")
print(result)


# -----------------------------------
# Draw detected gauge and needle
# -----------------------------------

if "gauge_center" in result:

    center_x, center_y = result["gauge_center"]
    radius = result["gauge_radius"]
    angle = result["needle_angle"]

    output = image.copy()

    # Draw gauge circle
    cv2.circle(
        output,
        (center_x, center_y),
        radius,
        (0, 255, 0),
        8
    )

    # Draw center
    cv2.circle(
        output,
        (center_x, center_y),
        15,
        (0, 0, 255),
        -1
    )

    # Convert angle to radians
    theta = math.radians(angle)

    # Needle endpoint
    needle_length = int(radius * 0.75)

    end_x = int(
        center_x + needle_length * math.cos(theta)
    )

    end_y = int(
        center_y + needle_length * math.sin(theta)
    )

    # Draw detected needle direction
    cv2.line(
        output,
        (center_x, center_y),
        (end_x, end_y),
        (255, 0, 0),
        12
    )

    output_path = "data/processed/needle_detection.jpg"

    cv2.imwrite(output_path, output)

    print("\nResult saved to:")
    print(output_path)