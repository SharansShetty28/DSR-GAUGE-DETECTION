import cv2
import numpy as np

from gauge_config import GAUGE_CONFIG


def read_analog_gauge(gauge_image, gauge_name=None):
    """
    Detect the gauge circle, estimate the needle pivot,
    detect the needle direction, and convert the angle
    into the configured gauge value.
    """

    if gauge_image is None:
        return {
            "value": None,
            "unit": None,
            "needle_angle": None,
            "confidence": 0.0
        }

    # --------------------------------
    # Get configuration for this gauge
    # --------------------------------
    gauge_config = GAUGE_CONFIG.get(gauge_name)

    if gauge_config is None:
        return {
            "value": None,
            "unit": None,
            "needle_angle": None,
            "confidence": 0.0,
            "error": f"Unknown gauge: {gauge_name}"
        }

    # --------------------------------
    # 1. Convert to grayscale
    # --------------------------------
    gray = cv2.cvtColor(
        gauge_image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------
    # 2. Detect gauge circle
    # --------------------------------
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=50,
        minRadius=100,
        maxRadius=1500
    )

    if circles is None:
        return {
            "value": None,
            "unit": gauge_config["unit"],
            "needle_angle": None,
            "confidence": 0.0
        }

    circles = np.round(
        circles[0]
    ).astype(int)

    height, width = gauge_image.shape[:2]

    valid_circles = []

    for x, y, r in circles:

        if (
            x - r > 0
            and y - r > 0
            and x + r < width
            and y + r < height
        ):
            valid_circles.append(
                (x, y, r)
            )

    if not valid_circles:
        return {
            "value": None,
            "unit": gauge_config["unit"],
            "needle_angle": None,
            "confidence": 0.0
        }

    # --------------------------------
    # 3. Select gauge circle
    # --------------------------------
    image_center_x = width / 2
    expected_y = height * 0.35

    def circle_score(c):
        x, y, r = c

        distance = (
            ((x - image_center_x) ** 2)
            + ((y - expected_y) ** 2)
        ) ** 0.5

        return distance - (r * 0.2)

    center_x, center_y, radius = min(
        valid_circles,
        key=circle_score
    )

    # Include complete gauge scale
    radius = int(radius * 1.25)

    # --------------------------------
    # 4. Detect actual needle pivot
    # --------------------------------

    gray_pivot = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    search_radius = int(
        radius * 0.20
    )

    best_pivot = None
    best_dark_score = -1

    for py in range(
        int(center_y - search_radius),
        int(center_y + search_radius),
        5
    ):

        for px in range(
            int(center_x - search_radius),
            int(center_x + search_radius),
            5
        ):

            if px < 0 or px >= width:
                continue

            if py < 0 or py >= height:
                continue

            patch_radius = 12

            patch = gray_pivot[
                max(0, py - patch_radius):
                min(height, py + patch_radius),

                max(0, px - patch_radius):
                min(width, px + patch_radius)
            ]

            if patch.size == 0:
                continue

            dark_ratio = np.mean(
                patch < 90
            )

            if dark_ratio > best_dark_score:

                best_dark_score = dark_ratio

                best_pivot = (
                    px,
                    py
                )

    # Use detected pivot if available
    if best_pivot is not None:

        pivot_x, pivot_y = best_pivot

    else:

        pivot_x = center_x
        pivot_y = center_y

    # --------------------------------
    # 5. Detect needle direction
    # --------------------------------

    angles = np.arange(
        0,
        360,
        1
    )

    scores = []

    # Start slightly away from pivot
    start_distance = int(
        radius * 0.08
    )

    # Stop before outer scale
    end_distance = int(
        radius * 0.65
    )

    for angle in angles:

        theta = np.deg2rad(angle)

        dark_pixels = 0
        total_pixels = 0

        for distance in range(
            start_distance,
            end_distance,
            5
        ):

            x = int(
                pivot_x
                + distance * np.cos(theta)
            )

            y = int(
                pivot_y
                + distance * np.sin(theta)
            )

            if (
                x < 0
                or x >= width
                or y < 0
                or y >= height
            ):
                continue

            patch = gray[
                max(0, y - 3):
                min(height, y + 4),

                max(0, x - 3):
                min(width, x + 4)
            ]

            if patch.size == 0:
                continue

            dark_pixels += np.sum(
                patch < 80
            )

            total_pixels += patch.size

        if total_pixels > 0:

            score = (
                dark_pixels / total_pixels
            )

        else:

            score = 0

        scores.append(score)

    scores = np.array(scores)

    # --------------------------------
    # 6. Find strongest needle direction
    # --------------------------------

    best_index = int(
        np.argmax(scores)
    )

    needle_angle = float(
        angles[best_index]
    )

    needle_score = float(
        scores[best_index]
    )

    # --------------------------------
    # 7. Confidence
    # --------------------------------

    confidence = min(
        1.0,
        needle_score * 3
    )

    # --------------------------------
    # 8. Get scale calibration
    # --------------------------------

    scale_start_angle = gauge_config.get(
        "scale_start_angle"
    )

    scale_end_angle = gauge_config.get(
        "scale_end_angle"
    )

    min_value = gauge_config["min_value"]
    max_value = gauge_config["max_value"]

    # --------------------------------
    # 9. Convert angle to actual value
    # --------------------------------

    # If this gauge has not been calibrated yet,
    # return the detected needle information
    # without crashing.
    if (
        scale_start_angle is None
        or scale_end_angle is None
    ):

        return {
            "value": None,
            "unit": gauge_config["unit"],
            "needle_angle": needle_angle,
            "gauge_center": [
                int(pivot_x),
                int(pivot_y)
            ],
            "gauge_radius": int(radius),
            "confidence": round(
                confidence,
                3
            ),
            "error": (
                "Scale calibration not configured "
                "for this gauge"
            )
        }

    # Angular distance from minimum
    angular_distance = (
        needle_angle
        - scale_start_angle
    ) % 360

    # Total scale span
    scale_span = (
        scale_end_angle
        - scale_start_angle
    ) % 360

    # Convert needle angle to actual value
    if scale_span > 0:

        value = (
            min_value
            + (
                angular_distance
                / scale_span
            )
            * (
                max_value
                - min_value
            )
        )

        # Keep value within configured range
        value = max(
            min_value,
            min(
                max_value,
                value
            )
        )

        value = round(
            value,
            2
        )

    else:

        value = None

    # --------------------------------
    # 10. Return result
    # --------------------------------

    return {
        "value": value,
        "unit": gauge_config["unit"],
        "needle_angle": needle_angle,
        "gauge_center": [
            int(pivot_x),
            int(pivot_y)
        ],
        "gauge_radius": int(radius),
        "confidence": round(
            confidence,
            3
        )
    }