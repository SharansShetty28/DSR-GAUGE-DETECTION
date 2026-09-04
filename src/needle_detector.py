import cv2
import numpy as np
import math


def detect_needle(image, center, radius):
    """
    Detect the needle direction inside an automatically
    detected gauge.

    Args:
        image: OpenCV image
        center: [x, y] detected gauge center
        radius: detected gauge radius

    Returns:
        {
            "needle_detected": True/False,
            "needle_angle": angle,
            "confidence": confidence
        }
    """

    if image is None or center is None or radius is None:
        return {
            "needle_detected": False,
            "needle_angle": None,
            "confidence": 0.0
        }

    height, width = image.shape[:2]

    center_x, center_y = center

    # ---------------------------------------
    # 1. Grayscale
    # ---------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ---------------------------------------
    # 2. Search for needle directions
    # ---------------------------------------

    angles = np.arange(0, 360, 1)

    scores = []

    # Start outside the center/pivot
    start_distance = max(
        20,
        int(radius * 0.08)
    )

    # Stop before the gauge edge
    end_distance = int(
        radius * 0.72
    )

    for angle in angles:

        theta = math.radians(angle)

        dark_pixels = 0
        total_pixels = 0

        for distance in range(
            start_distance,
            end_distance,
            4
        ):

            x = int(
                center_x +
                distance * math.cos(theta)
            )

            y = int(
                center_y +
                distance * math.sin(theta)
            )

            if (
                x < 0
                or x >= width
                or y < 0
                or y >= height
            ):
                continue

            # Small area around the line
            patch = gray[
                max(0, y - 2):
                min(height, y + 3),

                max(0, x - 2):
                min(width, x + 3)
            ]

            if patch.size == 0:
                continue

            dark_pixels += np.sum(
                patch < 80
            )

            total_pixels += patch.size

        if total_pixels > 0:
            score = (
                dark_pixels /
                total_pixels
            )
        else:
            score = 0

        scores.append(score)

    scores = np.array(scores)

    # ---------------------------------------
    # 3. Find strongest direction
    # ---------------------------------------

    best_index = int(
        np.argmax(scores)
    )

    needle_angle = float(
        angles[best_index]
    )

    needle_score = float(
        scores[best_index]
    )

    # ---------------------------------------
    # 4. Confidence
    # ---------------------------------------

    confidence = min(
        1.0,
        needle_score * 3
    )

    return {
        "needle_detected": confidence > 0.15,
        "needle_angle": needle_angle,
        "confidence": round(
            confidence,
            3
        )
    }