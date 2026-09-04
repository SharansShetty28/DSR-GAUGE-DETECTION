import cv2
import numpy as np


def detect_gauge(image):
    """
    Automatically detect the main analog gauge.

    Step 1:
    Detect only the gauge circle.
    This function does NOT detect the needle or reading yet.

    Returns:
        {
            "gauge_detected": True/False,
            "center": [x, y],
            "radius": r,
            "confidence": ...
        }
    """

    if image is None:
        return {
            "gauge_detected": False,
            "center": None,
            "radius": None,
            "confidence": 0.0
        }

    height, width = image.shape[:2]

    # ---------------------------------------
    # 1. Convert to grayscale
    # ---------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        1.5
    )

    # ---------------------------------------
    # 2. Image dimensions
    # ---------------------------------------

    min_dimension = min(
        height,
        width
    )

    # Allow both small and large gauges
    min_radius = max(
        40,
        int(min_dimension * 0.08)
    )

    max_radius = int(
        min_dimension * 0.55
    )

    # ---------------------------------------
    # 3. Detect circular candidates
    # ---------------------------------------

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(
            60,
            int(min_dimension * 0.15)
        ),
        param1=100,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:

        return {
            "gauge_detected": False,
            "center": None,
            "radius": None,
            "confidence": 0.0
        }

    circles = np.round(
        circles[0]
    ).astype(int)

    # ---------------------------------------
    # 4. Edge image
    # ---------------------------------------

    edge_image = cv2.Canny(
        gray,
        50,
        150
    )

    candidates = []

    # ---------------------------------------
    # 5. Score every circle
    # ---------------------------------------

    for x, y, r in circles:

        # -----------------------------------
        # IMPORTANT:
        # Do NOT require the entire circle
        # to be inside the image.
        #
        # A gauge can be slightly cropped.
        # -----------------------------------

        allowed_margin = int(
            r * 0.15
        )

        if (
            x + r < -allowed_margin
            or
            x - r > width + allowed_margin
            or
            y + r < -allowed_margin
            or
            y - r > height + allowed_margin
        ):
            continue

        # -----------------------------------
        # Circular edge strength
        # -----------------------------------

        angles = np.arange(
            0,
            360,
            3
        )

        edge_hits = 0
        samples = 0

        for angle in angles:

            theta = np.deg2rad(
                angle
            )

            px = int(
                x +
                r * np.cos(theta)
            )

            py = int(
                y +
                r * np.sin(theta)
            )

            if (
                0 <= px < width
                and
                0 <= py < height
            ):

                samples += 1

                patch = edge_image[
                    max(0, py - 3):
                    min(height, py + 4),

                    max(0, px - 3):
                    min(width, px + 4)
                ]

                if np.any(
                    patch > 0
                ):
                    edge_hits += 1

        if samples == 0:
            continue

        edge_score = (
            edge_hits /
            samples
        )

        # -----------------------------------
        # Gauge size score
        #
        # Prefer a real large gauge circle
        # over small random circles.
        # -----------------------------------

        size_score = min(
            1.0,
            r /
            (min_dimension * 0.40)
        )

        # -----------------------------------
        # Center score
        #
        # Only a small influence.
        #
        # We DON'T assume the gauge must be
        # exactly in the image center.
        # -----------------------------------

        image_center_x = width / 2
        image_center_y = height / 2

        center_distance = np.sqrt(
            (x - image_center_x) ** 2
            +
            (y - image_center_y) ** 2
        )

        max_distance = np.sqrt(
            image_center_x ** 2
            +
            image_center_y ** 2
        )

        center_score = max(
            0.0,
            1.0 -
            (
                center_distance /
                max_distance
            )
        )

        # -----------------------------------
        # Final score
        # -----------------------------------

        score = (
            edge_score * 0.55
            +
            size_score * 0.35
            +
            center_score * 0.10
        )

        candidates.append(
            (
                score,
                x,
                y,
                r,
                edge_score
            )
        )

    # ---------------------------------------
    # 6. No candidates
    # ---------------------------------------

    if not candidates:

        return {
            "gauge_detected": False,
            "center": None,
            "radius": None,
            "confidence": 0.0
        }

    # ---------------------------------------
    # 7. Select best candidate
    # ---------------------------------------

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, x, y, r, edge_score = (
        candidates[0]
    )

    # ---------------------------------------
    # 8. Confidence
    # ---------------------------------------

    confidence = min(
        1.0,
        float(best_score)
    )

    # ---------------------------------------
    # 9. Return result
    # ---------------------------------------

    return {
        "gauge_detected": True,

        "center": [
            int(x),
            int(y)
        ],

        "radius": int(r),

        "confidence": round(
            confidence,
            3
        )
    }