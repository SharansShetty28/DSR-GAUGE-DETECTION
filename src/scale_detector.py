import cv2
import numpy as np
import math


def circular_distance(a, b):
    diff = abs(a - b)
    return min(diff, 360 - diff)


def detect_scale(image, center, radius):

    cx, cy = map(int, center)
    radius = float(radius)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve tick visibility
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(gray, 40, 120)

    # ---------------------------------------------------------
    # ONLY SEARCH INSIDE THE GAUGE
    # ---------------------------------------------------------

    mask = np.zeros_like(edges)

    # Tick marks are normally inside the outer gauge boundary
    outer_r = int(radius * 0.88)
    inner_r = int(radius * 0.58)

    cv2.circle(
        mask,
        (cx, cy),
        outer_r,
        255,
        -1
    )

    cv2.circle(
        mask,
        (cx, cy),
        inner_r,
        0,
        -1
    )

    # Keep only the annular tick region
    ring_edges = cv2.bitwise_and(
        edges,
        mask
    )

    # ---------------------------------------------------------
    # HOUGH LINE DETECTION
    # ---------------------------------------------------------

    lines = cv2.HoughLinesP(
        ring_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(radius * 0.035),
        maxLineGap=8
    )

    if lines is None:
        return {
            "major_angles": [],
            "confidence": 0.0
        }

    candidates = []

    # ---------------------------------------------------------
    # CHECK EACH LINE
    # ---------------------------------------------------------

    for line in lines:

        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        length = math.sqrt(
            dx * dx + dy * dy
        )

        if length < radius * 0.035:
            continue

        # Midpoint
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        # Distance of midpoint from gauge center
        mr = math.sqrt(
            (mx - cx) ** 2 +
            (my - cy) ** 2
        )

        # Ignore lines outside tick region
        if mr < inner_r or mr > outer_r:
            continue

        # Angle from center to midpoint
        radial_angle = math.degrees(
            math.atan2(
                my - cy,
                mx - cx
            )
        )

        radial_angle %= 360

        # Line orientation
        line_angle = math.degrees(
            math.atan2(dy, dx)
        ) % 180

        # Expected radial line orientation
        expected = radial_angle % 180

        # Difference between line and radial direction
        diff = abs(
            line_angle - expected
        )

        diff = min(
            diff,
            180 - diff
        )

        # Tick marks should point approximately
        # toward the gauge center
        if diff > 18:
            continue

        candidates.append({
            "angle": radial_angle,
            "length": length,
            "radius": mr
        })

    # ---------------------------------------------------------
    # GROUP NEARBY ANGLES
    # ---------------------------------------------------------

    candidates.sort(
        key=lambda x: x["angle"]
    )

    groups = []

    for candidate in candidates:

        placed = False

        for group in groups:

            if circular_distance(
                candidate["angle"],
                group["angle"]
            ) < 4:

                group["items"].append(candidate)

                # Weighted average angle
                group["angle"] = np.mean(
                    [
                        x["angle"]
                        for x in group["items"]
                    ]
                ) % 360

                placed = True
                break

        if not placed:

            groups.append({
                "angle": candidate["angle"],
                "items": [candidate]
            })

    # ---------------------------------------------------------
    # SCORE EACH ANGLE
    # ---------------------------------------------------------

    scored = []

    for group in groups:

        items = group["items"]

        max_length = max(
            x["length"]
            for x in items
        )

        total_length = sum(
            x["length"]
            for x in items
        )

        scored.append({
            "angle": group["angle"],
            "max_length": max_length,
            "total_length": total_length
        })

    # ---------------------------------------------------------
    # MAJOR TICKS
    # ---------------------------------------------------------
    #
    # Major ticks are longer than ordinary ticks.
    #

    if not scored:

        return {
            "major_angles": [],
            "confidence": 0.0
        }

    lengths = np.array(
        [x["max_length"] for x in scored]
    )

    # Adaptive threshold
    #
    # We don't use one fixed pixel value because
    # gauge images have different sizes.
    #

    threshold = max(
        radius * 0.055,
        np.percentile(lengths, 65)
    )

    major = [
        x
        for x in scored
        if x["max_length"] >= threshold
    ]

    # ---------------------------------------------------------
    # REMOVE VERY CLOSE DUPLICATES
    # ---------------------------------------------------------

    major.sort(
        key=lambda x: x["max_length"],
        reverse=True
    )

    selected = []

    for item in major:

        angle = item["angle"]

        if all(
            circular_distance(
                angle,
                selected_angle
            ) >= 7
            for selected_angle in selected
        ):

            selected.append(angle)

    selected.sort()

    # ---------------------------------------------------------
    # LIMIT EXTREME FALSE POSITIVES
    # ---------------------------------------------------------

    if len(selected) > 12:

        selected_data = []

        for angle in selected:

            item = min(
                scored,
                key=lambda x:
                circular_distance(
                    x["angle"],
                    angle
                )
            )

            selected_data.append(item)

        selected_data.sort(
            key=lambda x: x["max_length"],
            reverse=True
        )

        selected = [
            x["angle"]
            for x in selected_data[:12]
        ]

        selected.sort()

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    if not selected:

        confidence = 0.0

    else:

        avg_length = np.mean([
            min(
                radius * 0.10,
                min(
                    scored,
                    key=lambda x:
                    circular_distance(
                        x["angle"],
                        angle
                    )
                )["max_length"]
            )
            for angle in selected
        ])

        confidence = min(
            avg_length / (radius * 0.08),
            1.0
        )

        confidence = round(
            float(confidence),
            2
        )

    return {
        "major_angles": [
            round(float(x), 1)
            for x in selected
        ],
        "confidence": confidence
    }